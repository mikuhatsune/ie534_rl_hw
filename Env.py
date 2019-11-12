# Parallel simulation environment interfaces

import gym
import multiprocessing as mp

try:
    import cv2
except:
    pass
import numpy as np

def preprocess_image(x):
    # original images from atari_py are 210 x 160 x 3 uint8 array
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x[34:160+34], (84, 84), interpolation=cv2.INTER_AREA)
    #x = x.transpose((2, 0, 1))  # H x W x C => C x H x W
    return x[None]

def process_env(pid, conn, name, frame_skip, frame_stack):
    print ("[pid %d] start" % pid)
    env = gym.make(name)
    env.seed(pid)  # seed randomness of env (if there's any)

    # add preprocessing step if it's Atari domain
    # observations are resized and stacked images, reward clipped to +/-1
    atari = len(env.observation_space.shape) == 3
    if pid == 0:
        if atari:
            from gym.spaces import Box
            org_sp = env.observation_space
            #print ('org_sp', org_sp.low, org_sp.high, org_sp.shape)
            #new_sp = Box(0, 255, (3*frame_stack, 84, 84))
            new_sp = Box(0, 255, (frame_stack, 84, 84))
            #print ('new_sp', new_sp.low, new_sp.high, new_sp.shape)
            conn.send((new_sp, env.action_space))
        else:
            conn.send((env.observation_space, env.action_space))

    if atari:
        def reset():
            env.reset()
            s, *_ = env.step(1) # 'fire' start
            for i in range(frame_skip):
                s, *_ = env.step(0)
            s = preprocess_image(s)
            s = np.tile(s, (frame_stack,1,1))
            lives = env.unwrapped.ale.lives()
            return s

        lives = 0
    else:
        def reset():
            return env.reset().astype(np.float32)

    while True:
        cmd, *params = conn.recv()
        if cmd == 'step':
            a, = params
            true_done = False
            if atari:
                r_raw, r, t = 0.0, 0.0, False
                # cat_image = True
                for i in range(frame_skip):
                    si, ri, ti, _ = env.step(a)
                    r_raw += ri
                    cat_image = i == frame_skip-1
                    # episodic life trick
                    new_lives = env.unwrapped.ale.lives()
                    if ti:
                        s, t, true_done = reset(), True, True
                    elif new_lives < lives and lives > 0:
                        t = True
                        si, _, _, _ = env.step(1)
                        cat_image = True
                        r = -0.5  # set a penalty for loss of life
                    lives = new_lives
                    if cat_image:
                        si = preprocess_image(si)
                        #s = np.concatenate((s[3:,:,:], si), 0)
                        s = np.concatenate((s[1:,:,:], si), 0)
                    cat_image = False

                # reward clipping trick
                if r_raw > 0:
                    r += 1.0
                elif r_raw < 0:
                    r += -1.0
            else:
                s, r, t, _ = env.step(a)
                r_raw, true_done = r, t
                s = s.astype(np.float32)
                if t: s = reset()  # override existing terminal state
            conn.send((s, r, t, (r_raw, true_done)))
        elif cmd == 'reset':
            s = reset()
            conn.send(s)
        else: #if cmd == 'end':
            break

    env.close()
    conn.close()
    print ("[pid %d] exit" % pid)


class ParallelEnv:
    def __init__(self, name, nproc, frame_skip, frame_stack):
        self.name = name
        self.nproc = nproc
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(nproc)])
        self.procs = [mp.Process(target=process_env,
                                 args=(i, self.child_conns[i], name, frame_skip, frame_stack))
                      for i in range(nproc)]
        for p in self.procs: p.start()
        self.obs_space, self.act_space = self.parent_conns[0].recv()

    def step(self, actions):
        for conn, a in zip(self.parent_conns, actions):
            conn.send(('step', a))
        ret = [conn.recv() for conn in self.parent_conns]
        return ret
        #return list(zip(*ret))

    def step_async(self, actions):
        for conn, a in zip(self.parent_conns, actions):
            conn.send(('step', a))
    
    def get_results(self):
        return [conn.recv() for conn in self.parent_conns]

    def reset(self):
        for conn in self.parent_conns:
            conn.send(('reset',))
        return [conn.recv() for conn in self.parent_conns]

    def close(self):
        for p, conn in zip(self.procs, self.parent_conns):
            conn.send(('end',))
            p.join()
            conn.close()



class LocalEnv:
    def __init__(self, name, nproc, frame_skip, frame_stack, save_img_dir=None):
        self.name = name
        self.nproc = nproc
        self.frame_skip, self.frame_stack = frame_skip, frame_stack

        self.envs = [gym.make(name) for _ in  range(nproc)]
        for e, env in enumerate(self.envs): env.seed(e)

        env = self.envs[0]
        self.atari = len(env.observation_space.shape) == 3
        if self.atari:
            from gym.spaces import Box
            org_sp = env.observation_space
            new_sp = Box(0, 255, (frame_stack, 84, 84))
            self.obs_space, self.act_space = new_sp, env.action_space

            self.lives = [0] * nproc
            self.s = [None] * nproc

            def reset(e):
                self.envs[e].reset()
                s, *_ = self.envs[e].step(1) # 'fire' start
                for i in range(frame_skip):
                    s, *_ = self.envs[e].step(0)
                s = preprocess_image(s)
                s = np.tile(s, (frame_stack,1,1))
                self.lives[e] = self.envs[e].unwrapped.ale.lives()
                self.s[e] = s
                return s
        else:
            self.obs_space, self.act_space = env.observation_space, env.action_space

            def reset(e):
                return self.envs[e].reset().astype(np.float32)

        self.reset_e = reset
        self.save_img_dir = save_img_dir
        if save_img_dir:
            self.img_id = 0

    def step(self, actions):
        self.ret = []
        for e, a in enumerate(actions):
            env = self.envs[e]
            if self.atari:
                true_done = False
                r_raw, r, t = 0.0, 0.0, False
                for i in range(self.frame_skip):
                    si, ri, ti, _ = env.step(a)
                    r_raw += ri
                    cat_image = i == self.frame_skip-1
                    # episodic life trick
                    new_lives = env.unwrapped.ale.lives()
                    if ti:
                        s, t, true_done = self.reset_e(e), True, True
                    elif new_lives < self.lives[e] and self.lives[e] > 0:
                        t = True
                        si, _, _, _ = env.step(1)
                        cat_image = True
                        r = -0.5  # set a penalty for loss of life
                    self.lives[e] = new_lives
                    if cat_image:
                        if self.save_img_dir:
                            cv2.imwrite(self.save_img_dir + '%03d.jpg' % self.img_id, si[:,:,::-1])
                            print ('write image %03d.jpg' % self.img_id)
                            self.img_id += 1
                        si = preprocess_image(si)
                        s = np.concatenate((self.s[e][1:,:,:], si), 0)
                    cat_image = False

                # reward clipping trick
                if r_raw > 0:
                    r += 1.0
                elif r_raw < 0:
                    r += -1.0
                self.s[e] = s
            else:
                s, r, t, _ = env.step(a)
                r_raw, true_done = r, t
                s = s.astype(np.float32)
                if t: s = self.reset_e(e)  # override existing terminal state

            self.ret.append( (s, r, t, (r_raw, true_done)) )
        return self.ret

    def step_async(self, actions):
        self.step(actions)
    
    def get_results(self):
        return self.ret

    def reset(self):
        return [self.reset_e(e) for e, env in enumerate(self.envs)]

    def close(self):
        for env in self.envs: env.close()