# Main file: train an agent with specific algorithm on specific environment
if __name__ == '__main__':

    import torch
    import Env, Algo
    import numpy as np
    import argparse, os
    from time import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=2000000, help='# training iterations')
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'a2c'])
    parser.add_argument('--nproc', type=int, default=2, help='# parallel processes')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train_freq', type=int, default=2, help='train every train_freq iterations')
    parser.add_argument('--train_start', type=int, default=5000, help='train start iteration')
    parser.add_argument('--batch_size', type=int, default=32, help='SGD batch size')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor (or gamma)')
    parser.add_argument('--replay_size', type=int, default=1000000, help='for DQN, replay buffer size')
    parser.add_argument('--target_update', type=int, default=2500, help='for DQN, update target net every target_update grad steps')
    # Difference between -v0, -v4, Deterministic, NoFrameSkip, etc. on https://github.com/openai/gym/issues/1280
    parser.add_argument('--frame_skip', type=int, default=1, help='for Atari, skip frames by repeating action, useful for NoFrameSkip-v4 envionments')
    parser.add_argument('--frame_stack', type=int, default=4, help='for Atari, stack 4 consecutive frames together as observations')
    parser.add_argument('--eps_decay', type=int, default=200000, help='for DQN, epsilon-greedy exploration over eps_decay iterations')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='for A2C, coefficient of entropy')
    parser.add_argument('--value_coef', type=float, default=0.5, help='for A2C, coefficient of value function loss')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--checkpoint_freq', type=int, default=500000)
    parser.add_argument('--save_dir', type=str, default='breakout/')
    parser.add_argument('--log', type=str, default='log.txt', help='the log file name, appended to save_dir')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--parallel_env', type=int, default=1)

    args = parser.parse_args()

    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    try: os.makedirs(args.save_dir)
    except: pass

    if len(args.log):
        # custom print function, line buffering, bad coding habit but works...
        log_file = open(args.save_dir + args.log, 'w', buffering=1)
        def logprint(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log_file)

    else:
        logprint = print

    logprint (args)

    # create simulation environment
    if args.parallel_env:
        env = Env.ParallelEnv(args.env, args.nproc, args.frame_skip, args.frame_stack)
    else:
        env = Env.LocalEnv(args.env, args.nproc, args.frame_skip, args.frame_stack)
    logprint ('observation space:', env.obs_space)
    logprint ('action space:', env.act_space)
    # logprint ()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logprint ('running on device', device)

    # create an instance of the agent and corresponding algorithm
    if args.algo == 'dqn':
        algo = Algo.DQN(
            env.obs_space, env.act_space,
            args.lr, args.replay_size, args.batch_size, args.discount, args.target_update, args.eps_decay,
            device)
    elif args.algo == 'a2c':
        assert args.nproc * args.train_freq == args.batch_size
        algo = Algo.ActorCritic(
            env.obs_space, env.act_space,
            args.lr, args.nproc, args.train_freq, args.discount, args.ent_coef, args.value_coef,
            device)

    if len(args.load):
        algo.load(args.load)
        logprint ("load pretrained model from", args.load)

    # initialize some statistics
    episode_lens = [0] * env.nproc
    avg_episode_len = float('nan')
    episode_rewards = [0] * env.nproc
    avg_episode_reward = float('nan')
    raw_episode_rewards = [0] * env.nproc
    avg_raw_episode_reward = float('nan')

    def exp_moving_average(x, new):
        # x != x means x is nan
        if x == x:
            return 0.9 * x + 0.1 * new
        else:
            return new

    num_env_steps = 0
    loss = 0
    n_episodes = 0

    # initial observations (or states) from game reset
    obses = env.reset()
    logprint ('obses on reset:', len(obses), 'x', obses[0].shape, obses[0].dtype)
    actions = algo.act(obses)
    results = env.step(actions)
    new_obses = next(zip(*results))
    new_actions = algo.act(new_obses)

    time_start = time()

    # import cv2

    # set up SIGTERM and SIGKILL handler to save the model
    import sys, signal
    sig_reenter = False
    def sigterm_handler(signal, frame):
        global sig_reenter
        if not sig_reenter:
            sig_reenter = True
            algo.save(args.save_dir + str(it) + '.pth')
            logprint ('sigterm received, save checkpoint to', args.save_dir + str(it) + '.pth')
            env.close()
            if len(args.log): log_file.close()
        else:
            logprint ('forced exit')
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)


    # train for 'niter' iterations
    for it in range(1, args.niter):
        # results is the format of list of nproc (new_state, reward, terminal) tuple
        # i.e. [(news_1, r_1, t_1), (news_2, r_2, t_2), ..., (news_nproc, r_nproc, t_nproc)]
        env.step_async(new_actions)
        # actions = algo.act(obses)
        # results = env.step(actions)

        algo.observe(obses, actions, results)
        if it % args.train_freq == 0 and it >= args.train_start:
            loss = algo.train()

        obses, actions = new_obses, new_actions

        for i, (sn,r,t, (r_raw,true_done)) in enumerate(results):
            episode_lens[i] += 1
            episode_rewards[i] += r
            if t:
                avg_episode_len = exp_moving_average(avg_episode_len, episode_lens[i])
                episode_lens[i] = 0
                avg_episode_reward = exp_moving_average(avg_episode_reward, episode_rewards[i])
                episode_rewards[i] = 0
                n_episodes += 1
            raw_episode_rewards[i] += r_raw
            if true_done:
                avg_raw_episode_reward = exp_moving_average(avg_raw_episode_reward, raw_episode_rewards[i])
                raw_episode_rewards[i] = 0

        num_env_steps += env.nproc
        if it % args.print_freq == 0:
            # cv2.imwrite('%d.jpg'%it, np.concatenate(obses[0]))
            time_now = time()
            time_passed = time_now - time_start
            time_remain = time_passed * (args.niter-1 - it) / it
            time_last = time_now
            time_passed = '%02d:%02d' % (time_passed // 60, time_passed % 60)
            time_remain = '%02d:%02d' % (time_remain // 60, time_remain % 60)
            #logprint ("iter %6d  loss %6.2f  n_eps %5d  avg_ep_len %6.1f  avg_ep_rew %6.2f  " \
            #       "avg_raw_ep_rew %6.2f  env_steps %6d  time %s remain %s" %
            logprint ("iter %6d |loss %6.2f |n_ep %5d |ep_len %6.1f |ep_rew %6.2f " \
                   "|raw_ep_rew %6.2f |env_step %6d |time %s rem %s" %
                (it, loss, n_episodes, avg_episode_len, avg_episode_reward,
                 avg_raw_episode_reward, num_env_steps, time_passed, time_remain))

        if it % args.checkpoint_freq == 0:
            logprint ('save checkpoint to', args.save_dir + str(it) + '.pth')
            algo.save(args.save_dir + str(it) + '.pth')

        results = env.get_results()
        new_obses = next(zip(*results))
        new_actions = algo.act(new_obses)

    logprint ('save checkpoint to', args.save_dir + str(it) + '.pth')
    algo.save(args.save_dir + str(it) + '.pth')
    env.close()

    if len(args.log): log_file.close()
