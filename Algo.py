# RL algorithms: DQN and simplified Actor-Critic

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import Model, Replay
import random
from collections import namedtuple

class DQN:
    def __init__(self, obs_space, act_space, lr=1e-4, replay_size=1000000, batch_size=32,
                 discount=0.99, target_update=2500, eps_decay=500000, device=None):
        self.obs_space, self.act_space = obs_space, act_space
        self.batch_size, self.discount, self.target_update = batch_size, discount, target_update
        self.device = device

        if len(obs_space.shape) == 1:
            self.q_func = Model.TwoLayerFCNet(n_in=obs_space.shape[0], n_out=act_space.n).to(device)
            self.target_q_func = Model.TwoLayerFCNet(n_in=obs_space.shape[0], n_out=act_space.n).to(device)
            self.state_dtype = torch.float32
        elif len(obs_space.shape) == 3:
            self.q_func = Model.SimpleCNN(n_in=obs_space.shape, n_out=act_space.n).to(device)
            self.target_q_func = Model.SimpleCNN(n_in=obs_space.shape, n_out=act_space.n).to(device)
            self.state_dtype = torch.uint8
        else:
            print ("observation shape not supported:", obs_space.shape)
            raise
        self.q_func.train()
        self.target_q_func.train()

        print ('parameters to optimize:',
            [(name, p.shape, p.requires_grad) for name,p in self.q_func.named_parameters()],
            '\n')
        # self.optimizer = torch.optim.RMSprop(self.q_func.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=lr, betas=(0.9,0.99), eps=1e-8)

        # number of action steps done
        self.num_act_steps = 0
        #self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.01, eps_decay
        self.eps_decay = eps_decay
        self.num_train_steps = 0
        self.double_q = True

        self.replay = Replay.NaiveReplay(replay_size)

    def compute_epsilon(self):

        ### <<< Your Code Here
        # linearly anneal from 1 to 0.01 in eps_decay steps
        eps = '???' + ('???' - '???') * self.num_act_steps / self.eps_decay
        eps = max('???', eps)
        ### Your Code Ends >>>

        return eps

    def act(self, obses):
        obses = torch.as_tensor(obses, device=self.device, dtype=self.state_dtype)

        # if self.num_act_steps < self.eps_decay // 4:
        #     eps = 1.0 + (0.1 - 1.0) * self.num_act_steps / (self.eps_decay // 4)
        # elif self.num_act_steps < self.eps_decay:
        #     eps = 0.1 + (0.01 - 0.1) * self.num_act_steps / self.eps_decay
        # else:
        #     eps = 0.01
        eps = self.compute_epsilon()
        self.num_act_steps += 1

        with torch.no_grad():
            greedy_actions = self.q_func(obses).max(1)[1].tolist()
        actions = []
        for i in range(len(obses)):
            if random.random() < eps:
                a = random.randrange(self.act_space.n)
            else:
                a = greedy_actions[i]
            actions.append(a)
        return actions

    def observe(self, obses, actions, transitions):
        for s,a,(sn,r,t,_) in zip(obses, actions, transitions):
            if t: # (s,a) leads to a terminal state
                sn = None
            self.replay.add((s,a,r,sn))
        if len(self.replay) > self.batch_size and self.replay.cur_batch is None:
            self.replay.sample_torch(self.batch_size, self.device)

    def train(self):
        #self.replay.get_current_batch(self.batch_size, self.device)
        state_batch, action_batch, reward_batch, non_terminal_mask, non_terminal_next_states = self.replay.cur_batch
        self.replay.cur_batch = None

        q_values = self.q_func(state_batch)[torch.arange(self.batch_size), action_batch]

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_q_values = self.target_q_func(non_terminal_next_states).detach()

        ### <<< Your Code Here
        if self.double_q:
            # Hint: try using self.q_func(non_terminal_next_states).argmax(1)
            next_state_values[non_terminal_mask] = next_q_values[
                torch.arange(len(next_q_values)), '???']
        else:
            next_state_values[non_terminal_mask] = next_q_values.max(1)[0]

        target_q_values = '???' * self.discount + '???'

        loss = '???'  # Hint: try using q_values, target_q_values
        ### Your Code Ends >>>

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_func.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), 10)
        self.optimizer.step()
        # The ugly patch for using Adam on BlueWaters...
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if state['step'] >= 1022:
                    state['step'] = 1022

        self.num_train_steps += 1
        if self.num_train_steps % self.target_update == 0:
            self.target_q_func.load_state_dict(self.q_func.state_dict())
        return loss.item()

    def save(self, path):
        torch.save([self.q_func.state_dict(), self.target_q_func.state_dict(), self.optimizer.state_dict()], path)

    def load(self, path):
        s1, s2, s3 = torch.load(path, map_location=self.device)
        self.q_func.load_state_dict(s1)
        self.target_q_func.load_state_dict(s2)
        self.optimizer.load_state_dict(s3)


# Buffer = namedtuple('Buffer', ('state', 'action', 'reward', 'terminal', 'last_state'))

class ActorCritic:
    def __init__(self, obs_space, act_space, lr=0.001, nproc=1, seg_len=16, discount=0.99,
                 entropy_coef=0.01, value_coef=0.5, device=None, shared_net=True):
        self.obs_space, self.act_space = obs_space, act_space
        self.discount, self.entropy_coef, self.value_coef = discount, entropy_coef, value_coef
        self.nproc, self.seg_len = nproc, seg_len
        self.device = device

        if len(obs_space.shape) == 1:
            shared_net = False
            self.actor = Model.TwoLayerFCNet(n_in=obs_space.shape[0], n_out=act_space.n).to(device)
            self.critic = Model.TwoLayerFCNet(n_in=obs_space.shape[0], n_out=1).to(device)
            self.state_dtype = torch.float32
        elif len(obs_space.shape) == 3:
            if shared_net:
                self.actor_and_critic = Model.SimpleCNN(n_in=obs_space.shape, n_out=[act_space.n, 1]).to(device)
                self.actor = lambda obses: self.actor_and_critic(obses, head=0)
                self.critic = lambda obses: self.actor_and_critic(obses, head=1)
            else:
                self.actor = Model.SimpleCNN(n_in=obs_space.shape, n_out=act_space.n).to(device)
                self.critic = Model.SimpleCNN(n_in=obs_space.shape, n_out=1).to(device)
            self.state_dtype = torch.uint8
        else:
            print ("observation shape not supported:", obs_space.shape)
            raise

        self.shared_net = shared_net
        # Other people like to use RMSprop for policy gradient... but neither of them is perfect
        # if shared_net:
        #     self.optimizer = torch.optim.RMSprop(self.actor_and_critic.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        # else:
        #     self.optimizer = torch.optim.RMSprop(list(self.actor.parameters()) + list(self.critic.parameters()),
        #                                       lr=lr, alpha=0.99, eps=1e-5)

        if shared_net:
            self.optimizer = torch.optim.Adam(self.actor_and_critic.parameters(), lr=lr,
                betas=(0.9,0.99), eps=1e-6)
            print ('shared net = True, parameters to optimize:',
                [(name, p.shape, p.requires_grad) for name,p in self.actor_and_critic.named_parameters()],
                '\n')
        else:
            self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr,
                betas=(0.9,0.99), eps=1e-6)
            print ('shared net = False, parameters to optimize:',
                [(name, p.shape, p.requires_grad) for name,p in list(self.actor.named_parameters()) + list(self.critic.named_parameters())],
                '\n')


        self.states = torch.empty((nproc, seg_len) + obs_space.shape, dtype=self.state_dtype)
        self.actions = torch.empty((nproc, seg_len) + act_space.shape, dtype=torch.long)
        self.rewards = torch.empty((nproc, seg_len), dtype=torch.float32)
        # self.advantages = torch.empty((nproc, seg_len), dtype=torch.float32)
        self.step = 0
        self.last_reset = [0] * nproc

    def act(self, obses):
        obses = torch.as_tensor(obses, device=self.device, dtype=self.state_dtype)
        dist = Categorical(logits=self.actor(obses))
        actions = dist.sample()
        return actions.tolist()

    def observe(self, obses, actions, transitions):
        nproc = len(obses)
        # self.states[:, self.step] = obses
        # self.actions[:, self.step] = actions
        for i,(sn,r,terminal,_) in enumerate(transitions):
            self.states[i, self.step] = torch.as_tensor(obses[i], dtype=self.state_dtype)
            self.actions[i, self.step] = actions[i]
            self.rewards[i, self.step] = r
            R = 0.0
            if self.step == self.seg_len-1 and not terminal:
                R = self.critic(torch.as_tensor([sn], dtype=self.state_dtype, device=self.device)).item()
                terminal = True
            if terminal:
                # compute discounted reward backward along the trajectory
                for t in range(self.step, self.last_reset[i]-1, -1):
                    R = self.rewards[i, t] + self.discount * R
                    self.rewards[i, t] = R
                self.last_reset[i] = (self.step + 1) % self.seg_len
        self.step = (self.step + 1) % self.seg_len

    def train(self):
        assert self.step == 0

        states = self.states.view(-1, *self.obs_space.shape).to(self.device)
        actions = self.actions.view(-1, *self.act_space.shape).to(self.device)
        rewards = self.rewards.view(-1).to(self.device)

        values = self.critic(states).squeeze(-1)
        advantage = rewards - values
        # advantage_detach = torch.clamp(advantage.detach(), -1, 1)
        # advantage = advantage.clamp(-2, 2).detach() + advantage - advantage.detach()
        # advantage = (advantage - advantage.mean().detach())/ (1.0 + advantage.std().detach())
        advantage = advantage / (1.0 + advantage.std().detach())
        advantage_detach = advantage.detach() #* self.advantage_normalizer

        # compute losses
        logits = self.actor(states)
        dist = Categorical(logits=logits)

        ### <<< Your Code Here
        # use 'advantage_detach', 'dist' and 'actions' to compute this; double-check the sign of your expression!
        policy_loss = '???'

        # use the entropy function of 'dist' to compute this
        entropy_loss = '???'

        # value loss is the squared loss on variable 'advantage'
        value_loss = '???'
        ### Your Code Ends >>>

        loss = -policy_loss - self.entropy_coef * entropy_loss + self.value_coef * value_loss

        # print (policy_loss.item(), entropy_loss.item(), value_loss.item())

        # do gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        if self.shared_net:
            torch.nn.utils.clip_grad_norm_(self.actor_and_critic.parameters(), 0.5)
        else:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()
        # The ugly patch for using Adam on BlueWaters...
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if state['step'] >= 1022:
                    state['step'] = 1022

        return loss.item()

    def save(self, path):
        if self.shared_net:
            torch.save([self.actor_and_critic.state_dict(), self.optimizer.state_dict()], path)
        else:
            torch.save([self.actor.state_dict(), self.critic.state_dict(), self.optimizer.state_dict()], path)

    def load(self, path):
        states = torch.load(path, map_location=self.device)
        if self.shared_net:
            self.actor_and_critic.load_state_dict(states[0])
        else:
            self.actor.load_state_dict(states[0])
            self.critic.load_state_dict(states[1])
        self.optimizer.load_state_dict(states[-1])
