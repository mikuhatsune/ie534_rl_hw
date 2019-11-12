# Replay buffer

import random
from collections import deque
import numpy as np
import torch

class NaiveReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.cur_batch = None

    # saves a transition
    def add(self, record):
        self.memory.append(record)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # sample efficiently a batch onto a device for pytorch
    def sample_torch(self, batch_size, device=None):
        if self.cur_batch is not None: return self.cur_batch

        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*transitions)

        states = torch.as_tensor(np.array(states), device=device)
        actions = torch.as_tensor(np.array(actions, dtype=np.int64), device=device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32), device=device)

        non_terminal_mask = torch.as_tensor(np.array([s is not None for s in next_states], 
            dtype=np.bool if hasattr(torch, 'bool') else np.uint8), device=device)
        non_terminal_next_states = torch.as_tensor(np.array([s for s in next_states if s is not None]), device=device)

        # for Atari, dtype: uint8, int64, float32, bool, uint8
        ret = [states, actions, rewards, non_terminal_mask, non_terminal_next_states]
        # if device: ret = [d.to(device) for d in ret]
        self.cur_batch = ret
        return self.cur_batch

    def __len__(self):
        return len(self.memory)
