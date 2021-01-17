import random
import numpy as np
import torch
import hickle
import os
from loguru import logger

import lz4.frame
import cloudpickle as pickle

def pack(data):
    data = pickle.dumps(data)
    data = lz4.frame.compress(data)
    # data = base64.b64encode(data).decode("ascii")
    return data

def unpack(data):
    # data = base64.b64decode(data)
    data = lz4.frame.decompress(data)
    data = pickle.loads(data)
    return data


class ReplayMemory2:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        batch = (state, action, reward, next_state, done)
        # batch = pack(batch) # slow it down 10x
        self.buffer[self.position] = batch
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # batch = [unpack(d) for d in batch]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save(self, memory_path=None):
        logger.info(f'Saving memory to {memory_path}')
        hickle.save(self.buffer, memory_path, compression='gzip', shuffle=True)

    def load(self, memory_path):
        logger.info('Loading memory from {memory_path}')
        if memory_path is not None:
            self.buffer = hickle.load(memory_path)
            self.position = len(self.buffer)


class ReplayMemory:
    def __init__(self, capacity, seed, observation_dim, action_dim):
        random.seed(seed)
        self.capacity = capacity
        self._observations = np.zeros((capacity, observation_dim), dtype='float16')
        self._actions = np.zeros((capacity, action_dim))
        self._rewards = np.zeros((capacity, 1))
        self._next_obs = np.zeros((capacity, observation_dim), dtype='float16')
        self._terminals = np.zeros((capacity, 1), dtype='uint8')
        self.position = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        self._observations[self.position] = state
        self._actions[self.position] = action
        self._rewards[self.position] = reward
        self._next_obs[self.position] = next_state
        self._terminals[self.position] = done
        self.position = (self.position + 1) % self.capacity
        if self._size<self.capacity:
            self._size += 1

    def sample(self, batch_size):
        n = min(self.position, self.capacity)
        indices = np.random.choice(n, size=batch_size)
        state = self._observations[indices]
        action = self._actions[indices]
        reward = self._rewards[indices]
        next_state = self._next_obs[indices]
        done = self._terminals[indices]
        return state, action, reward, next_state, done

    def __len__(self):
        return self._size


# class BatchedReplayMemory:
#     def __init__(self, capacity, seed, action_dim, observation_dim):
#         random.seed(seed)
#         self.capacity = capacity
#         self._observations = np.zeros((capacity, observation_dim))
#         self._actions = np.zeros((capacity, action_dim), dtype='float16')
#         self._rewards = np.zeros((capacity, 1))
#         self._next_obs = np.zeros((capacity, observation_dim), dtype='float16')
#         self._terminals = np.zeros((capacity, 1), dtype='uint8')
#         self.position = 0
#         raise NotImplementedError()

#     def push(self, state, action, reward, next_state, done):
#         self._observations[self.position] = state
#         self._actions[self.position] = action
#         self._rewards[self.position] = reward
#         self._next_obs[self.position] = next_state
#         self._terminals[self.position] = done
#         if self.position > self.capacity:
#             # write to a dask capable file
#         self.position = (self.position + 1) % self.capacity
#         raise NotImplementedError()

#     def sample(self, batch_size):
#         # first choose a historic dask file, and this one
#         # sample from both
#         indices = np.random.choice(self._size, size=batch_size)
#         state = self._observations[indices]
#         action = self._actions[indices]
#         reward = self._rewards[indices]
#         next_state = self._next_obs[indices]
#         done = self._terminals[indices]
#         return state, action, reward, next_state, done

#     def __len__(self):
#         return len(self._observations)
