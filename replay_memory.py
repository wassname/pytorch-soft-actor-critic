import random
import numpy as np
import torch
import hickle
import os
from loguru import logger

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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
