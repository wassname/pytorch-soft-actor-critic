import random
import numpy as np
import pickle
import os

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

    def save(self, env_name, suffix="", memory_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if memory_path is None:
            memory_path = "models/memory_buffer_{}_{}".format(env_name, suffix)
        print('Saving memory to {}'.format(memory_path))
        pickle.dump(self.buffer, open(memory_path, 'wb'))

    def load(self, memory_path):
        print('Loading memory from {}'.format(memory_path))
        if memory_path is not None:
            self.buffer = pickle.load(open(memory_path, 'rb'))
