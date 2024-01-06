import numpy as np
import random
from collections import deque
import itertools
from .settings import NEARLY_DATA


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size

    def sample(self, batchsize):
        batch = []
        buffer_length = self.get_length()
        batchsize = min(batchsize, buffer_length)
        new_trajectory = list(itertools.islice(self.buffer, buffer_length-NEARLY_DATA, buffer_length))
        # print("new_trajectory",new_trajectory)
        batch = random.sample(self.buffer, batchsize-NEARLY_DATA)
        # print("batch",batch)
        batch.extend(new_trajectory)
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def get_length(self):
        return len(self.buffer)

    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.buffer.append(transition)
