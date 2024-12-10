## partical Filtering 

import warnings
warnings.filterwarnings("ignore")

from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines.common.vec_env import VecNormalize
from typing import Optional
import numpy as np
import random

class VP_ReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super(VP_ReplayBuffer, self).__init__(size)

        self.volatility = []
    
    def add(self, obs_t, action, reward, obs_tp1, done, vol): 
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self.volatility.append(vol)
        else:
            self._storage[self._next_idx] = data
            self.volatility[self._next_idx] = vol
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def extend(self, obs_t, action, reward, obs_tp1, done, vol):
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
                self.volatility.append(vol)
            else:
                self._storage[self._next_idx] = data
                self.volatility[self._next_idx] = vol
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, vol, batch_size: int, env: Optional[VecNormalize] = None):
        priorities = self._calc_priorities(vol)
        # sample_list = random.choices(
        #     list(range(0, len(self._storage))), weights = priorities, k = batch_size
        # )
        sample_list = np.random.choice(
            list(range(0, len(self._storage))), 
            size=batch_size, 
            replace=False, 
            p=priorities
        )
        return self._encode_sample(sample_list, env=env)

    # ================
    # Helper Functions
    # ================

    def _calc_priorities(self, input_val): 
        vol_range = np.ptp(self.volatility)
        if vol_range != 0:
            buffer_norm = (self.volatility - np.min(self.volatility))/vol_range
            input_norm = (input_val - np.min(self.volatility))/vol_range
        else: 
            buffer_norm = self.volatility
            input_norm = input_val

        def invert(distance):
            return 1 - abs(input_norm-distance)
        
        probabilities = list(map(invert, buffer_norm))

        return probabilities/np.array(probabilities).sum()


