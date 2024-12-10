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
    def __init__(self, size, target_volatility):
        super(VP_ReplayBuffer, self).__init__(size)

        self.distance = []
        self.target_volatility = target_volatility
    
    def add(self, obs_t, action, reward, obs_tp1, done, vol): 
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self.distance.append(abs(self.target_volatility-vol))
        else:
            self._storage[self._next_idx] = data
            self.distance[self._next_idx] = abs(self.target_volatility-vol)
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def extend(self, obs_t, action, reward, obs_tp1, done, vol):
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
                self.distance.append(abs(self.target_volatility-vol))
            else:
                self._storage[self._next_idx] = data
                self.distance[self._next_idx] = abs(self.target_volatility-vol)
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        priorities = self._calc_priorities()

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

    def _calc_priorities(self): 
        vol_range = np.ptp(self.distance)
        # print(self.distance)
        if vol_range != 0:
            buffer_norm = (self.distance - np.min(self.distance))/vol_range
        else: 
            buffer_norm = self.distance

        def invert(input):
            return 1 - input + 0.000000001
        
        probabilities = list(map(invert, buffer_norm))
        return probabilities/np.array(probabilities).sum()


