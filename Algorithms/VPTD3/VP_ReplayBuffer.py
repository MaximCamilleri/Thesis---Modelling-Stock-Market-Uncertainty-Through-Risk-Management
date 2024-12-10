from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from typing import List, Dict, Any, Union, Optional, Generator
import torch as th
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.type_aliases import ReplayBufferSamples


try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class VP_ReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
        
        self.vols = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        vol: np.ndarray,
        target_vol: float,
        infos: List[Dict[str, Any]],
    ) -> None:
        
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.vols[self.pos] = abs(target_vol - np.array(vol).copy())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        priorities = self._calc_priorities()
        upper_bound = self.buffer_size if self.full else self.pos
        
        batch_inds = np.random.choice(
            list(range(0, len(upper_bound))), 
            size=batch_size, 
            replace=False, 
            p=priorities
        )
        return self._get_samples(batch_inds, env=env)

    def _calc_priorities(self): 
        vol_range = np.ptp(self.vols)
        # print(self.distance)
        if vol_range != 0:
            buffer_norm = (self.distance - np.min(self.distance))/vol_range
        else: 
            buffer_norm = self.distance

        def invert(input):
            return 1 - input + 0.000000001
        
        probabilities = list(map(invert, buffer_norm))
        return probabilities/np.array(probabilities).sum()

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
