from threading import Thread

import numpy as np
import torch

from cache_emu import CacheEnv
from .ewdrl import EWDQN
from .ewdrl.utils import torch_utils as ptu


class CacheRunner(Thread):
    def __init__(self, **kwargs):
        super().__init__()
        self.env: CacheEnv = None
        self.result = None
    
    def run(self):
        assert self.env is not None
        info = {}
        terminal = False
        observation = self.env.reset()
        while not terminal:
            action = self.forward(observation)
            next_observation, reward, terminal, info = self.env.step(action)
            self.backward(reward, terminal, next_observation)
            observation = next_observation
        self.result = info
        return info
    
    def close(self):
        if self.env is not None:
            self.env.close()
    
    def forward(self, observation):
        raise NotImplementedError()
    
    def backward(self, reward, terminal, next_observation):
        pass
    
    def get_result(self):
        return self.result


class RandomCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_random_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LruCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_lru_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LfuCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_lfu_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class SwfCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        observation = observation[:, -1]
        return np.argmin(observation.flatten())


class OgdOptCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_ogd_opt_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLruCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_ogd_lru_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLfuCacheRunner(CacheRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['feature_config'] = {'use_ogd_lfu_feature': True}
        self.env = CacheEnv(**kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class EwdqnCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super().__init__(**kwargs)
        self.env = CacheEnv(capacity=capacity, list_wise_mode=True, **kwargs)
        self.agent = EWDQN(content_dim=capacity, feature_dim=self.env.feature_manger.dim)
        
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
    
    def forward(self, observation):
        self.observation = ptu.float_tensor(observation)
        self.action = self.agent.forward(self.observation)
        return np.argmin(self.action)
    
    def backward(self, reward, terminal, next_observation):
        if self.observation is not None and self.action is not None:
            self.reward = ptu.float_tensor(reward)
            self.action = ptu.tensor(self.action, dtype=torch.long)
            self.next_observation = ptu.float_tensor(next_observation)
            self.agent.backward(self.observation, self.action, self.reward, self.next_observation)
