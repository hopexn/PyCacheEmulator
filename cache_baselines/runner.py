from threading import Thread

import numpy as np

from cache_emu import CacheEnv


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
