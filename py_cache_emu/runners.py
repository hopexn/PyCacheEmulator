import multiprocessing as mp

import numpy as np

from .envs import CacheEnv


class CacheRunner(mp.Process):
    def __init__(self, capacity, data_config, **kwargs):
        super().__init__()
        
        self.capacity = capacity if not isinstance(capacity, list) else capacity[kwargs.get('rank', 0)]
        self.data_config = data_config
        
        self.msg_queue: mp.Queue = kwargs.get("msg_queue", None)
        self.env = None
        
        self.main_tag = "{}/{}/{}".format(
            data_config.get("name", ""),
            kwargs.get("rank", 0),
            self.capacity,
        )
        self.sub_tag = self.__class__.__name__[:-11]
        self.rank = kwargs.get('rank', 0)
    
    def run(self, **kwargs):
        assert self.env is not None
        
        result = {}
        res = self.on_run_begin(**kwargs)
        if res is not None:
            result.update(res)
        
        terminal = False
        observation = self.env.reset()
        while not terminal:
            action = self.forward(observation)
            next_observation, reward, terminal, info = self.env.step(action)
            self.backward(reward, terminal, next_observation)
            observation = next_observation
        
        res = self.env.close()
        if res is not None:
            result.update(res)
        
        res = self.on_run_end(**kwargs)
        if res is not None:
            result.update(res)
        
        if self.msg_queue is not None:
            self.msg_queue.put({"{}/{}".format(self.main_tag, self.sub_tag): result})
        
        return result
    
    def on_run_begin(self, **kwargs):
        pass
    
    def on_run_end(self, **kwargs):
        pass
    
    def forward(self, observation):
        raise NotImplementedError()
    
    def backward(self, reward, terminal, next_observation):
        pass


class RandomCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(RandomCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_random_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LruCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(LruCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_lru_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LfuCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(LfuCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_lfu_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdOptCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdOptCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_ogd_opt_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLruCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdLruCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_ogd_lru_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLfuCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdLfuCacheRunner, self).__init__(capacity, data_config, **kwargs)
        self.env = CacheEnv(capacity=capacity,
                            data_config=data_config, feature_config={'use_ogd_lfu_feature': True},
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())
