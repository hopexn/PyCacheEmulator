import multiprocessing as mp

import numpy as np

from .config.data import *
from .config.feature import *
from .envs import CacheEnv


class CacheRunner(mp.Process):
    def __init__(self, capacity, **kwargs):
        super().__init__()
        
        self.capacity = capacity
        self.kwargs = kwargs
        self.msg_queue: mp.Queue = kwargs.get("msg_queue", None)
        
        self.env = None
        
        # 解析参数
        self.data_config = kwargs.pop("data_config", IQIYI_DATA_CONFIG)
        self.feature_config = kwargs.pop("feature_config", None)
        
        self.main_tag = "{}/{}/{}".format(
            self.data_config.get("name", ""),
            kwargs.get("rank", 0),
            self.capacity,
        )
        self.sub_tag = self.__class__.__name__[:-11]
    
    def run(self, **kwargs):
        assert self.env is not None
        
        result = {}
        res = self.on_run_begin(**self.kwargs, **kwargs)
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
        
        res = self.on_run_end(**self.kwargs, **kwargs)
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
    def __init__(self, capacity, **kwargs):
        super(RandomCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_RANDOM_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LruCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(LruCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_LRU_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LfuCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(LfuCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_LFU_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdOptCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(OgdOptCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_OGD_OPT_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLruCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(OgdLruCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_OGD_LRU_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLfuCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(OgdLfuCacheRunner, self).__init__(capacity, **kwargs)
        self.feature_config = DEFAULT_OGD_LFU_FEATURE_CONFIG
        self.env = CacheEnv(capacity=capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
    
    def forward(self, observation):
        return np.argmin(observation.flatten())
