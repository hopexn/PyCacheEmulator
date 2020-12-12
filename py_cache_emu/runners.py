import multiprocessing as mp

import numpy as np

from .envs import CacheEnv
from .misc.arc import ARC


class CacheRunner(mp.Process):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super().__init__()
        
        self.capacity = capacity if not isinstance(capacity, list) else capacity[kwargs.get('rank', 0) % len(capacity)]
        self.data_config = data_config
        self.feature_config = feature_config
        
        self.msg_queue: mp.Queue = kwargs.get("msg_queue", None)
        self.env: CacheEnv = None
        
        self.rank = kwargs.get('rank', 0)
        self.data_rank = kwargs.get('data_rank', 0)
        
        self.main_tag = "{}/{}/{}".format(
            data_config.get("name", ""),
            self.data_rank,
            self.capacity,
        )
        self.sub_tag = self.__class__.__name__[:-11]
        self.kwargs = kwargs
    
    def run(self, **kwargs):
        result = {}
        res = self.on_run_begin(**kwargs)
        assert self.env is not None
        
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
        self.env = CacheEnv(capacity=self.capacity,
                            data_config=self.data_config, feature_config=self.feature_config,
                            main_tag=self.main_tag, sub_tag=self.sub_tag,
                            **self.kwargs)
        self.env.reset()
    
    def on_run_end(self, **kwargs):
        pass
    
    def forward(self, observation):
        raise NotImplementedError()
    
    def backward(self, reward, terminal, next_observation):
        pass


class RandomCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(RandomCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_random_feature': True}
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LruCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(LruCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_lru_feature': True}
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class LfuCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(LfuCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_lfu_feature': True}
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdOptCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdOptCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_ogd_opt_feature': True}
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLruCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdLruCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_ogd_lru_feature': True}
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class OgdLfuCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(OgdLfuCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.feature_config = {'use_ogd_lfu_feature': True},
    
    def forward(self, observation):
        return np.argmin(observation.flatten())


class ArcCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(ArcCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        self.arc = ARC(capacity)
    
    def run(self, **kwargs):
        self.on_run_begin(**kwargs)
        while not self.env.loader.finished():
            req_slice = self.env.loader.next_slice()
            while not req_slice.finished():
                t, cid = req_slice.next()
                self.arc.re(cid)
        mean_hit_rate = self.arc.get_hit_ratio()
        result = {
            "mean_hit_rate"  : "{:.1f}%".format(mean_hit_rate * 100),
            "mean_hit_rate50": "{:.1f}%".format(mean_hit_rate * 100),
            "total_hit_cnt"  : str(self.arc.hit_cnt),
            "total_req_cnt"  : str(self.arc.req_cnt)
        }
        self.msg_queue.put({"{}/{}".format(self.main_tag, self.sub_tag): result})
        return result
    
    def forward(self, observation):
        return np.argmin(observation.flatten())
