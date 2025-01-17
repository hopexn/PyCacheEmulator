import gym
import numpy as np

from .cache import Cache, NoneContentType
from .callback import CallbackManager
from .feature.manager import FeatureManager
from .request import RequestLoader, RequestSlice


class CacheEnv(gym.Env):
    def __init__(self, capacity, data_config={}, feature_config={}, callback_config={}, **kwargs):
        self.capacity = capacity if not isinstance(capacity, list) else capacity[kwargs.get('rank', 0)]
        self.cache = Cache(capacity=self.capacity)
        self.loader = RequestLoader(**{**kwargs, **data_config})
        self.feature_manger = FeatureManager(
            max_contents=self.loader.get_max_contents(), **{**kwargs, **feature_config})
        self.callback_manager = CallbackManager(
            total_steps=self.loader.n_slices, **{**kwargs, **callback_config})
        
        # 定义状态空间动作空间
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.capacity + 1, self.feature_manger.dim), dtype=np.float32),
            high=np.ones((self.capacity + 1, self.feature_manger.dim), dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(self.capacity + 1)
        
        self.req_slice: RequestSlice = None
        self.missed_content = NoneContentType
        
        self.next_observation = None
        
        self.slice_hit_cnt = 0
        self.slice_req_cnt = 0
        self.total_hit_cnt = 0
        self.total_req_cnt = 0
    
    def reset(self):
        self.cache.reset()
        self.callback_manager.reset()
        self.loader.reset()
        
        self.next_observation = None
        
        self.slice_hit_cnt = 0
        self.slice_req_cnt = 0
        self.total_hit_cnt = 0
        self.total_req_cnt = 0
        
        for i in range(self.capacity):
            self.cache.store(i)
        
        self.req_slice = self.loader.next_slice()

        self.callback_manager.on_game_begin()
        
        return self._get_observation()
    
    def close(self):
        return self.callback_manager.on_game_end()
    
    def step(self, action: int):
        assert 0 <= action <= self.capacity
        
        if self.missed_content != NoneContentType and action != self.capacity:
            self.cache.evict(action)
            self.cache.store(self.missed_content)
        
        step_hit_cnt = 0
        step_req_cnt = 0
        self.missed_content = NoneContentType
        
        info = {}
        
        while True:
            while not self.req_slice.finished():
                timestamp, content_id = self.req_slice.next()
                hit = self.cache.hit_test(content_id)
                
                step_hit_cnt += hit
                step_req_cnt += 1
                
                self.slice_hit_cnt += hit
                self.slice_req_cnt += 1
                
                self.total_hit_cnt += hit
                self.total_req_cnt += 1
                
                self.feature_manger.update(timestamp, content_id)
                
                if not hit:
                    self.missed_content = content_id
                    self.next_observation = self._get_observation()
                    reward = self._get_reward()
                    info.update({
                        "step_req_cnt"  : step_req_cnt,
                        "step_hit_cnt"  : step_hit_cnt,
                        "missed_content": self.missed_content
                    })
                    
                    return self.next_observation, reward, False, info
            
            self.callback_manager.on_step_end(slice_hit_cnt=self.slice_hit_cnt, slice_req_cnt=self.slice_req_cnt)
            self.slice_hit_cnt = 0
            self.slice_req_cnt = 0
            
            if not self.loader.finished():
                self.feature_manger.update_batch(self.req_slice.timestamps, self.req_slice.content_ids)
                self.req_slice = self.loader.next_slice()
            else:
                self.next_observation = self._get_observation()
                reward = self._get_reward()
                info.update({
                    "total_req_cnt": self.total_req_cnt,
                    "total_hit_cnt": self.total_hit_cnt,
                    "mean_hit_rate": "{:.1f}%".format(100 * self.total_hit_cnt / (self.total_req_cnt + 1e-6))
                })
                return self.next_observation, reward, True, info
    
    def _get_observation(self):
        candidates = np.concatenate([self.cache.get_contents(), [self.missed_content]])
        observation = self.feature_manger.forward(candidates)
        return observation
    
    def _get_reward(self):
        return self.cache.get_frequencies().sum()
    
    def render(self, mode='human'):
        pass


class ListWiseCacheEnv(CacheEnv):
    def __init__(self, capacity, data_config={}, feature_config={}, callback_config={}, **kwargs):
        super().__init__(capacity, data_config, feature_config, callback_config, **kwargs)
        self.action_space = gym.spaces.MultiDiscrete(self.capacity + 1)
    
    def step(self, action):
        return super().step(np.argmin(action))
    
    def _get_reward(self):
        candidates = np.concatenate([self.cache.get_contents(), [self.missed_content]])
        return self.cache.get_frequencies(candidates)
