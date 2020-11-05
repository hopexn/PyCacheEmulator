import gym
import numpy as np

from .cache import Cache
from .callback import CallbackManager
from .feature.manager import FeatureManager
from .request import RequestLoader, RequestSlice
from .utils import NoneContentType


class CacheEnv(gym.Env):
    def __init__(self, capacity: int,
                 data_config={}, feature_config={}, callback_config={}, list_wise_mode=False, **kwargs):
        self.capacity = capacity
        self.cache = Cache(capacity=capacity)
        self.loader = RequestLoader(**data_config)
        self.feature_manger = FeatureManager(max_contents=self.loader.get_max_contents(), **feature_config)
        self.callback_manager = CallbackManager(total_steps=self.loader.n_slices, **callback_config)
        
        self.list_wise_mode = list_wise_mode
        
        # 定义状态空间动作空间
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.capacity + 1, self.feature_manger.dim), dtype=np.float32),
            high=np.ones((self.capacity + 1, self.feature_manger.dim), dtype=np.float32)
        )
        if not self.list_wise_mode:
            self.action_space = gym.spaces.Discrete(capacity + 1)
        else:
            self.action_space = gym.spaces.MultiBinary(capacity + 1)
        
        self.req_slice: RequestSlice = None
        self.missed_content = NoneContentType
        self.request_cnt = 0
        self.hit_cnt = 0
        
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
        self.info = {}
    
    def reset(self):
        self.cache.reset()
        self.callback_manager.reset()
        self.loader.reset()
        self.info.clear()
        
        for i in range(self.capacity):
            self.cache.store(i)
        
        self.req_slice = self.loader.next_slice()
        self.observation = self._get_observation()
        return self.observation
    
    def close(self):
        return self.callback_manager.on_game_end()
    
    def step(self, action: int):
        assert 0 <= action <= self.capacity
        self.action = action
        
        if self.missed_content != NoneContentType and self.action != self.capacity:
            self.cache.evict(self.action)
            self.cache.store(self.missed_content)
        
        self.reward = 0
        self.missed_content = NoneContentType
        
        while True:
            while not self.req_slice.finished():
                timestamp, content_id = self.req_slice.next()
                hit = self.cache.hit_test(content_id)
                
                self.hit_cnt += hit
                self.request_cnt += 1
                
                self.feature_manger.update(timestamp, content_id)
                
                if hit:
                    self.reward += hit
                else:
                    self.missed_content = content_id
                    self.observation = self.next_observation
                    self.next_observation = self._get_observation()
                    self.info.update({"hit_rate": self.hit_cnt / self.request_cnt})
                    if self.list_wise_mode:
                        self.reward = self.cache.get_frequencies()
                    return self.next_observation, self.reward, False, self.info
            
            self.callback_manager.on_step_end(postfix=self.info.copy())
            
            if not self.loader.finished():
                self.feature_manger.update_batch(self.req_slice.timestamps, self.req_slice.content_ids)
                self.req_slice = self.loader.next_slice()
            else:
                self.info.update({"hit_rate": self.hit_cnt / self.request_cnt})
                if self.list_wise_mode:
                    self.reward = self.cache.get_frequencies()
                return self.next_observation, self.reward, True, self.info
    
    def _get_observation(self):
        candidates = np.concatenate([self.cache.get_contents(), [self.missed_content]])
        observation = self.feature_manger.forward(candidates)
        return observation
    
    def render(self, mode='human'):
        pass
