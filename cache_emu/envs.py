import gym
import numpy as np

from .cache import Cache
from .callback import CallbackManager
from .feature import FeatureManager
from .request import RequestLoader, RequestSlice
from .utils import NoneContentType


class CacheEnv(gym.Env):
    def __init__(self, capacity: int, data_config={}, feature_config={},
                 callback_manager: CallbackManager = None):
        self.capacity = capacity
        self.cache = Cache(capacity=capacity)
        self.loader = RequestLoader(**data_config)
        self.feature_manger = FeatureManager(**feature_config)
        self.callback_manager = callback_manager
        
        # 定义状态空间动作空间
        self.action_space = gym.spaces.Discrete(capacity)
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.capacity, self.feature_manger.dim), dtype=np.float32),
            high=np.ones((self.capacity, self.feature_manger.dim), dtype=np.float32)
        )
        
        self.req_slice: RequestSlice = None
        self.missed_content = NoneContentType
        self.request_cnt = 0
        self.hit_cnt = 0
        
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
    
    def reset(self):
        self.cache.reset()
        if self.callback_manager is not None:
            self.callback_manager.reset()
        
        for i in range(self.capacity):
            self.cache.store(i)
        
        self.req_slice = self.loader.next_slice()
        
        content_ids = self.cache.get_contents()
        self.observation = self.feature_manger.forward(content_ids)
        return self.observation
    
    def close(self):
        return self.callback_manager.on_game_end()
    
    def step(self, action: int):
        assert 0 <= action < self.capacity
        self.action = action
        
        if self.missed_content != NoneContentType:
            self.cache.evict(self.action)
            self.cache.store(self.missed_content)
        
        info = {}
        self.reward = 0
        self.missed_content = NoneContentType
        
        while True:
            while not self.req_slice.finished():
                timestamp, content_id = self.req_slice.next()
                hit = self.cache.hit_test(content_id)
                
                self.hit_cnt += hit
                self.request_cnt += 1
                
                if hit:
                    self.reward += hit
                else:
                    self.missed_content = content_id
                    self.observation = self.next_observation
                    self.next_observation = self._get_observation()
                    info.update({"hit_rate": self.hit_cnt / self.request_cnt})
                    return self.next_observation, self.reward, False, info
            
            if self.callback_manager is not None:
                self.callback_manager.on_step_end()
            
            if not self.loader.finished():
                self.feature_manger.update(self.req_slice.timestamps, self.req_slice.content_ids)
                self.req_slice = self.loader.next_slice()
            else:
                self.observation = self.next_observation
                self.next_observation = self._get_observation()
                info.update({"hit_rate": self.hit_cnt / self.request_cnt})
                return self.next_observation, self.reward, True, info
    
    def _get_observation(self):
        content_ids = self.cache.get_contents()
        observation = self.feature_manger.forward(content_ids)
        return observation
    
    def render(self, mode='human'):
        pass
