import torch

from cache_emu import ListWiseCacheEnv
from cache_emu import torch_utils as ptu
from cache_emu.runners import CacheRunner
from .config import DEFAULT_DRL_FEATURE_CONFIG, DEFAULT_DRL_AGENT_CONFIG
from .drl import eval_agent_class


class RlCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(RlCacheRunner, self).__init__(capacity, **kwargs)
        
        if self.feature_config is None:
            self.feature_config = DEFAULT_DRL_FEATURE_CONFIG
        
        self.agent_config = kwargs.pop("agent_config", DEFAULT_DRL_AGENT_CONFIG)
        agent_class_name = self.agent_config.get("class_name", "EWDQN")
        
        self.sub_tag = agent_class_name
        self.env = ListWiseCacheEnv(
            capacity=capacity,
            data_config=self.data_config, feature_config=self.feature_config,
            main_tag=self.main_tag, sub_tag=self.sub_tag,
            **kwargs)
        
        self.agent = eval_agent_class(agent_class_name)(
            content_dim=capacity, feature_dim=self.env.feature_manger.dim,
            **self.agent_config, **kwargs)
        
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
    
    def forward(self, observation):
        self.observation = ptu.float_tensor(observation)
        self.action = self.agent.forward(self.observation)
        return self.action
    
    def backward(self, reward, terminal, next_observation):
        if self.observation is not None and self.action is not None:
            self.reward = ptu.float_tensor(reward)
            self.action = ptu.tensor(self.action, dtype=torch.long)
            self.next_observation = ptu.float_tensor(next_observation)
            return self.agent.backward(self.observation, self.action, self.reward, self.next_observation)
