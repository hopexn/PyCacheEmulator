import numpy as np
import torch

from cache_emu import CacheEnv
from .drl import eval_agent_class
from .drl.utils import torch_utils as ptu
from .runners import CacheRunner


class RlCacheRunner(CacheRunner):
    def __init__(self, capacity, agent_class="EWDQN", agent_kwargs={}, **kwargs):
        super(RlCacheRunner, self).__init__(**kwargs)
        kwargs = kwargs.copy()
        kwargs.update({"sub_tag": agent_class})
        self.env = CacheEnv(capacity=capacity, list_wise_mode=True, **kwargs)
        self.agent = eval_agent_class(agent_class)(content_dim=capacity, feature_dim=self.env.feature_manger.dim,
                                                   **agent_kwargs, **kwargs)
        
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
            return self.agent.backward(self.observation, self.action, self.reward, self.next_observation)
