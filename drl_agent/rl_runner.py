import torch

from cache_emu import ListWiseCacheEnv
from cache_emu import thread_utils as tru
from cache_emu import torch_utils as ptu
from cache_emu.runners import CacheRunner
from . import config
from .callbacks import eval_callback_class
from .drl import eval_agent_class


class RlCacheRunner(CacheRunner):
    def __init__(self, capacity, **kwargs):
        super(RlCacheRunner, self).__init__(capacity, **kwargs)
        
        if self.feature_config is None:
            self.feature_config = config.DEFAULT_DRL_FEATURE_CONFIG
        
        self.agent_config = kwargs.pop("agent_config", config.DEFAULT_DRL_AGENT_CONFIG)
        self.agent_class_name = self.agent_config.get("class_name", "EWDQN")
        
        self.sub_tag = self.agent_class_name
        self.env = ListWiseCacheEnv(
            capacity=capacity,
            data_config=self.data_config, feature_config=self.feature_config,
            main_tag=self.main_tag, sub_tag=self.sub_tag,
            **kwargs)
        
        # 在线程启动后的on_run_begin()处初始化, 这样线程id才能与torch_device绑定。
        self.agent = None
        
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
    
    def on_run_begin(self):
        # 初始化rl agent
        self.agent = eval_agent_class(self.agent_class_name)(
            content_dim=self.capacity, feature_dim=self.env.feature_manger.dim,
            **self.agent_config, **self.kwargs)
        
        tru.register_thread()
        kd_config = self.kwargs.get("kd_config", config.DEFAULT_DISTILLING_CONFIG)
        kd_class_name = eval_callback_class(kd_config.get("class_name", None))
        if kd_class_name is not None:
            self.env.callback_manager.register_callback(
                kd_class_name(model=self.agent.get_distilling_model(), memory=self.agent.memory,
                              main_tag=self.main_tag, sub_tag=self.sub_tag,
                              **kd_config, **self.kwargs, ))
