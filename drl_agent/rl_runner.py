import torch

from cache_emu import ListWiseCacheEnv
from cache_emu import torch_utils as ptu
from cache_emu.runners import CacheRunner
from cache_emu.utils import mp_utils as mpu
from . import config
from .kd_callback import eval_callback_class
from .ewdrl import eval_agent_class


class RlCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(RlCacheRunner, self).__init__(capacity, data_config, **kwargs)
        
        self.agent_config = kwargs.pop("agent_config", config.DEFAULT_DRL_AGENT_CONFIG)
        self.agent_class_name = self.agent_config.get("class_name", "EWDQN")
        self.agent = None
        
        if feature_config is None or len(feature_config) == 0:
            feature_config = config.DEFAULT_DRL_FEATURE_CONFIG
        
        # 是否启用蒸馏
        self.enable_distilling = kwargs.get("enable_distilling", False)
        
        # 更新sub_tag
        self.sub_tag = self.agent_class_name
        self.sub_tag += "_kd" if self.enable_distilling else ""
        
        # 初始化环境
        self.env = ListWiseCacheEnv(
            capacity=capacity,
            data_config=data_config, feature_config=feature_config,
            main_tag=self.main_tag, sub_tag=self.sub_tag,
            **kwargs)
        
        # 初始化变量
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
        self.kwargs = kwargs
    
    def on_run_begin(self):
        # 注册cuda运行设备
        ptu.register_device(**self.kwargs)
        # 初始化agent
        
        self.agent = eval_agent_class(self.agent_class_name)(
            content_dim=self.capacity, feature_dim=self.env.feature_manger.dim,
            **{**self.kwargs, **self.agent_config})
        
        if self.enable_distilling:
            mpu.register_process()
            kd_config = self.kwargs.get("kd_config", config.DEFAULT_DISTILLING_CONFIG)
            kd_class = eval_callback_class(kd_config.get("class_name", "HardKDCallback"))
            kd_callback = kd_class(
                model=self.agent.get_distilling_model(), memory=self.agent.memory,
                main_tag=self.main_tag, sub_tag=self.sub_tag,
                **{**self.kwargs, **kd_config}
            )
            self.env.callback_manager.register_callback(kd_callback)
    
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
