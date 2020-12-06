import torch

from py_cache_emu import ListWiseCacheEnv
from py_cache_emu import torch_utils as ptu
from py_cache_emu.runners import CacheRunner
from py_cache_emu.utils import mp_utils as mpu
from . import config
from .ewdrl import eval_agent_class
from .kd_callback import eval_callback_class


class RlCacheRunner(CacheRunner):
    def __init__(self, capacity, data_config, feature_config, **kwargs):
        super(RlCacheRunner, self).__init__(capacity, data_config, feature_config, **kwargs)
        
        self.agent_config = kwargs.pop("agent_config", config.DEFAULT_DRL_AGENT_CONFIG)
        self.agent_class_name = self.agent_config.get("class_name", "EWDNN")
        self.agent = None
        
        if self.feature_config is None or len(self.feature_config) == 0:
            self.feature_config = config.DEFAULT_DRL_FEATURE_CONFIG
        
        # 是否启用蒸馏
        self.kd_mode = int(kwargs.get("kd_mode", 0))
        self.sub_tag = self.agent_class_name
        if self.kd_mode == 1:
            self.sub_tag += "_kd_fixed"
        elif self.kd_mode == 2:
            self.sub_tag += "_kd_adaptive"
        
        # 初始化变量
        self.observation = None
        self.action = None
        self.reward = None
        self.next_observation = None
        self.kwargs = kwargs
    
    def on_run_begin(self):
        # 注册cuda运行设备
        ptu.register_device(**self.kwargs)
        
        self.env = ListWiseCacheEnv(
            capacity=self.capacity,
            data_config=self.data_config, feature_config=self.feature_config,
            main_tag=self.main_tag, sub_tag=self.sub_tag,
            **self.kwargs)
        
        # 初始化agent
        self.agent = eval_agent_class(self.agent_class_name)(
            content_dim=self.capacity, feature_dim=self.env.feature_manger.dim,
            **{**self.kwargs, **self.agent_config})
        
        if self.kd_mode > 0:
            mpu.register_process(self.rank)
            kd_config = self.kwargs.get("kd_config", config.DEFAULT_DISTILLING_CONFIG)
            kd_class_name = kd_config.get("class_name", "HardKDCallback")
            print(self.sub_tag, kd_class_name)
            kd_class = eval_callback_class(kd_class_name)
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
