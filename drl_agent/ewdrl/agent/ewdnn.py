import copy

from py_cache_emu.utils import torch_utils as ptu
from ..core import Agent
from ..memory import *
from ..model import RLModel
from ..nn import EWMLP
from ..policy import *


class EWVModel(RLModel):
    def build_model(self, **kwargs):
        net = EWMLP(self.feature_dim, self.hidden_layer_units)
        optim = torch.optim.Adam(net.parameters(), self.lr)
        loss_fn = torch.nn.MSELoss()
        return net, optim, loss_fn
    
    def forward_distilling(self, x):
        return super().forward(x)


class EWDNN(Agent):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32,
                 hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2, **kwargs):
        
        super().__init__(content_dim, feature_dim, **kwargs)
        
        self.v_model = EWVModel(content_dim, feature_dim, hidden_layer_units, lr)
        self.target_v_model = copy.deepcopy(self.v_model)
        
        # 创建ReplayBuffer
        self.memory = Memory(memory_size)
        
        # 动作策略参数
        self.policy = DecayEpsGreedyQPolicy(content_dim, eps_min=0)
        
        # RL超参数
        self.gamma = gamma
        self.target_update = target_update
        self.batch_size = batch_size
        
        self.update_count = 0
    
    def forward(self, observation):
        with torch.no_grad():
            state_values = self.v_model.forward(observation)
        return self.policy.select_action(ptu.get_numpy(state_values))
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
        act_idx = action.min(dim=-1)[1]
        if act_idx < self.content_dim:
            sample = torch.cat([observation[act_idx].unsqueeze(0),
                                next_observation[act_idx].unsqueeze(0)], dim=0)
            self.memory.store_kd_sample(sample)
        
        observation = observation[:self.content_dim].unsqueeze(0)
        action = action[:self.content_dim].unsqueeze(0)
        reward = reward[:self.content_dim].unsqueeze(0)
        next_observation = next_observation[:self.content_dim].unsqueeze(0)
        
        self.memory.store_transition(observation, action, reward, next_observation)
        
        observations, actions, rewards, next_observations = self.memory.sample_batch(self.batch_size)
        
        info = self._update(observations, actions, rewards, next_observations)
        self._update_target()
        return info
    
    def _update(self, observations, actions, rewards, next_observations):
        with torch.no_grad():
            next_values = self.target_v_model.forward_target(next_observations)
            target_values = rewards + self.gamma * next_values
        
        v_loss = self.v_model.fit(observations, target_values)
        return {"v_loss": v_loss}
    
    def _update_target(self):
        if self.target_update > 1 and self.update_count % self.target_update == 0:
            ptu.copy_model_params_from_to(self.v_model, self.target_v_model)
        else:
            ptu.soft_update_from_to(self.v_model, self.target_v_model, self.target_update)
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        self.v_model.save_weights(path, prefix, suffix)
        self.target_v_model.save_weights(path, prefix + "target_", suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        res1 = self.v_model.load_weights(path, prefix, suffix)
        res2 = self.target_v_model.load_weights(path, prefix + "target_", suffix)
        return res1 and res2
    
    def get_distilling_model(self):
        return self.v_model
    
    def get_models(self):
        return [self]
