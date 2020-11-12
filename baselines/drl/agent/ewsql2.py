import copy

import torch

from ..core import Agent
from ..memory import Memory
from ..model import RLModel, TemperatureModel
from ..nn import EWMLP
from ..policy import *
from ..utils import torch_utils as ptu


class EWSqlPiModel(RLModel):
    def __init__(self, feature_dim, hidden_layer_units: list, lr, use_kl_div_loss=False):
        super(EWSqlPiModel, self).__init__(feature_dim, hidden_layer_units, lr)
        
        self.net = EWMLP(feature_dim, hidden_layer_units)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()
        
        # 策略更新是否启用KL散度
        self.use_kl_div_loss = use_kl_div_loss
    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    
    def forward_target(self, x):
        probs = torch.softmax(self.net(x), dim=-1).unsqueeze(-1)
        return torch.cat([probs, 1 - probs], dim=-1)
    
    def forward_distilling(self, x):
        return torch.softmax(self.net(x), dim=-1)
    
    def entropy(self, x):
        probs = self.forward(x)
        entropy = -(probs * torch.log(probs + 1e-6))
        return entropy
    
    def backward(self, observations, q_values, tau):
        probs = self.forward(observations)
        
        with torch.no_grad():
            target_probs = torch.softmax((q_values[:, :, 0] - q_values[:, :, 1]) / (tau + 1e-6), dim=-1)
            log_target_probs = torch.log(target_probs + 1e-6)
        
        if self.use_kl_div_loss:
            # 根据kl散度定义实现
            log_probs = torch.log(probs + 1e-6)
            loss = (probs * (log_probs - log_target_probs.detach())).mean()
            # 使用pytorch的kl_div
            # target_probs = torch.exp(log_target_probs)
            # loss = torch.nn.functional.kl_div(probs, target_probs, reduction="batchmean")
        else:
            loss = - (probs * log_target_probs.detach()).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.cpu().item()
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        super().save_weights(path, prefix, suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        return super().load_weights(path, prefix, suffix)


class EWSqlQModel(RLModel):
    def __init__(self, feature_dim, hidden_layer_units: list, lr):
        super(EWSqlQModel, self).__init__(feature_dim, hidden_layer_units, lr)
        
        self.net = ptu.build_mlp(feature_dim, hidden_layer_units, 2)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        super().save_weights(path, prefix, suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        return super().load_weights(path, prefix, suffix)


class EWSQL2(Agent):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32,
                 hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2,
                 log_tau=-1, min_entropy_ratio=0.1, update_tau_freq=1, use_kl_div_loss=True,
                 distilling_pi_model=True, **kwargs):
        super(EWSQL2, self).__init__(content_dim, feature_dim, **kwargs)
        
        # 策略函数
        print("Pi use kl div loss: ", use_kl_div_loss)
        
        self.pi_model = EWSqlPiModel(feature_dim, hidden_layer_units, lr, use_kl_div_loss=use_kl_div_loss)
        self.target_pi_model = copy.deepcopy(self.pi_model)
        
        # Q值函数
        self.q_model = EWSqlQModel(feature_dim, hidden_layer_units, lr)
        self.target_q_model = copy.deepcopy(self.q_model)
        
        # 策略函数
        self.policy = GreedyQPolicy(content_dim)
        
        # 创建ReplayBuffer
        self.memory = Memory(memory_size)
        
        # 动作策略参数
        self.policy = GreedyQPolicy(content_dim)
        # self.policy = DecayEpsGreedyQPolicy(content_dim, eps_min=0.01)
        
        # RL超参数
        self.gamma = gamma
        self.target_update = target_update
        self.batch_size = batch_size
        
        # 设置最小熵为 ratio * max_entropy, 其中 max_entropy = log(content_dim)
        self.tau = TemperatureModel(log_tau=log_tau, min_entropy=min_entropy_ratio * np.log(content_dim + 1))
        # 更新温度的频率，值为0时温度为定值
        self.update_tau_freq = update_tau_freq
        
        self.update_count = 0
        
        if distilling_pi_model:
            self.distilling_model = self.pi_model
        else:
            self.distilling_model = self.q_model
    
    def forward(self, observation):
        with torch.no_grad():
            probs = self.pi_model.forward(observation)
        
        num_candidates = observation.shape[0]
        action = np.random.choice(np.arange(num_candidates), p=ptu.get_numpy(probs))
        action_oh = np.ones(num_candidates, dtype=np.bool)
        action_oh[action] = 0
        
        return action_oh
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
        observation = observation[:self.content_dim + 1].unsqueeze(0)
        action = action[:self.content_dim + 1].unsqueeze(0)
        reward = reward[:self.content_dim + 1].unsqueeze(0)
        next_observation = next_observation[:self.content_dim + 1].unsqueeze(0)
        
        self.memory.store_transition(observation, action, reward, next_observation)
        
        observations, actions, rewards, next_observations = self.memory.sample_batch(self.batch_size)
        
        info = self._update(observations, actions, rewards, next_observations)
        self._update_target()
        return info
    
    def _update(self, observations, actions, rewards, next_observations):
        # update q net
        actions = actions.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        
        with torch.no_grad():
            next_q_values = self.target_q_model.forward_target(next_observations)
            next_probs = self.target_pi_model.forward_target(next_observations)
            
            next_state_values = (next_probs * next_q_values).sum(dim=-1, keepdim=True)
            target_q_values = rewards + self.gamma * next_state_values
            
            min_act = torch.argmin(actions.squeeze(-1), dim=-1, keepdim=True)
            entropy = self.target_pi_model.entropy(observations).gather(dim=-1, index=min_act)
            target_q_values += self.tau * entropy.unsqueeze(-1)
            
            q_values = self.q_model.forward(observations)
            new_q_values = q_values.scatter(dim=-1, index=actions, src=target_q_values)
        
        # update q model
        q_loss = self.q_model.fit(observations[:, :self.content_dim], new_q_values[:, :self.content_dim])
        
        # update policy
        q_values = self.target_q_model(observations)
        pi_loss = self.pi_model.backward(observations, q_values, self.tau.value())
        
        info = {"q_loss": q_loss, "pi_loss": pi_loss}
        
        # 每隔一段时间更新温度
        if self.update_tau_freq != 0 and self.update_count % self.update_tau_freq == 0:
            with torch.no_grad():
                probs = self.pi_model(observations)
            tau_info = self.tau.backward(probs)
            info.update(tau_info)
        
        return info
    
    def _update_target(self):
        if self.target_update > 1 and self.update_count % self.target_update == 0:
            ptu.copy_model_params_from_to(self.q_model, self.target_q_model)
            ptu.copy_model_params_from_to(self.pi_model, self.target_pi_model)
        else:
            ptu.soft_update_from_to(self.q_model, self.target_q_model, self.target_update)
            ptu.soft_update_from_to(self.pi_model, self.target_pi_model, self.target_update)
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        self.q_model.save_weights(path, prefix, suffix)
        self.target_q_model.save_weights(path, prefix + "target_", suffix)
        self.pi_model.save_weights(path, prefix, suffix)
        self.target_pi_model.save_weights(path, prefix + "target_", suffix)
        self.tau.save_weights(path, prefix, suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        res1 = self.q_model.load_weights(path, prefix, suffix)
        res2 = self.target_q_model.load_weights(path, prefix + "target_", suffix)
        res3 = self.pi_model.load_weights(path, prefix, suffix)
        res4 = self.target_pi_model.load_weights(path, prefix + "target_", suffix)
        res5 = self.tau.load_weights(path, prefix, suffix)
        return res1 and res2 and res3 and res4 and res5
    
    def get_distilling_model(self):
        return self.distilling_model
    
    def get_models(self):
        return [self]
