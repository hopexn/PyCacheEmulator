import copy

import torch

from .ewdqn import EWDQN
from ..memory import Memory
from ..model import RLModel, Temperature
from ..nn import EWMLP
from ..policy import *
from ..utils import torch_utils as ptu


class EWSqlPiModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr, use_kl_div_loss=False):
        super(EWSqlPiModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        
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
        probs = self.forward_target(x)
        entropy = -(probs * torch.log(probs + 1e-6))
        return entropy
    
    def backward(self, observations, actions, target_q_values, state_values, tau):
        probs = self.forward_target(observations).gather(dim=-1, index=actions).squeeze(-1)
        
        log_target_probs = (target_q_values - state_values) / (tau + 1e-6)
        
        if self.use_kl_div_loss:
            log_probs = torch.log(probs + 1e-6)
            loss = (probs * (log_probs - log_target_probs.detach())).mean()
        else:
            loss = -(probs * log_target_probs).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.cpu().item()


class EWSQL(EWDQN):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32,
                 hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2,
                 log_tau=-1, log_tau_clip=(-5, 1), min_entropy_ratio=0.1, update_tau_freq=1, use_kl_div_loss=True,
                 distilling_pi_model=True, **kwargs):
        super(EWSQL, self).__init__(content_dim, feature_dim, memory_size, batch_size, hidden_layer_units, lr, gamma,
                                    target_update, **kwargs)
        # 策略函数
        print("Policy use kl div loss: ", use_kl_div_loss)
        self.pi_model = EWSqlPiModel(content_dim, feature_dim, hidden_layer_units, lr, use_kl_div_loss=use_kl_div_loss)
        self.target_pi_model = copy.deepcopy(self.pi_model)
        
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
        self.tau = Temperature(log_tau=log_tau, min_entropy=min_entropy_ratio * np.log(2),
                               log_tau_clip=log_tau_clip)
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
        
        with torch.no_grad():
            next_state_values = self.target_q_model.forward_target(next_observations)
            target_q_values = rewards + self.gamma * next_state_values
            
            entropy = self.target_pi_model.entropy(observations).gather(dim=-1, index=actions).squeeze(-1)
            target_q_values = target_q_values + self.tau * entropy
        
        # update q model
        v_loss = self.q_model.fit(observations[:, :self.content_dim], target_q_values[:, :self.content_dim])
        
        # update policy
        state_values = self.target_q_model(observations)
        pi_loss = self.pi_model.backward(observations, actions, target_q_values, state_values, self.tau.value())
        
        info = {"v_loss": v_loss, "pi_loss": pi_loss}
        
        # 每隔一段时间更新温度
        if self.update_tau_freq != 0 and self.update_count % self.update_tau_freq == 0:
            with torch.no_grad():
                probs = self.target_pi_model(observations)
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
        super().save_weights(path, prefix, suffix)
        self.pi_model.save_weights(path, prefix, suffix)
        self.target_pi_model.save_weights(path, prefix + "target_", suffix)
        self.tau.save_weights(path, prefix, suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        res1 = super().load_weights(path, prefix, suffix)
        res2 = self.pi_model.load_weights(path, prefix, suffix)
        res3 = self.target_pi_model.load_weights(path, prefix + "target_", suffix)
        res4 = self.tau.load_weights(path, prefix, suffix)
        return res1 and res2 and res3 and res4
    
    def get_distilling_model(self):
        return self.distilling_model
    
    def get_models(self):
        return [self]
