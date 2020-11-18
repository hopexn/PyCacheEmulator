import copy

from ..core import Agent
from ..memory import *
from ..model import RLModel, Temperature
from ..nn import EWMLP, GaussianEWMLP
from ..policy import *
from cache_emu.utils import torch_utils as ptu



class EWSacPiModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr, log_std_min=-10, log_std_max=2):
        super(EWSacPiModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        
        self.net = GaussianEWMLP(feature_dim, hidden_layer_units, log_std_min, log_std_max)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        mu, log_std = self.net.forward(x)
        z = mu + log_std.exp() * torch.randn_like(mu)
        action = torch.tanh(z)
        return action
    
    def evaluate(self, x):
        mu, log_std = self.net.forward(x)
        std = log_std.exp()
        z = mu + std * torch.randn_like(mu)
        action = torch.tanh(z)
        
        log_probs = ptu.gaussian_likelihood(z, mu, log_std, std)
        log_probs -= torch.log(-torch.pow(action, 2) + 1 + 1e-6)
        return action, log_probs
    
    def backward(self, observations, q_model, tau):
        # 更新策略网络
        soft_actions, log_probs = self.evaluate(observations)
        soft_q_values = q_model.forward(
            torch.cat([observations, soft_actions.unsqueeze(-1)], dim=2)
        )
        loss = -torch.mean(soft_q_values - tau * log_probs)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.cpu().item()


class EWSacQModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr=3e-4):
        super(EWSacQModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        
        self.net = EWMLP(feature_dim, hidden_layer_units)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()


class EWSAC(Agent):
    def __init__(self, content_dim, feature_dim, memory_size=100000, batch_size=32,
                 hidden_layer_units=[32, 8], lr=3e-4, gamma=0.99, target_update=2,
                 log_std_min=-20, log_std_max=2, log_tau=-1, min_entropy_ratio=0.1,
                 **kwargs):
        super().__init__(content_dim, feature_dim, **kwargs)
        
        # 创建Actor网络
        self.pi_model = EWSacPiModel(content_dim, feature_dim, hidden_layer_units, lr)
        
        # 创建Critic网络
        self.q_model = EWSacQModel(content_dim, feature_dim + 1, hidden_layer_units, lr)
        self.target_q_model = copy.deepcopy(self.q_model)
        
        # 动作策略参数
        self.policy = DecayEpsGreedyQPolicy(content_dim)
        
        # 创建ReplayBuffer
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        
        # RL超参数
        self.gamma = gamma
        self.target_update = target_update
        
        # SAC超参数
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 设置最小熵为 ratio * max_entropy, 其中 max_entropy = log(content_dim)
        self.tau = Temperature(log_tau=log_tau, min_entropy=min_entropy_ratio * np.log(content_dim))
        # 更新温度的频率，值为0时温度为定值
        self.update_tau_freq = 2
        
        self.update_count = 0
        self.last_pi_values = None
    
    def forward(self, observation):
        with torch.no_grad():
            pi_values = self.pi_model.forward(observation)
        self.last_pi_values = pi_values
        return self.policy.select_action(ptu.get_numpy(self.last_pi_values))
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
        observation = observation[:self.content_dim].unsqueeze(0)
        action = action[:self.content_dim].unsqueeze(0)
        reward = reward[:self.content_dim].unsqueeze(0)
        next_observation = next_observation[:self.content_dim].unsqueeze(0)
        
        action = self.last_pi_values[:self.content_dim].unsqueeze(0)
        self.memory.store_transition(observation, action, reward, next_observation)
        
        observations, actions, rewards, next_observations = self.memory.sample_batch(self.batch_size)
        
        actions = actions.float()
        
        info = self._update(observations, actions, rewards, next_observations)
        self._update_target()
        return info
    
    def _update(self, observations, actions, rewards, next_observations):
        # 更新Q网络
        with torch.no_grad():
            next_actions, next_log_probs = self.pi_model.evaluate(next_observations)
            next_q_values = self.target_q_model.forward_target(
                torch.cat([next_observations, next_actions.unsqueeze(-1)], dim=2)
            )
            next_state_value = next_q_values - self.tau * next_log_probs
            target_q_values = rewards + self.gamma * next_state_value
        
        q_loss = self.q_model.fit(torch.cat([observations, actions.unsqueeze(-1)], dim=-1), target_q_values)
        
        # 更新策略网络
        pi_loss = self.pi_model.backward(observations, self.q_model, self.tau)
        
        info = {"q_loss": q_loss, "pi_loss": pi_loss}
        
        # 每隔一段时间更新温度
        if self.update_tau_freq != 0 and self.update_count % self.update_tau_freq == 0:
            next_probs = torch.exp(next_log_probs)
            tau_info = self.tau.backward(next_probs)
            info.update(tau_info)
        
        return info
    
    def _update_target(self):
        if self.target_update > 1 and self.update_count % self.target_update == 0:
            ptu.copy_model_params_from_to(self.q_model, self.target_q_model)
        else:
            ptu.soft_update_from_to(self.q_model, self.target_q_model, self.target_update)
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        self.pi_model.save_weights(path, prefix, suffix)
        self.q_model.save_weights(path, prefix, suffix)
        self.target_q_model.save_weights(path, prefix + "target_", suffix)
        self.tau.save_weights(path, prefix, suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        res1 = self.pi_model.load_weights(path, prefix, suffix)
        res2 = self.q_model.load_weights(path, prefix, suffix)
        res3 = self.target_q_model.load_weights(path, prefix + "target_", suffix)
        res4 = self.tau.load_weights(path, prefix, suffix)
        return res1 and res2 and res3 and res4
    
    def get_distilling_model(self):
        return self.pi_model
    
    def get_models(self):
        return [self]
