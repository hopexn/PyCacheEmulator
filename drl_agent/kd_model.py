import os

import numpy as np
import torch
from torch import nn

from drl_agent.ewdrl.memory import Memory
from py_cache_emu import torch_utils as  ptu


class TemperatureModel(nn.Module):
    def __init__(self, log_tau, min_entropy, tau_lr=3e-3, log_tau_clip=(-4, 3), **kwargs):
        super(TemperatureModel, self).__init__()
        
        self.min_entropy = min_entropy
        self.log_tau = nn.Parameter(ptu.tensor(float(log_tau)), requires_grad=True)
        self.optim = torch.optim.Adam([self.log_tau], lr=tau_lr)
        self.log_tau_clip = log_tau_clip
    
    def forward(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def __mul__(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def value(self):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip))
    
    def backward(self, probs):
        tau = torch.exp(self.log_tau.clamp(*self.log_tau_clip))
        # 为了避免log(probs)出现nan, probs需要加上eps
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
        tau_loss = tau * (entropy.detach() - self.min_entropy)
        
        self.optim.zero_grad()
        tau_loss.backward()
        self.optim.step()
        
        return {"tau"     : tau.cpu().item(),
                "entropy" : entropy.cpu().item(),
                "tau_loss": tau_loss.cpu().item()}
    
    def save_weights(self, path, prefix="", suffix=""):
        ptu.save_model(self, os.path.join(path, prefix + "tau" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        res = ptu.load_model(self, os.path.join(path, prefix + "tau" + suffix + ".pt"))
        return res


class KDWeights(nn.Module):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-4, **kwargs):
        super(KDWeights, self).__init__()
        self.num_agents = num_agents
        self.lr = kdw_lr
        
        self.kd_mode = int(kwargs.get("kd_mode", 0))
        self.alpha = float(kwargs.get('alpha', 0.1))
        self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
        self.ws_optim = torch.optim.Adam([self.ws], lr=self.lr)
        self.ws_sm = None
        
        self.n_neighbors = self.num_agents if n_neighbors <= 0 else n_neighbors
        self.n_neighbors = min(self.n_neighbors, self.num_agents)
        
        hidden_layer_units = [32, 8]
        self.critic = ptu.build_mlp(self.n_neighbors + 1, hidden_layer_units, 1)
        self.target_critic = ptu.build_mlp(self.n_neighbors + 1, hidden_layer_units, 1)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.memory = Memory(10000)
        
        self.noise = 1e-5
        self.noise_clip = 0.5
        
        self.last_reward = None
        self.last_observation = None
        self.last_action = None
        self.ref_ws = None
        self.indices = None
        self.observation = None
        # torch.autograd.set_detect_anomaly(True)
    
    def forward(self, losses: list, **kwargs):
        ws_sm = self.ws.softmax(dim=0)
        
        indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        # indices = np.random.choice(np.arange(self.num_agents), self.n_neighbors,
        #                            replace=False, p=ptu.get_numpy(ws_sm))
        
        losses = torch.cat([losses[i].unsqueeze(0) for i in indices], dim=0)
        indices = ptu.tensor(indices, dtype=torch.long)
        
        self.indices = indices
        if self.kd_mode == 1:
            loss = sum(losses) / self.n_neighbors
        else:
            ws_ms_selected = ws_sm.index_select(dim=0, index=indices).detach()
            loss = (losses * ws_ms_selected).sum().unsqueeze(0)
            
            self.last_reward = ptu.float_tensor(kwargs.get('reward', 0)).unsqueeze(0)
            self.last_observation = self.observation
            self.observation = torch.cat([losses.detach(), loss.detach()], dim=-1).unsqueeze(0)
            self.last_action = indices.unsqueeze(0)
        
        return loss + self.alpha * (ws_sm * ws_sm.log()).mean()
    
    def backward(self, observation, action, reward, next_observation):
        self.memory.store_transition(observation, action, reward, next_observation)
        
        observations, actions, rewards, next_observations = self.memory.sample_batch(512)
        next_observations = next_observations[:, :-1]
        
        ws_sm = self.ws.softmax(dim=0)
        
        with torch.no_grad():
            weights = ws_sm.index_select(dim=0, index=actions.flatten()).reshape(actions.shape)
            next_actions_eval = (next_observations * weights).sum(dim=-1, keepdim=True)
            noise = self.noise * torch.randn_like(next_actions_eval).clamp(-self.noise_clip, self.noise_clip)
            next_actions_eval = next_actions_eval + noise
            
            target_q_values = rewards.unsqueeze(-1) + 0.99 * self.target_critic.forward(
                torch.cat([next_observations, next_actions_eval], dim=-1))
        
        q_values = self.critic.forward(observations)
        q_loss = torch.nn.functional.mse_loss(q_values, target_q_values.detach())
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        
        ptu.soft_update_from_to(self.critic, self.target_critic, 0.995)
        
        weights = ws_sm.index_select(dim=0, index=actions.flatten()).reshape(actions.shape)
        actions_eval = (next_observations * weights).sum(dim=-1, keepdim=True)
        pi_loss = -self.critic.forward(torch.cat([next_observations, actions_eval], dim=-1)).mean()
        
        self.ws_optim.zero_grad()
        pi_loss.backward()
        self.ws_optim.step()
    
    def zero_grad(self):
        if self.kd_mode > 1:
            self.ws_optim.zero_grad()
    
    def step(self):
        if self.kd_mode > 1:
            self.ws_optim.step()
            if self.last_observation is not None and self.last_action is not None:
                self.backward(self.last_observation, self.last_action, self.last_reward, self.observation)
    
    def get_dict(self):
        ws_sm = ptu.get_numpy(self.ws.softmax(dim=0))
        return {"KDW{}".format(i): ws_sm[i] for i in range(self.num_agents)}
    
    def save_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
        ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
        ptu.save_model(self.critic, os.path.join(path, prefix + "kd_weights_critic" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
        res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
        res = res and ptu.load_model(self.critic, os.path.join(path, prefix + "kd_weights_critic" + suffix + ".pt"))
        return res
    
    # class KDWeights(nn.Module):
    #     def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-3, **kwargs):
    #         super(KDWeights, self).__init__()
    #         self.num_agents = num_agents
    #         self.lr = kdw_lr
    #
    #         self.kd_mode = int(kwargs.get("kd_mode", 0))
    #         self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
    #         self.ws_optim = torch.optim.Adam([self.ws], lr=self.lr)
    #         self.ws_sm = None
    #
    #         self.n_neighbors = self.num_agents if n_neighbors <= 0 else n_neighbors
    #         self.n_neighbors = min(self.n_neighbors, self.num_agents)
    #
    #         self.critic = ptu.build_mlp(self.num_agents + self.n_neighbors, [32, 8], 1)
    #         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
    #
    #         self.memory = Memory(5000)
    #
    #         self.last_reward = None
    #         self.last_observation = None
    #         self.last_action = None
    #         self.ref_ws = None
    #         self.indices = None
    #         self.action_noise = 1e-3
    #
    #     def forward(self, losses: list, **kwargs):
    #         # indices = np.random.choice(
    #         #     np.arange(self.num_agents), self.n_neighbors,
    #         #     replace=False, p=ptu.get_numpy(self.ws_sm)
    #         # )
    #
    #         indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
    #         loss = ptu.float_tensor(0, requires_grad=True)
    #         losses = [losses[i] for i in indices]
    #
    #         self.indices = indices
    #         if self.kd_mode == 1:
    #             loss = sum(losses) / self.n_neighbors
    #         else:
    #             ref_ws = (self.ws + self.action_noise * torch.randn_like(self.ws)).softmax(dim=0)
    #             for i, l in zip(indices, losses):
    #                 loss = loss + l * ref_ws[i]
    #
    #             observation = torch.tensor(losses)
    #
    #             if self.last_reward is not None and self.last_observation is not None and self.last_action is not None:
    #                 self.backward(self.last_observation, self.last_action,
    #                               kwargs.get('reward', 0), torch.cat([observation, ref_ws]))
    #
    #             self.last_observation = observation
    #             self.last_action = self.ref_ws = ref_ws
    #
    #         return loss
    #
    #     def backward(self, observation, action, reward, next_observation):
    #         self.memory.store_transition(observation, action, reward, next_observation)
    #         observations, actions, rewards, next_observations = self.memory.sample_batch()
    #
    #         q_values = self.critic.forward(torch.cat([observations, actions], dim=-1))
    #         with torch.no_grad():
    #             target_q_values = rewards + 0.99 * self.critic.forward(next_observations)
    #
    #         q_loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    #         self.critic_optim.zero_grad()
    #         q_loss.backward()
    #         self.critic_optim.step()
    #
    #         ws_sm = (self.ws + self.action_noise * torch.randn_like(self.ws)).softmax(dim=0).unsqueeze(0)
    #         ws_sm = ws_sm.repeat(observations.shape[0], 1)
    #         pi_loss = -self.critic(torch.cat([observations, ws_sm], dim=-1)).mean()
    #
    #         self.ws_optim.zero_grad()
    #         pi_loss.backward()
    #         self.ws_optim.step()
    #
    #     def zero_grad(self):
    #         if self.kd_mode > 1:
    #             self.ws_optim.zero_grad()
    #
    #     def step(self):
    #         if self.kd_mode > 1:
    #             self.ws_optim.step()
    #
    #     def get_dict(self):
    #         if self.ref_ws is not None:
    #             return {"KDW{}".format(i): self.ws[i] for i in self.indices}
    #         else:
    #             return {}
    #
    #     def save_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         ptu.save_model(self.critic, os.path.join(path, prefix + "kd_weights_critic" + suffix + ".pt"))
    #
    #     def load_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         ptu.load_model(self.critic, os.path.join(path, prefix + "kd_weights_critic" + suffix + ".pt"))
    #         return res
    
    # class KDWeights(nn.Module):
    #     def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-3, **kwargs):
    #         super(KDWeights, self).__init__()
    #         self.num_agents = num_agents
    #         self.lr = kdw_lr
    #
    #         self.kd_mode = int(kwargs.get("kd_mode", 0))
    #         self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
    #         self.optim = torch.optim.Adam([self.ws], lr=self.lr)
    #         self.ws_sm = None
    #
    #         self.n_neighbors = self.num_agents if n_neighbors <= 0 else n_neighbors
    #         self.n_neighbors = min(self.n_neighbors, self.num_agents)
    #
    #         # 定义状态空间动作空间
    #         self.observation_space = gym.spaces.Box(
    #             low=np.zeros((self.n_neighbors,), dtype=np.float32),
    #             high=np.ones((self.n_neighbors,), dtype=np.float32)
    #         )
    #         self.action_space = gym.spaces.Box(
    #             low=np.zeros((self.n_neighbors,), dtype=np.float32),
    #             high=np.ones((self.n_neighbors,), dtype=np.float32)
    #         )
    #
    #         self.observation_shape = gym.spaces.Space
    #         self.rlagent = td3.TD3Agent(self.observation_space, self.action_space,
    #                                     nb_steps_warm_up=1,
    #                                     action_noise=1e-4,
    #                                     target_noise=2e-4,
    #                                     batch_size=128
    #                                     )
    #
    #         self.last_reward = None
    #         self.last_observation = None
    #         self.last_action = None
    #         self.ref_ws = None
    #         self.indices = None
    #
    #     def forward(self, losses: list, **kwargs):
    #         # indices = np.random.choice(
    #         #     np.arange(self.num_agents), self.n_neighbors,
    #         #     replace=False, p=ptu.get_numpy(self.ws_sm)
    #         # )
    #         indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
    #         loss = ptu.float_tensor(0, requires_grad=True)
    #         losses = [losses[i] for i in indices]
    #
    #         self.indices = indices
    #         if self.kd_mode == 1:
    #             loss = sum(losses)
    #         else:
    #             observation = torch.tensor(losses).detach()
    #             observation = ptu.get_numpy(observation)
    #
    #             ref_ws = self.rlagent.forward(observation)
    #             for ref_w, l in zip(ref_ws, losses):
    #                 loss = loss + l * ref_w
    #
    #             if self.last_reward is not None and self.last_observation is not None and self.last_action is not None:
    #                 self.rlagent.backward(self.last_observation, self.last_action, kwargs.get("reward", 0), False,
    #                                       observation)
    #
    #             self.last_action = ref_ws
    #             self.last_observation = observation
    #             self.ref_ws = ref_ws
    #
    #         return loss
    #
    #     def zero_grad(self):
    #         if self.kd_mode > 1:
    #             self.optim.zero_grad()
    #
    #     def step(self):
    #         if self.kd_mode > 1:
    #             self.optim.step()
    #
    #     def get_dict(self):
    #         if self.ref_ws is not None:
    #             return {"KDW{}".format(i): w for i, w in zip(self.indices, self.ref_ws)}
    #         else:
    #             return {}
    #
    #     def save_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         self.rlagent.save_weights(os.path.join(path, prefix + "kd_weights" + suffix))
    #
    #     def load_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         self.rlagent.load_weights(os.path.join(path, prefix + "kd_weights" + suffix))
    #         return res
    
    # class KDWeights(nn.Module):
    #     def __init__(self, num_agents, kdw_lr=3e-3, kwd_log_tau=0, min_entropy_ratio=0.99, alpha=0.1, **kwargs):
    #         super(KDWeights, self).__init__()
    #         self.num_agents = num_agents
    #         self.lr = kdw_lr
    #
    #         self.tau = TemperatureModel(log_tau=kwd_log_tau,
    #                                     min_entropy=min_entropy_ratio * np.log(num_agents),
    #                                     **kwargs)
    #
    #         self.kd_mode = int(kwargs.get("kd_mode", 0))
    #         self.alpha = alpha
    #         self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
    #         self.optim = torch.optim.Adam([self.ws], lr=self.lr)
    #         self.ws_sm = None
    #
    #     def forward(self, losses: list, k=0):
    #         k = self.num_agents if k <= 0 else k
    #         k = min(k, self.num_agents)
    #         self.ws_sm = (self.ws / self.tau.value().detach()).softmax(dim=0)
    #
    #         indices = np.random.choice(
    #             np.arange(self.num_agents), k,
    #             replace=False, p=ptu.get_numpy(self.ws_sm)
    #         )
    #
    #         loss = ptu.float_tensor(0, requires_grad=True)
    #         loss_ws = ptu.float_tensor(0, requires_grad=True)
    #
    #         if self.kd_mode == 1:
    #             for idx in indices:
    #                 loss = loss + losses[idx] * self.ws_sm[idx]
    #         elif self.kd_mode == 2:
    #             for idx in indices:
    #                 loss = loss + losses[idx] * self.ws_sm[idx]
    #         elif self.kd_mode == 3:
    #             sum_exp_minus_loss = 0
    #             for idx in indices:
    #                 loss = loss + losses[idx]
    #                 exp_minus_loss = (-losses[idx].detach()).exp().detach()
    #                 loss_ws = loss_ws + self.ws_sm[idx].log() * exp_minus_loss
    #                 sum_exp_minus_loss = sum_exp_minus_loss + exp_minus_loss
    #
    #             loss = loss / k + self.alpha * loss_ws / sum_exp_minus_loss
    #
    #         return loss
    #
    #     # def forward(self, losses: list, k=0):
    #     #     # Backup
    #     #     k = self.num_agents if k <= 0 else k
    #     #     k = min(k, self.num_agents)
    #     #
    #     #     indices = np.random.permutation(np.arange(self.num_agents))[:k]
    #     #     loss = ptu.float_tensor(0, requires_grad=True)
    #     #
    #     #     ws_sm = (self.ws / self.tau.value().detach()).softmax(dim=0)
    #     #
    #     #     for idx in indices:
    #     #         loss = loss + losses[idx] * ws_sm[idx]
    #     #     if self.kd_mode == 2:
    #     #         self.tau.backward(ws_sm)
    #     #
    #     #     return loss / k
    #
    #     def zero_grad(self):
    #         if self.kd_mode > 1:
    #             self.optim.zero_grad()
    #
    #     def step(self):
    #         if self.kd_mode > 1:
    #             self.optim.step()
    #             self.tau.backward(self.ws_sm)
    #
    #     def get_dict(self):
    #         softmax_ws = ptu.get_numpy((self.ws / self.tau.value()).softmax(dim=0))
    #         return {"KDW{}".format(i): w for i, w in enumerate(softmax_ws)}
    #
    #     def save_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         self.tau.save_weights(os.path.join(path), prefix=prefix, suffix=suffix)
    #
    #     def load_weights(self, path, prefix="", suffix=""):
    #         suffix = "-" + str(self.num_agents) + suffix + "_{}".format(self.kd_mode)
    #         res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    #         self.tau.load_weights(os.path.join(path), prefix=prefix, suffix=suffix)
    #         return res
    #
