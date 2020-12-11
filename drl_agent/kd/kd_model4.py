import numpy as np

from drl_agent.ewdrl.memory import Memory
from .common import *


class KDWeights4(KDWeightsIF):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-4, **kwargs):
        super(KDWeights4, self).__init__(num_agents, n_neighbors, kdw_lr, **kwargs)
        
        self.alpha = float(kwargs.get('alpha', 0.1))
        
        kdw_memory_size = kwargs.get("kdw_memory_size", 10000)
        kdw_hidden_layer_units = kwargs.get("kdw_hidden_layer_units", [32, 8])
        self.critic = ptu.build_mlp(self.n_neighbors + 1, kdw_hidden_layer_units, 1)
        self.target_critic = ptu.build_mlp(self.n_neighbors + 1, kdw_hidden_layer_units, 1)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = Memory(kdw_memory_size)
        
        self.noise = 1e-5
        self.noise_clip = 0.5
        
        self.last_reward = None
        self.last_observation = None
        self.last_action = None
        self.observation = None
        
        # torch.autograd.set_detect_anomaly(True)
    
    def forward(self, losses: torch.tensor, reward, **kwargs):
        self.ref_ws = self.ws.softmax(dim=0)
        
        self.indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        # indices = np.random.choice(np.arange(self.num_agents), self.n_neighbors,
        #                            replace=False, p=ptu.get_numpy(self.ref_ws))
        self.indices = ptu.tensor(self.indices, dtype=torch.long)
        
        losses_selected = losses.index_select(dim=0, index=self.indices)
        ref_ws_selected = self.ref_ws.index_select(dim=0, index=self.indices).detach()
        loss = (losses_selected * ref_ws_selected).sum().unsqueeze(0)
        
        self.last_reward = ptu.float_tensor(reward).unsqueeze(0)
        self.last_observation = self.observation
        self.observation = torch.cat([losses_selected.detach(), loss.detach()], dim=-1).unsqueeze(0)
        self.last_action = self.indices.unsqueeze(0)
        
        entropy = - (self.ref_ws * self.ref_ws.log()).sum()
        
        return loss - self.alpha * entropy
    
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
        
        self.optim.zero_grad()
        pi_loss.backward()
        self.optim.step()
    
    def step(self):
        super().step()
        if self.last_observation is not None and self.last_action is not None:
            self.backward(self.last_observation, self.last_action, self.last_reward, self.observation)

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