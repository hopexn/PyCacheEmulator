import numpy as np

from drl_agent.ewdrl.memory import Memory
from .common import *


class KDWeights4(KDWeightsIF):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-4, **kwargs):
        super(KDWeights4, self).__init__(num_agents, n_neighbors, kdw_lr, **kwargs)
        
        self.alpha = float(kwargs.get('alpha', 0.1))
        
        kdw_memory_size = kwargs.get("kdw_memory_size", 10000)
        kdw_hidden_layer_units = kwargs.get("kdw_hidden_layer_units", [32, 8])
        self.critic = ptu.build_mlp(self.n_neighbors * 3 + 1, kdw_hidden_layer_units, 1)
        self.target_critic = ptu.build_mlp(self.n_neighbors * 3 + 1, kdw_hidden_layer_units, 1)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = Memory(kdw_memory_size)
        
        self.noise = 1e-5
        self.noise_clip = 0.5
        
        self.last_reward = None
        self.last_observation = None
        self.last_action = None
        self.observation = None
        
        # torch.autograd.set_detect_anomaly(True)
    
    def forward(self, losses, reward, rewards, sample_rewards, **kwargs):
        self.ref_ws = self.ws.softmax(dim=0)
        
        if self.random_index:
            self.indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        else:
            self.indices = np.random.choice(np.arange(self.num_agents), self.n_neighbors,
                                            replace=False, p=ptu.get_numpy(self.ref_ws))
        
        self.indices = ptu.tensor(self.indices, dtype=torch.long)
        
        losses_selected = losses.index_select(dim=0, index=self.indices)
        sample_rewards_selected = sample_rewards.index_select(dim=0, index=self.indices).detach()
        rewards_selected = rewards.index_select(dim=0, index=self.indices).detach()
        
        ref_ws_selected = self.ref_ws.index_select(dim=0, index=self.indices).detach()
        loss = (losses_selected * ref_ws_selected).sum().unsqueeze(0)
        
        self.last_reward = reward.detach()
        self.last_observation = self.observation
        self.observation = torch.cat([
            losses_selected.clone().detach(),
            sample_rewards_selected.clone().detach(),
            rewards_selected.clone().detach(),
            loss.clone().detach()
        ], dim=-1).unsqueeze(0)
        
        self.last_action = self.indices.unsqueeze(0).clone().detach()
        
        entropy = - (self.ref_ws * self.ref_ws.log()).sum()
        
        return loss - self.alpha * entropy
    
    def backward(self, observation, action, reward, next_observation):
        self.memory.store_transition(observation, action, reward, next_observation)
        
        observations, actions, rewards, next_observations = self.memory.sample_batch(512)
        ws_sm = self.ws.softmax(dim=0)
        
        with torch.no_grad():
            weights = ws_sm.index_select(dim=0, index=actions.flatten()).reshape(actions.shape)
            next_actions_eval = (next_observations[:, :self.n_neighbors] * weights).sum(dim=-1, keepdim=True)
            noise = self.noise * torch.randn_like(next_actions_eval).clamp(-self.noise_clip, self.noise_clip)
            next_actions_eval = next_actions_eval + noise
            
            target_q_values = rewards.unsqueeze(-1) + 0.99 * self.target_critic.forward(
                torch.cat([next_observations[:, :-1], next_actions_eval], dim=-1))
        
        q_values = self.critic.forward(observations)
        q_loss = torch.nn.functional.mse_loss(q_values, target_q_values.detach())
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        
        ptu.soft_update_from_to(self.critic, self.target_critic, 0.995)
        
        weights = ws_sm.index_select(dim=0, index=actions.flatten()).reshape(actions.shape)
        actions_eval = (next_observations[:, :self.n_neighbors] * weights).sum(dim=-1, keepdim=True)
        pi_loss = -self.critic.forward(torch.cat([next_observations[:, :-1], actions_eval], dim=-1)).mean()
        
        self.optim.zero_grad()
        pi_loss.backward()
        self.optim.step()
    
    def step(self):
        super().step()
        if self.last_observation is not None and self.last_action is not None:
            self.backward(self.last_observation, self.last_action, self.last_reward, self.observation)
