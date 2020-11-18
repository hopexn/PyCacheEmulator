import torch

import cache_emu.utils.torch_utils as ptu
from .ewdnn import EWDNN
from ..model import RLModel
from ..policy import ProbabilisticQPolicy


class EWDdpgPiModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr):
        super(EWDdpgPiModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        
        self.net = ptu.build_mlp(content_dim * feature_dim, hidden_layer_units, content_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        x = x.reshape((-1, self.feature_dim * self.content_dim))
        return torch.softmax(self.net(x), dim=-1)


class EWDDPG(EWDNN):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32, hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2, **kwargs):
        super().__init__(content_dim + 1, feature_dim + 1, memory_size, batch_size, hidden_layer_units, lr, gamma,
                         target_update, **kwargs)
        self.pi_model = EWDdpgPiModel(self.content_dim, feature_dim, hidden_layer_units, lr)
        self.policy = ProbabilisticQPolicy(content_dim)
        self.last_action = None
    
    def forward(self, observation):
        with torch.no_grad():
            probs = self.pi_model.forward(observation).squeeze(0)
        self.last_action = probs
        return self.policy.select_action(ptu.get_numpy(probs))
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
        observation = observation.unsqueeze(0)
        # action = action.unsqueeze(0)
        action = self.last_action.unsqueeze(0)
        reward = reward.unsqueeze(0)
        next_observation = next_observation.unsqueeze(0)
        
        self.memory.store_transition(observation, action, reward, next_observation)
        observations, actions, rewards, next_observations = self.memory.sample_batch(self.batch_size)
        
        # update q value
        next_actions = self.pi_model(next_observations).reshape((-1, self.content_dim, 1))
        target_q_values = rewards + self.gamma * self.target_v_model(
            torch.cat([next_observations, next_actions], dim=-1))
        
        observations_actions = torch.cat([observations, actions.unsqueeze(-1).float()], dim=-1)
        v_loss = self.v_model.fit(observations_actions[:, :self.content_dim - 1],
                                  target_q_values[:, :self.content_dim - 1])
        self._update_target()
        
        # update pi model
        action_eval = self.pi_model(observations).reshape((-1, self.content_dim, 1))
        q_value = self.v_model(torch.cat([observations, action_eval], dim=-1))
        pi_loss = -q_value.mean()
        
        self.pi_model.zero_grad()
        pi_loss.backward()
        self.pi_model.step()
        
        return {"v_loss": v_loss, "pi_loss": pi_loss.cpu().item()}
