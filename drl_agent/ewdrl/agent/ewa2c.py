import torch

import cache_emu.utils.torch_utils as ptu
from .ewdnn import EWDNN
from ..model import RLModel
from ..nn import EWMLP
from ..policy import *


class EWA2cPiModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr):
        super(EWA2cPiModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        
        self.net = ptu.build_mlp(content_dim * feature_dim, hidden_layer_units, content_dim)
        self.net = EWMLP(feature_dim, hidden_layer_units)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.net(x).softmax(dim=-1)


class EWA2C(EWDNN):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32, hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2, **kwargs):
        super().__init__(content_dim + 1, feature_dim, memory_size, batch_size, hidden_layer_units, lr, gamma,
                         target_update, **kwargs)
        self.feature_dim = feature_dim
        self.pi_model = EWA2cPiModel(content_dim + 1, feature_dim, hidden_layer_units, lr)
        self.policy = GreedyQPolicy(content_dim)
        self.last_action = None
    
    def forward(self, observation):
        with torch.no_grad():
            probs = self.pi_model.forward(observation).squeeze(0)
        self.last_action = probs
        
        # probs = ptu.get_numpy(probs)
        # action = np.random.choice(np.arange(len(probs)), p=probs)
        # action_oh = np.ones_like(probs, dtype=np.bool)
        # action_oh[action] = 0
        # return action_oh
        
        return self.policy.select_action(ptu.get_numpy(probs))
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
        observation = observation.unsqueeze(0)
        action = action.unsqueeze(0)
        reward = reward.unsqueeze(0)
        next_observation = next_observation.unsqueeze(0)
        
        self.memory.store_transition(observation, action, reward, next_observation)
        observations, actions, rewards, next_observations = self.memory.sample_batch(self.batch_size)
        
        # update v model
        with torch.no_grad():
            next_values = self.target_v_model.forward_target(next_observations)
            target_values = rewards + self.gamma * next_values
        v_loss = self.v_model.fit(observations[:, :self.content_dim - 1], target_values[:, :self.content_dim - 1])
        self._update_target()
        
        # update pi model
        probs = self.pi_model(observations)
        with torch.no_grad():
            q_values = self.v_model(observations)
        pi_loss = - (q_values * torch.log(probs + 1e-6)).mean()
        
        self.pi_model.zero_grad()
        pi_loss.backward()
        self.pi_model.step()
        
        return {"v_loss": v_loss, "pi_loss": pi_loss.cpu().item()}
    
    def get_distilling_model(self):
        return self.v_model