import copy

from ..core import Agent
from ..memory import *
from ..model import RLModel
from ..policy import *
from ..utils import torch_utils as ptu


class EWQModel(RLModel):
    def __init__(self, content_dim, feature_dim, hidden_layer_units: list, lr=3e-4):
        super(EWQModel, self).__init__(content_dim, feature_dim, hidden_layer_units, lr)
        self.content_dim = content_dim
        
        self.net = ptu.build_mlp(feature_dim, hidden_layer_units, 2)
        
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        self.loss_fn = torch.nn.MSELoss()


class EWDQN(Agent):
    def __init__(self, content_dim, feature_dim, memory_size=10000, batch_size=32,
                 hidden_layer_units=[32, 8], lr=3e-4,
                 gamma=0.99, target_update=2, enable_double_q=False, **kwargs):
        
        super().__init__(content_dim, feature_dim, **kwargs)
        
        print("Enable Double Q: ", enable_double_q)
        
        self.q_model = EWQModel(content_dim, feature_dim, hidden_layer_units, lr)
        self.target_q_model = copy.deepcopy(self.q_model)
        
        # 创建ReplayBuffer
        self.memory = Memory(memory_size)
        
        # 动作策略参数
        self.policy = GreedyQPolicy(content_dim)
        # self.policy = DecayEpsGreedyQPolicy(content_dim, eps_min=1e-4)
        
        # RL超参数
        self.gamma = gamma
        self.target_update = target_update
        self.batch_size = batch_size
        self.use_double_q = enable_double_q
        
        self.update_count = 0
    
    def forward(self, observation):
        with torch.no_grad():
            q_values = self.q_model.forward(observation)
        return self.policy.select_action(ptu.get_numpy(q_values[:, 1]))
    
    def backward(self, observation, action, reward, next_observation):
        self.update_count += 1
        
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
            if self.use_double_q:
                next_q_values = self.target_q_model.forward_target(next_observations)
                next_indices = self.q_model.forward(next_observations).max(dim=-1, keepdims=True)[1]
                next_state_values = next_q_values.gather(dim=-1, index=next_indices).squeeze(-1)
            else:
                next_state_values = self.target_q_model.forward_target(next_observations).max(dim=-1)[0]
            
            target_q_values = rewards + self.gamma * next_state_values
            
            q_values = self.q_model.forward(observations)
            actions_onehot = ptu.to_catagorical(actions, 2)
            new_q_values = q_values.masked_scatter(actions_onehot, target_q_values)
        
        q_loss = self.q_model.fit(observations, new_q_values)
        return {"q_loss": q_loss}
    
    def _update_target(self):
        if self.target_update > 1 and self.update_count % self.target_update == 0:
            ptu.copy_model_params_from_to(self.q_model, self.target_q_model)
        else:
            ptu.soft_update_from_to(self.q_model, self.target_q_model, self.target_update)
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        self.q_model.save_weights(path, prefix, suffix)
        self.target_q_model.save_weights(path, prefix + "target_", suffix)
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        res1 = self.q_model.load_weights(path, prefix, suffix)
        res2 = self.target_q_model.load_weights(path, prefix + "target_", suffix)
        return res1 and res2
    
    def get_distilling_model(self):
        return self.q_model
    
    def get_models(self):
        return [self]
