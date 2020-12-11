import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from py_cache_emu import torch_utils as ptu
from .memory import Memory


class ActorNet(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_layer_units: list, lr):
        super(ActorNet, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.hidden_layer_units = hidden_layer_units
        self.lr = lr
        
        self.obs_dim = np.prod(self.observation_shape).item()
        self.act_dim = np.prod(self.action_shape).item()
        
        layers = self.build_layers()
        
        self.net = nn.Sequential(*layers).to(ptu.device)
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
    
    def init_optim(self):
        self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
    
    def build_layers(self):
        layer_units = [self.obs_dim]
        layer_units.extend(self.hidden_layer_units)
        
        layers = []
        for in_features, out_features in zip(layer_units[:-1], layer_units[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_units[-1], self.act_dim))
        layers.append(nn.Softmax(dim=-1))
        return layers
    
    def forward(self, obs):
        return self.net(obs)
    
    def backward(self, critic_net, obs):
        action = self.forward(obs)
        loss = -critic_net.Q1(obs, action).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()


class CriticNet(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_layer_units: list, lr):
        super(CriticNet, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.hidden_layer_units = hidden_layer_units
        self.lr = lr
        
        self.obs_dim = np.prod(self.observation_shape).item()
        self.act_dim = np.prod(self.action_shape).item()
        
        self.q_net1 = nn.Sequential(*self.build_layers()).to(ptu.device)
        self.q_net2 = nn.Sequential(*self.build_layers()).to(ptu.device)
        
        self.optim1 = torch.optim.Adam(self.q_net1.parameters(), self.lr)
        self.optim2 = torch.optim.Adam(self.q_net1.parameters(), self.lr)
    
    def init_optim(self):
        self.optim1 = torch.optim.Adam(self.q_net1.parameters(), self.lr)
        self.optim2 = torch.optim.Adam(self.q_net1.parameters(), self.lr)
    
    def build_layers(self):
        layer_units = [self.obs_dim + self.act_dim]
        layer_units.extend(self.hidden_layer_units)
        
        layers = []
        for in_features, out_features in zip(layer_units[:-1], layer_units[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_units[-1], 1))
        return layers
    
    def forward(self, observation, action):
        input_tensor = torch.cat([observation, action], dim=1)
        return self.q_net1(input_tensor), self.q_net2(input_tensor)
    
    def Q1(self, observation, action):
        input_tensor = torch.cat([observation, action], dim=1)
        return self.q_net1(input_tensor)
    
    def backward(self, observations, actions, target_q_values):
        target_q_values = target_q_values.detach()
        
        q_values1, q_values2 = self.forward(observations, actions)
        
        loss = F.mse_loss(q_values1, target_q_values) + F.mse_loss(q_values2, target_q_values)
        self.optim1.zero_grad()
        self.optim2.zero_grad()
        loss.backward()
        self.optim1.step()
        self.optim2.step()


class TD3Agent:
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_layer_units=[32, 8],
                 memory_size=10000,
                 gamma=0.99,
                 nb_steps_warm_up=2000,
                 sigma=0.3,
                 polyak=0.995,
                 pi_lr=3e-4,
                 q_lr=3e-4,
                 batch_size=32,
                 action_noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 training=True):
        self.gamma = gamma
        self.sigma = sigma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.action_space = action_space
        self.nb_actions = action_space.shape[0]
        self.observation_shape = observation_space.shape
        self.action_shape = action_space.shape
        self.nb_steps_warm_up = nb_steps_warm_up
        self.training = training
        
        self.memory = Memory(
            capacity=memory_size,
            observation_shape=self.observation_shape,
            action_shape=self.action_shape
        )
        
        self.actor_model = ActorNet(self.observation_shape, self.action_shape, hidden_layer_units, pi_lr)
        self.critic_model = CriticNet(self.observation_shape, self.action_shape, hidden_layer_units, q_lr)
        
        self.target_actor_model = copy.deepcopy(self.actor_model)
        self.target_critic_model = copy.deepcopy(self.critic_model)
        
        self.step_count = 0
    
    def forward(self, observation):
        self.step_count += 1
        
        if self.step_count < self.nb_steps_warm_up and self.training:
            return self.action_space.sample()
        else:
            observation = ptu.from_numpy(observation.flatten())
            action = self.actor_model(observation).reshape(self.nb_actions)
            if self.training:
                noise = ptu.normal(0.0, self.action_noise, (self.nb_actions,)).clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise).clamp(-1, 1)
            return ptu.get_numpy(action)
    
    def backward(self, observation, action, reward, terminal, next_observation):
        self.memory.store_transition(observation, action, reward, terminal, next_observation)
        
        if self.step_count < self.nb_steps_warm_up:
            return
        else:
            self._update()
    
    def _update(self):
        observations, actions, rewards, terminals, next_observations = self.memory.sample_batch(self.batch_size)
        
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        rewards = ptu.from_numpy(rewards)
        terminals = ptu.from_numpy(terminals)
        next_observations = ptu.from_numpy(next_observations)
        
        self._update_critic(observations, actions, rewards, terminals, next_observations)
        self._update_actor(observations)
        
        if self.step_count % self.policy_delay == 0:
            # 更新critic的target网络
            ptu.soft_update_from_to(self.critic_model,
                                    self.target_critic_model,
                                    self.polyak)
            
            # 更新actor的target网络
            ptu.soft_update_from_to(self.actor_model,
                                    self.target_actor_model,
                                    self.polyak)
    
    def _update_critic(self, observations, actions, rewards, terminals, next_observations):
        batch_size = observations.shape[0]
        
        with torch.no_grad():
            action_noise = ptu.normal(0.0, self.target_noise, (batch_size, self.nb_actions))
            action_noise = action_noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (self.target_actor_model(next_observations) + action_noise).clamp(-1, 1)
            
            q_values_next1, q_values_next2 = self.target_critic_model(next_observations, next_actions)
            
            target_q_values = rewards + self.gamma * (1 - terminals) * torch.min(q_values_next1, q_values_next2)
        
        self.critic_model.backward(observations, actions, target_q_values)
    
    def _update_actor(self, observations):
        self.actor_model.backward(self.critic_model, observations)
    
    def switch_mode(self, training=None):
        """
        :param training:  agent所处的模式，
            training=True： 训练模式
            training=False: 测试模式
        """
        if training is None:
            self.training = ~self.training
        else:
            self.training = training
        
        if self.training:
            print("Switch to train mode.")
        else:
            print("Switch to test mode.")
    
    def save_weights(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        ptu.save_model(self.actor_model, os.path.join(path, "actor.pkl"))
        ptu.save_model(self.target_actor_model, os.path.join(path, "target_actor.pkl"))
        ptu.save_model(self.critic_model, os.path.join(path, "critic.pkl"))
        ptu.save_model(self.target_critic_model, os.path.join(path, "target_critic.pkl"))
        
        print("Save weights successfully: ", path)
    
    def load_weights(self, path):
        ptu.load_model(self.actor_model, os.path.join(path, "actor.pkl"))
        ptu.load_model(self.target_actor_model, os.path.join(path, "target_actor.pkl"))
        ptu.load_model(self.critic_model, os.path.join(path, "critic.pkl"))
        ptu.load_model(self.target_critic_model, os.path.join(path, "target_critic.pkl"))
        
        print("Load weights successfully: ", path)
