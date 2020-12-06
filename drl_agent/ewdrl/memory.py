import random
from collections import deque

import numpy as np
import torch


class Memory:
    def __init__(self, capacity, **kwargs):
        self.capacity = capacity
        
        # 创建容器
        self.observations = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        
        # 计数
        self.step = 0
        self.distilling_buffer = deque(maxlen=2048)
    
    def __len__(self):
        return min(self.step, self.capacity)
    
    def store_transition(self, observation, action, reward, next_observation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        
        self.step += 1
    
    def _sample_indices(self, batch_size):
        memory_size = self.__len__()
        batch_size = min(memory_size, batch_size)
        return np.random.choice(np.arange(memory_size), batch_size, replace=False)
    
    def sample_batch(self, batch_size=32):
        indices = self._sample_indices(batch_size)
        
        observations = torch.cat([self.observations[i] for i in indices], dim=0)
        actions = torch.cat([self.actions[i] for i in indices], dim=0)
        rewards = torch.cat([self.rewards[i] for i in indices], dim=0)
        next_observations = torch.cat([self.next_observations[i] for i in indices], dim=0)
        
        return observations, actions, rewards, next_observations
    
    def store_kd_sample(self, sample):
        self.distilling_buffer.append(sample)
    
    def sample_kd_samples(self, batch_size=32):
        n_samples = len(self.distilling_buffer)
        if n_samples == 0:
            return None
        
        samples = random.sample(self.distilling_buffer, min(n_samples, batch_size))
        # self.distilling_buffer.clear()
        
        return torch.cat(samples, dim=0)


class SequentialMemory(Memory):
    def _sample_indices(self, batch_size):
        memory_size = len(self)
        batch_size = min(batch_size, memory_size)
        indices = range(memory_size - batch_size, memory_size)
        return indices


def get_memory_instance(memory_class_name):
    return eval(memory_class_name)
