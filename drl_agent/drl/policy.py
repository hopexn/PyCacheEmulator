import numpy as np

from .core import QPolicy
from cache_emu import numpy_utils as npu


class RandomQPolicy(QPolicy):
    def __init__(self, num_targets, **kwargs):
        self.num_targets = num_targets
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        
        target_indices = npu.random_index(q_values.shape[0], self.num_targets)
        
        action = np.zeros_like(q_values, dtype=np.bool)
        action[target_indices] = 1
        
        return action


class GreedyQPolicy(QPolicy):
    def __init__(self, num_targets, **kwargs):
        self.num_targets = num_targets
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        
        target_indices = npu.top_k_index(q_values, self.num_targets)
        
        action = np.zeros_like(q_values, dtype=np.bool)
        action[target_indices] = 1
        
        return action


class EpsGreedyQPolicy(GreedyQPolicy):
    def __init__(self, num_targets, eps=0.02, **kwargs):
        super().__init__(num_targets)
        self.eps = eps
        
        self.random_policy = RandomQPolicy(num_targets)
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        if np.random.uniform() < self.eps:
            return self.random_policy.select_action(q_values)
        else:
            return super().select_action(q_values)


class DecayEpsGreedyQPolicy(EpsGreedyQPolicy):
    def __init__(self, num_targets, eps=1.0, eps_decay=0.995, eps_min=0.05, **kwargs):
        super().__init__(num_targets, eps)
        self.eps_decay = eps_decay
        self.eps_min = eps_min
    
    def select_action(self, q_values):
        action = super().select_action(q_values)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return action


class ProbabilisticQPolicy(QPolicy):
    def __init__(self, num_targets, **kwargs):
        self.num_targets = num_targets
    
    def select_action(self, probs):
        assert probs.ndim == 1
        
        probs = np.clip(probs, 1e-6, 1)
        probs = probs / probs.sum()
        target_indices = npu.random_index(probs.shape[0], self.num_targets, p=probs)
        
        action = np.zeros_like(probs, dtype=np.bool)
        action[target_indices] = 1
        
        return action


class BoltzmannQPolicy(ProbabilisticQPolicy):
    def __init__(self, num_targets, tau=1., **kwargs):
        super().__init__(num_targets)
        self.tau = tau
    
    def select_action(self, q_values):
        assert q_values.ndim == 1
        
        probs = np.clip(npu.softmax(q_values / self.tau), 1e-6, 1)
        probs /= probs.sum()
        
        return super().select_action(probs)


def get_policy_instance(policy_class_name):
    return eval(policy_class_name)
