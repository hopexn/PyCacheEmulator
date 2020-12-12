import numpy as np

from drl_agent.ewdrl.model import Temperature
from .common import *


class KDWeights3(KDWeightsIF):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-3, **kwargs):
        super(KDWeights3, self).__init__(num_agents, n_neighbors, kdw_lr, **kwargs)
        
        kwd_log_tau = kwargs.get('kwd_log_tau', 0)
        min_entropy_ratio = kwargs.get('min_entropy_ratio', 0.99)
        self.tau = Temperature(
            log_tau=kwd_log_tau,
            min_entropy=min_entropy_ratio * np.log(num_agents),
            **kwargs)
    
    def forward(self, losses: torch.Tensor, **kwargs):
        self.ref_ws = (self.ws / self.tau.value().detach()).softmax(dim=0)
        
        if self.random_index:
            self.indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        else:
            self.indices = np.random.choice(np.arange(self.num_agents), self.n_neighbors,
                                            replace=False, p=ptu.get_numpy(self.ref_ws))
        
        self.indices = ptu.tensor(self.indices, dtype=torch.long)
        
        losses_selected = losses.index_select(dim=0, index=self.indices)
        ref_ws_selected = self.ref_ws.index_select(dim=0, index=self.indices)
        loss = (losses_selected * ref_ws_selected).sum()
        
        return loss
    
    def step(self):
        super().step()
        self.tau.backward(self.ref_ws)
