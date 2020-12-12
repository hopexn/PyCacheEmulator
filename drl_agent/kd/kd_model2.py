import numpy as np

from .common import *


class KDWeights2(KDWeightsIF):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-3, **kwargs):
        super(KDWeights2, self).__init__(num_agents, n_neighbors, kdw_lr, **kwargs)
        self.alpha = kwargs.get("alpha", 0.1)
    
    def forward(self, losses: torch.tensor, **kwargs):
        self.ref_ws = self.ws.softmax(dim=0)
        
        if self.random_index:
            self.indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        else:
            self.indices = np.random.choice(np.arange(self.num_agents), self.n_neighbors,
                                            replace=False, p=ptu.get_numpy(self.ref_ws))
        
        self.indices = ptu.tensor(self.indices, dtype=torch.long)
        
        losses_selected = losses.index_select(dim=0, index=self.indices)
        ref_ws_selected = self.ref_ws.index_select(dim=0, index=self.indices)
        
        exp_minus_loss = (-losses_selected).softmax(dim=-1)
        loss_ws = (ref_ws_selected.log() * exp_minus_loss).sum()
        loss = losses_selected.mean() + self.alpha * loss_ws
        
        return loss
