import numpy as np

from .common import *


class KDWeights1(KDWeightsIF):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-3, **kwargs):
        super(KDWeights1, self).__init__(num_agents, n_neighbors, kdw_lr, **kwargs)
    
    def forward(self, losses: torch.Tensor, **kwargs):
        self.indices = np.random.permutation(np.arange(self.num_agents))[:self.n_neighbors]
        self.indices = ptu.tensor(self.indices, dtype=torch.long)
        losses_selected = losses.index_select(dim=0, index=self.indices)
        return losses_selected.sum() / self.n_neighbors