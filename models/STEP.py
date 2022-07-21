import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.discrete_graph_learning import DiscreteGraphLearning

from models.TSFormer.model import TSFormer
from models.GWNet.model import gwnet

class STEP(nn.Module):
    def __init__(self, cfg, **model_args):
        super().__init__()
        tsformer_args = model_args['TSFORMER']
        backend_args  = model_args['BACKEND']
        self.dataset_name = cfg.DATASET_NAME
        self.Backend = gwnet(**backend_args.GWNET)

        # pretraining model for time series
        self.TSFormer = TSFormer(mode='inference', **tsformer_args)
        self.load_pretrained_model()
        
        # discrete graph learning
        self.dgl = DiscreteGraphLearning(cfg)
        
    def load_pretrained_model(self):
        if self.dataset_name == 'METR-LA':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_METR-LA.pt")
        elif self.dataset_name == 'PEMS04':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_PEMS04.pt")
        elif self.dataset_name == 'PEMS-BAY':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_PEMS-BAY.pt")
        else:
            assert False, "Error"
        self.TSFormer.load_state_dict(checkpoint_dict['model_state_dict'])
        for param in self.TSFormer.parameters():
            param.requires_grad = False

    def forward(self, S_X, L_X=None):
        """feedforward of STEP. Details can be found in Section 3.2 of the paper.

        Args:
            S_X (torch.Tensor): short-term historical MTS with shape [B, P, N, C], which is the commonly used input of existing STGNNs.
            L_X (torch.Tensor, optional): very long-term historical MTS with shape [B, L * P, N, C], which is used in the TSFormer.

        Returns:
            torch.Tensor: prediction with shape [B, N, P].
            torch.Tensor: the Bernulli distribution parameters with shape [B, N, N].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
        """
        B, L, N, C = S_X.shape
        
        # discrete graph learning
        x, H, adj_knn, adj = self.dgl(L_X, self.TSFormer)
        
        # enhancing downstream STGNNs
        H = H[:, :, -1, :]
        Y_hat = self.Backend(S_X, His=H, adj=adj)

        return Y_hat, x.softmax(-1)[..., 0].clone().reshape(B, N, N), adj_knn
