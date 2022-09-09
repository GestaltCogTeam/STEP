import torch
from torch import nn

from .tsformer import TSFormer
from .graphwavenet import GraphWaveNet
from .discrete_graph_learning import DiscreteGraphLearning


class STEP(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, cfg, **model_args):
        super().__init__()
        self.dataset_name = cfg.DATASET_NAME

        # initialize pre-training model (TSFormer) and backed model (Graph WaveNet)
        tsformer_args = model_args["TSFORMER"]          # TSFormer arguments
        # backend model (Graph WaveNet) arguments
        backend_args = model_args["BACKEND"]["GWNET"]
        self.tsformer = TSFormer(**tsformer_args, mode="inference")
        self.backend = GraphWaveNet(**backend_args)

        # load pre-trained model
        self.load_pre_trained_model()

        # discrete graph learning
        self.dynamic_graph_learning = DiscreteGraphLearning(cfg)

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        if self.dataset_name == "METR-LA":
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_METR-LA.pt")
        elif self.dataset_name == "PEMS04":
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_PEMS04.pt")
        elif self.dataset_name == "PEMS-BAY":
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_PEMS-BAY.pt")
        else:
            assert False, "Error"
        # load parameters
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.tsformer.parameters():
            param.requires_grad = False

    def forward(self, short_term_history, long_term_history=None):
        """Feed forward of STEP. Details can be found in Section 3.2 of the paper.

        Args:
            short_term_history (torch.Tensor):
                                                Short-term historical MTS with shape [B, L, N, C],
                                                which is the commonly used input of existing STGNNs.
                                                L is the length of segments (patches).
            long_term_history (torch.Tensor, optional):
                                                Very long-term historical MTS with shape [B, P * L, N, C],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
            torch.Tensor: the Bernoulli distribution parameters with shape [B, N, N].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
        """

        batch_size, _, num_nodes, _ = short_term_history.shape

        # discrete graph learning & feed forward of TSFormer
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.dynamic_graph_learning(long_term_history, self.tsformer)

        # enhancing downstream STGNNs
        hidden_states = hidden_states[:, :, -1, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj)

        return y_hat, bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes), adj_knn
