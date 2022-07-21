# Discrete Graph Learning
from regex import E
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.log import load_pkl
from utils.Similarity_e import batch_cosine_similarity, batch_dot_similarity

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class DiscreteGraphLearning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # attributes
        self.K = cfg['K']
        self.num_nodes = {"METR-LA": 207, "PEMS04": 307, "PEMS-BAY": 325}[cfg.DATASET_NAME]
        self.train_length = {"METR-LA": 23990, "PEMS04": 13599, "PEMS-BAY": 36482}[cfg.DATASET_NAME]
        self.node_feas = torch.from_numpy(load_pkl("datasets/" + cfg.DATASET_NAME + "/data.pkl")).float()[:self.train_length, :, 0]
        
        # CNN for global feature extraction
        ## network parameter
        self.dim_fc = {"METR-LA": 383552, "PEMS04": 217296, "PEMS-BAY": 217296}[cfg.DATASET_NAME]
        self.embedding_dim = 100
        ## network structure
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        
        # FC for transforming dimension of features from TSFormer
        self.dim_fc_mean = {"METR-LA": 16128, "PEMS04": 16128 * 2, "PEMS-BAY": 16128}[cfg.DATASET_NAME]
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc_mean = nn.Linear(self.dim_fc_mean, 100)

        # discrete graph learning
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.dropout = nn.Dropout(0.5)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
            return labels_onehot
        self.rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))

    def get_k_NN_neighbor(self, data, k=11*207, metric='cosine'):
        """
        data: tensor B, N, D
        metric: cosine or dot
        """
        if metric == 'cosine':
            batch_sim = batch_cosine_similarity(data, data)
        elif metric == 'dot':
            batch_sim = batch_dot_similarity(data, data)    # B, N, N
        else:
            assert False, 'unknown metric'
        B, N, N = batch_sim.shape
        adj = batch_sim.view(B, N*N)
        res = torch.zeros_like(adj)
        topk, indices = torch.topk(adj, k, dim=-1)
        res.scatter_(-1, indices, topk)
        adj = torch.where(res!=0, 1.0, 0.0).detach().clone()
        adj = adj.view(B, N, N)
        adj.requires_grad = False
        return adj

    def forward(self, L_X, TSFormer):
        """learning discrete graph structure based on TSFormer.

        Args:
            L_X (torch.Tensor): very long-term historical MTS with shape [B, L * P, N, C].
            TSFormer (nn.Module): pre-trained TSFormer.

        Returns:
            torch.Tensor: Bernoulli parameter (unnormalized) of each edge of the learned dependency graph. Shape: [B, N * N, 2].
            torch.Tensor: the output of TSFormer with shape [B, N, L, d].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
            torch.Tensor: the sampled graph with shape [B, N, N].
        """
        device = L_X.device
        B, _, N, _ = L_X.shape
        # generate global feature
        x = self.node_feas.to(device).transpose(1, 0).view(N, 1, -1)
        x = self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))
        x = x.view(N, -1)
        x = F.relu(self.fc(x))
        x = self.bn3(x)
        x = x.unsqueeze(0).expand(B, N, -1)                     # Gi in Eq. (2)

        # generate dynamic feature based on TSFormer
        H = TSFormer(L_X[..., [0]].permute(0, 2, 3, 1))
        his_ave = F.relu(self.fc_mean(H.reshape(B, N, -1)))     # relu(FC(Hi)) in Eq. (2)

        # time series feature
        x = x + his_ave                                         # Zi in Eq. (2)

        # learning discrete graph structure
        receivers = torch.matmul(self.rel_rec.to(x.device), x)
        senders = torch.matmul(self.rel_send.to(x.device), x)
        x = torch.cat([senders, receivers], dim=-1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)                                      # Bernoulli parameter (unnormalized) Theta_{ij} in Eq. (2)

        # sampling
        adj = gumbel_softmax(x, temperature=0.5, hard=True)     # differentiable sampling via Gumbel-Softmax in Eq. (4)
        adj = adj[..., 0].clone().reshape(B, N, -1)
        mask = torch.eye(N, N).unsqueeze(0).bool().to(adj.device)   # remove self-loop, which is re-added in the 
        adj.masked_fill_(mask, 0)

        # prior graph based on TSFormer
        adj_knn = self.get_k_NN_neighbor(H.reshape(B, N, -1), k=self.K*self.num_nodes, metric='cosine')
        mask = torch.eye(N, N).unsqueeze(0).bool().to(adj.device)
        adj_knn.masked_fill_(mask, 0)

        return x, H, adj_knn, adj
