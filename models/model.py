import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.MAE_TS.model import MAE_TS
from models.GWNet.model import gwnet
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

class STGNN(nn.Module):
    def __init__(self, cfg, **model_args):
        super().__init__()
        tsformer_args = model_args['TSFORMER']
        backend_args  = model_args['BACKEND']
        self.K = cfg['K']
        self.dataset_name = cfg.DATASET_NAME
        self.Backend = gwnet(**backend_args.GWNET)

        self.TSFormer = MAE_TS(mode='inference', **tsformer_args)
        self.load_pretrained_model()
        if self.dataset_name == 'METR-LA':
            self.dim_fc = int(383552)
        elif self.dataset_name == 'PEMS04':
            self.dim_fc = int(217296)
        elif self.dataset_name == 'PEMS-BAY':
            self.dim_fc = int(583424)
        else:
            assert False, "Error"
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.fc = torch.nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear((self.embedding_dim) * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        if self.dataset_name == 'METR-LA':
            self.fc_mean = nn.Linear(16128, 100)            #  288 * 7 / 12 * 96
        elif self.dataset_name == 'PEMS04':
            self.fc_mean = nn.Linear(16128 * 2, 100)        # 288 * 7 * 2 / 12 * 96
        elif self.dataset_name == 'PEMS-BAY':
            self.fc_mean = nn.Linear(16128, 100)            # 288 * 7 * 2 / 12 * 96
        else:
            assert False, "Error"
        self.dropout = nn.Dropout(0.5)
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot
        # Generate off-diagonal interaction graph
        if self.dataset_name == 'METR-LA':
            self.num_nodes = 207
            off_diag = np.ones([207, 207])
        elif self.dataset_name == 'PEMS04':
            self.num_nodes = 307
            off_diag = np.ones([307, 307])
        elif self.dataset_name == 'PEMS-BAY':
            self.num_nodes = 325
            off_diag = np.ones([325, 325])
        else:
            assert False, "Error"
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)
        
        if self.dataset_name == 'METR-LA':
            sensor_ids, sensor_id_to_ind, adj_mx = load_pkl("datasets/sensor_graph/adj_mx_la.pkl")
        elif self.dataset_name == 'PEMS04':
            adj_mx = load_pkl("datasets/sensor_graph/adj_mx_04.pkl")
        elif self.dataset_name == 'PEMS-BAY':
            sensor_ids, sensor_id_to_ind, adj_mx = load_pkl("datasets/sensor_graph/adj_mx_bay.pkl")
        else:
            assert False, "Error"
        self.adj_ori = torch.tensor(adj_mx)

        if self.dataset_name == 'METR-LA':
            self.node_feas = torch.from_numpy(load_pkl("datasets/METR-LA/data.pkl")).float()[:23990, :, 0]
        elif self.dataset_name == 'PEMS04':
            self.node_feas = torch.from_numpy(load_pkl("datasets/PEMS04/data.pkl")).float()[:13599, :, 0]
        elif self.dataset_name == 'PEMS-BAY':
            self.node_feas = torch.from_numpy(load_pkl("datasets/PEMS-BAY/data.pkl")).float()[:36482, :, 0]
        else:
            assert False, "Error"
        train_feas = self.node_feas.cpu().numpy()
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(train_feas.T, 10, metric='cosine')
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g)
        
    def load_pretrained_model(self):
        if self.dataset_name == 'METR-LA':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_metr.pt")
        elif self.dataset_name == 'PEMS04':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_pems04.pt")
        elif self.dataset_name == 'PEMS-BAY':
            checkpoint_dict = torch.load("TSFormer_CKPT/TSFormer_pemsbay.pt")
        else:
            assert False, "Error"
        self.TSFormer.load_state_dict(checkpoint_dict['model_state_dict'])
        for param in self.TSFormer.parameters():
            param.requires_grad = False

    def sparse_ratio(self, adj):
        B, N, N = adj.shape
        total_element = N * N * B
        one_element = adj.sum(-1).sum(-1).sum(-1)
        ratio = one_element / total_element
        print("Sparse Ratio: {:.2f}".format(ratio*100))

    def hit_ratio(self, adj):
        adj_test = adj[0]
        # adj_test = adj_test - torch.eye(adj_test.shape[0]).to(adj_test.device)
        adj_ori = torch.where(self.adj_ori>0, 1., 0.).to(adj.device)
        # hit
        ## total number
        total_hit = adj_ori.sum(-1).sum(-1) 
        ## hit pos
        zeros = torch.zeros_like(adj_test)
        adj_hit = torch.where(adj_ori>0, adj_test, zeros)
        ## hit number
        hit_num = adj_hit.sum(-1).sum(-1)
        ## hit ratio
        hit_ratio = (hit_num / total_hit).cpu().item()
        print("kNN Hit Ratio: {:.2f}".format(hit_ratio*100))

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

    def forward(self, S_X, L_X=None, label=None, His=None, batch_seen=None, epoch=0):
        """
        Feedforward

        Args:
            S_X: [B, P, N, C]
            L_X: [B, L, N, C]
            His: [B, N, D]
        
        Returns:
            Y_hat: [B, N, P]
        """
        B, L, N, C = S_X.shape
        x = self.node_feas.to(S_X.device).transpose(1, 0).view(N, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(N, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)
        H = self.TSFormer(L_X[..., [0]].permute(0, 2, 3, 1))        # B, N, L/P, D
        his_ave = H.reshape(B, N, -1)
        his_ave = F.relu(self.fc_mean(his_ave))
        x = x.unsqueeze(0).expand(B, N, -1)
        x = x + his_ave

        receivers = torch.matmul(self.rel_rec.to(x.device), x)
        senders = torch.matmul(self.rel_send.to(x.device), x)
        x = torch.cat([senders, receivers], dim=-1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = gumbel_softmax(x, temperature=0.5, hard=True)
        adj = adj[..., 0].clone().reshape(B, N, -1)
        mask = torch.eye(N, N).unsqueeze(0).bool().to(adj.device)
        adj.masked_fill_(mask, 0)

        his = H.reshape(B, N, -1)        # B*N, D, L_P
        adj_knn = self.get_k_NN_neighbor(his, k=self.K*self.num_nodes, metric='cosine')
        mask = torch.eye(N, N).unsqueeze(0).bool().to(adj.device)
        adj_knn.masked_fill_(mask, 0)

        if batch_seen is not None and batch_seen % 50 == 0:
            print(epoch)
            self.hit_ratio(adj)
            self.sparse_ratio(adj)

        H = H[:, :, -1, :]

        Y_hat = self.Backend(S_X, His=H, adj=adj)
        return Y_hat, x.softmax(-1)[..., 0].clone().reshape(B, N, N), adj_knn
