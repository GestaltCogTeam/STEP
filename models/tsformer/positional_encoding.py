import torch
from torch import nn

class LearnableTemporalPositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, input_data, index):
        """Positional encoding.

        Args:
            input_data (torch.tensor): input sequence with shape [B*N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        if index is None:
            pe = self.pe[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.pe[index].unsqueeze(0)
        input_data  = input_data + pe
        input_data  = self.dropout(input_data)
        return input_data

class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        # temporal embedding
        input_data = self.tem_pe(input_data.view(batch_size*num_nodes, num_patches, num_feat), index=index)
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        # absolute positional embedding
        return input_data
