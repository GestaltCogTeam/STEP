import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    """Transformer layers."""

    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        batch_size, num_nodes, num_patches, num_feats = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(batch_size*num_nodes, num_patches, num_feats)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(batch_size, num_nodes, num_patches, num_feats)
        return output
