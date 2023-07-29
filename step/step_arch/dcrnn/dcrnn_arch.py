import torch
from torch import nn
import numpy as np

from .dcrnn_cell import DCGRUCell


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(
            model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, sampled_adj=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], sampled_adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, sampled_adj=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], sampled_adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Codes are modified from the official repo:
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py,
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    """

    def __init__(self, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False))
        self.fc_his = nn.Sequential(nn.Linear(96, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, sampled_adj):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state, sampled_adj)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None, sampled_adj=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state, sampled_adj)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def _calculate_random_walk_matrix(self, adj_mx):
        B, N, N = adj_mx.shape

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, hidden_states: torch.Tensor = None, sampled_adj = None, **kwargs) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]
            future_data (torch.Tensor, optional): future data with shape [L, B, N*C_out]
            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction wit shape [L, B, N*C_out]
        """

        if not self.training:
            future_data = None # Future data should not be visible when validating/testing.

        # reshape data
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        history_data = history_data.transpose(0, 1)         # [L, B, N*C]

        sampled_adj = [self._calculate_random_walk_matrix(sampled_adj), self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]

        if future_data is not None:
            future_data = future_data[..., [0]]     # teacher forcing only use the first dimension.
            batch_size, length, num_nodes, channels = future_data.shape
            future_data = future_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            future_data = future_data.transpose(0, 1)         # [L, B, N*C]

        # DCRNN
        encoder_hidden_state = self.encoder(history_data, sampled_adj)
        encoder_hidden_state = encoder_hidden_state.view(self.num_rnn_layers, batch_size, num_nodes, -1)
        hidden_states = self.fc_his(hidden_states).unsqueeze(0)        # B, N, D
        encoder_hidden_state += hidden_states
        encoder_hidden_state = encoder_hidden_state.view(self.num_rnn_layers, batch_size, -1)
        outputs = self.decoder(encoder_hidden_state, future_data,
                               batches_seen=batch_seen, sampled_adj=sampled_adj)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1).transpose(1, 2)

        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print(count_parameters(self))
        return outputs
