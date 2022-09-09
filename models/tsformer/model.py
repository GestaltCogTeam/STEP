import torch
from torch import nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from models.tsformer.patch import Patch
from models.tsformer.mask import MaskGenerator
from models.tsformer.transformer_layers import TransformerLayers
from models.tsformer.positional_encoding import PositionalEncoding


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class TSFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, L, mode="Pretrain"):
        super().__init__()
        self.patch_size = patch_size
        self.selected_feature = 0
        self.mode = mode
        self.patch = Patch(patch_size, in_channel, out_channel)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, L)
        self.decoder = TransformerLayers(out_channel, 1)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=.02)
        self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pre_train(self, long_term_history):
        """Feed forward of the TSFormer in the pre-training stage.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N], where r is the masking ratio.
            torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N].
            dict: data for plotting.
        """

        batch_size, num_nodes, _, _ = long_term_history.shape
        # get patches and embed input
        patches = self.patch(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        # positional embedding
        patches = self.pe(patches)

        # mask tokens
        unmasked_token_index, masked_token_index = self.mask()
        encoder_input = patches[:, :, unmasked_token_index, :]

        # encoder
        hidden_states_unmasked = self.encoder(encoder_input)         # B, N, P*(1-r), d
        # encoder to decoder
        hidden_states_unmasked = self.encoder_2_decoder(hidden_states_unmasked)
        # decoder
        # H_unmasked = self.pe(H, index=unmasked_token_index)
        hidden_states_masked = self.pe(self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]), index=masked_token_index)
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        reconstruction_full = self.output_layer(self.decoder(hidden_states_full))       # B, N, P, L

        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

        label_full = long_term_history.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()  # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)   #      # B, r*P*d, N

        # prepare plot
        # note that the output_full and label_full are not aligned. The out_full is shuffled.
        # therefore, we need to unshuffle the out_full for better plotting.
        unshuffled_index = unshuffle(unmasked_token_index + masked_token_index)
        out_full_unshuffled = reconstruction_full[:, :, unshuffled_index, :]
        plot_args = {}
        plot_args["out_full_unshuffled"] = out_full_unshuffled
        plot_args["label_full"] = label_full
        plot_args["unmasked_token_index"] = unmasked_token_index
        plot_args["masked_token_index"] = masked_token_index

        return reconstruction_masked_tokens, label_masked_tokens, plot_args

    def _forward_backend(self, long_term_history):
        """Feed forward process in the forecasting stage.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, P * L, N, C],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """

        # get patches and exec input embedding
        patches = self.patch(long_term_history)             # B, N, d, L
        patches = patches.transpose(-1, -2)     # B, N, L, d
        # positional embedding
        patches = self.pe(patches)

        encoder_input = patches          # no mask when running the backend.

        # encoder
        hidden_states = self.encoder(encoder_input)         # B, N, L, d
        return hidden_states

    def forward(self, input_data):
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            input_data (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        if self.mode == "Pretrain":
            return self._forward_pre_train(input_data)
        else:
            return self._forward_backend(input_data)
