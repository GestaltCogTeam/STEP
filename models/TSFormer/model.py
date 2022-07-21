import torch
import torch.nn as nn

from models.TSFormer.Transformer_layers import TransformerLayers
from models.TSFormer.mask import MaskGenerator
from models.TSFormer.patch import Patch
from models.TSFormer.positional_encoding import PositionalEncoding
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

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
    def __init__(self, patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, L=6, mode='Pretrain', spectral=True):
        super().__init__()
        self.patch_size = patch_size
        self.seleted_feature = 0
        self.mode = mode
        self.spectral = spectral
        self.patch = Patch(patch_size, in_channel, out_channel, spectral=spectral)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask  = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, L)
        self.decoder = TransformerLayers(out_channel, 1)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=.02)
        if self.spectral:
            self.output_layer = nn.Linear(out_channel, int(patch_size/2+1)*2)
        else:
            self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pretrain(self, input):
        """feed forward of the TSFormer in the pre-training stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
            torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
            dict: data for plotting.
        """
        B, N, C, L = input.shape
        # get patches and exec input embedding
        patches = self.patch(input)             # B, N, d, L/P
        patches = patches.transpose(-1, -2)     # B, N, L/P, d
        # positional embedding
        patches = self.pe(patches)
        
        # mask tokens
        unmasked_token_index, masked_token_index = self.mask()

        encoder_input = patches[:, :, unmasked_token_index, :]        

        # encoder
        H = self.encoder(encoder_input)         # B, N, L/P*(1-r), d
        # encoder to decoder
        H = self.encoder_2_decoder(H)
        # decoder
        # H_unmasked = self.pe(H, index=unmasked_token_index)
        H_unmasked = H
        H_masked   = self.pe(self.mask_token.expand(B, N, len(masked_token_index), H.shape[-1]), index=masked_token_index)
        H_full = torch.cat([H_unmasked, H_masked], dim=-2)   # # B, N, L/P, d
        H      = self.decoder(H_full)

        # output layer
        if self.spectral:
            # output = H
            spec_feat_H_ = self.output_layer(H)
            real = spec_feat_H_[..., :int(self.patch_size/2+1)]
            imag = spec_feat_H_[..., int(self.patch_size/2+1):]
            spec_feat_H = torch.complex(real, imag)
            out_full = torch.fft.irfft(spec_feat_H)
        else:
            out_full = self.output_layer(H)

        # prepare loss
        B, N, _, _ = out_full.shape 
        out_masked_tokens = out_full[:, :, len(unmasked_token_index):, :]
        out_masked_tokens = out_masked_tokens.view(B, N, -1).transpose(1, 2)

        label_full  = input.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.seleted_feature, :].transpose(1, 2)  # B, N, L/P, P
        label_masked_tokens  = label_full[:, :, masked_token_index, :].contiguous()
        label_masked_tokens  = label_masked_tokens.view(B, N, -1).transpose(1, 2)

        # prepare plot
        ## note that the output_full and label_full are not aligned. The out_full is shuffled.
        unshuffled_index = unshuffle(unmasked_token_index + masked_token_index)     # therefore, we need to unshuffle the out_full for better plotting.
        out_full_unshuffled = out_full[:, :, unshuffled_index, :]
        plot_args = {}
        plot_args['out_full_unshuffled']    = out_full_unshuffled
        plot_args['label_full']             = label_full
        plot_args['unmasked_token_index']   = unmasked_token_index
        plot_args['masked_token_index']     = masked_token_index

        return out_masked_tokens, label_masked_tokens, plot_args

    def _forward_backend(self, input):
        """the feed forward process in the forecasting stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        B, N, C, LP = input.shape
        # get patches and exec input embedding
        patches = self.patch(input)             # B, N, d, L
        patches = patches.transpose(-1, -2)     # B, N, L, d
        # positional embedding
        patches = self.pe(patches)
        
        encoder_input = patches          # no mask when running the backend.

        # encoder
        H = self.encoder(encoder_input)         # B, N, L, d
        return H

    def forward(self, input_data):
        """feed forward of the TSFormer.
        TSFormer has two modes: the pre-training mode and the forecasting model, which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            input_data (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.
        
        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
                torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
                dict: data for plotting.
            forecasting: 
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        if self.mode == 'Pretrain':
            return self._forward_pretrain(input_data)
        else:
            return self._forward_backend(input_data)
