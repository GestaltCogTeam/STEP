import random
import torch.nn as nn
import numpy as np

class MaskGenerator(nn.Module):
    def __init__(self, mask_size, mask_ratio):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.mask_size)))
        random.shuffle(mask)
        mask_len = int(self.mask_size * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens
