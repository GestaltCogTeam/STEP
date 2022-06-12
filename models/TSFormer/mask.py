import random
import torch.nn as nn
import numpy as np

class MaskGenerator(nn.Module):
    def __init__(self, mask_size, mask_ratio, distribution='uniform', lm=-1):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.sort = True
        self.average_patch = lm
        self.distribution = distribution
        if self.distribution == "geom":
            assert lm != -1
        assert distribution in ['geom', 'uniform']

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

    def geometric_rand(self):
        mask = geom_noise_mask_single(self.mask_size, lm=self.average_patch, masking_ratio=self.mask_ratio)     # 1: masked, 0:unmasked
        self.masked_tokens = np.where(mask)[0].tolist()
        self.unmasked_tokens = np.where(~mask)[0].tolist()
        # assert len(self.masked_tokens) > len(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        if self.distribution == 'geom':
            self.unmasked_tokens, self.masked_tokens = self.geometric_rand()
        elif self.distribution == 'uniform':
            self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        else:
            raise Exception("ERROR")
        return self.unmasked_tokens, self.masked_tokens

# https://github.com/gzerveas/mvts_transformer/blob/fe3b539ccc2162f55cf7196c8edc7b46b41e7267/src/datasets/dataset.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L274
def geom_noise_mask_single(L, lm, masking_ratio):
    """
    *** NOT USED ***
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    L = int(L)
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m/(p_m+p_u), p_u/(p_m+p_u)]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state
    mask = ~keep_mask
    return mask
