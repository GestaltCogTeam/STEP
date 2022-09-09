import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils import clock, load_pkl, dump_pkl

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MTSDataset(Dataset):
    def __init__(self, raw_file_path, index_file_path, seq_len, throw=False, pretrain=False) -> None:
        super().__init__()
        self.pretrain = pretrain
        self.seq_len = seq_len
        # read full data
        self.data = torch.from_numpy(load_pkl(raw_file_path)).float()
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])
        # read index
        index = load_pkl(index_file_path)       # [idx-12, idx, idx+12]
        if throw:
            self.index = self.pre_process(index)
        else:
            self.index = index

    def pre_process(self, index):
        for i, idx in enumerate(index):
            # _1 = idx[0]
            _2 = idx[1]
            # _3 = idx[2]
            if _2 - self.seq_len < 0 :
                continue
            else:
                break
        return index[i:]

    def data_reshaper(self, data):
        """Reshape data to any models."""
        if not self.pretrain:
            pass
        else:
            data = data[..., [0]]
            data = data.permute(1, 2, 0)
        return data

    def __getitem__(self, index):
        idx = self.index[index]
        y   = self.data[idx[1]:idx[2], ...]
        short_x = self.data[idx[0]:idx[1], ...]
        if self.pretrain:
            long_x  = self.data[idx[1]-self.seq_len:idx[1], ...]
            long_x = self.data_reshaper(long_x)
            y = None
            short_x = None
            abs_idx = torch.Tensor(range(idx[1]-self.seq_len, idx[1], 12))
            return long_x, abs_idx
        else:
            if idx[1]-self.seq_len < 0:
                long_x = self.mask
            else:
                long_x = self.data[idx[1]-self.seq_len:idx[1], ...]
            return y, short_x, long_x
        
    def __len__(self):
        return len(self.index)
