import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math

class Covariance(nn.Module):
    def __init__(self, triu_size):
        super().__init__()
        self.triu_size = triu_size

        # for channel cov, triu_size is in_channels // groups
        # for spatial cov, triu_size is kernel_size[0] * kernel_size[1]

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        scat_idx = triu[0] * self.triu_size + triu[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = triu.shape[1]
        #tri_vec = self.weight.new_zeros((triu_len,))
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)


    def cov(self, R):
        return R.T @ R

    def sqrt(self, R):
        return R

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        #r = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        r = torch.zeros((n*n), device=self.tri_vec.device,
                dtype=self.tri_vec.dtype).scatter_(0, scat_idx, vec).view(n, n)
        r = torch.diagonal_scatter(r, r.diagonal().exp_())
        return r

class LowRankCovariance(Covariance):
    def __init__(self, triu_size, rank):
        super().__init__(triu_size)
        self.k = math.ceil(rank * triu_size)
#        print("Rank: ", self.k)
        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] < self.k
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

