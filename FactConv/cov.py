import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math

class Covariance(nn.Module):
    """ Module for learnable weight covariance matrices 
        Input: the size of the covariance matrix
        For channel covariances, triu_size is in_channels // groups
        For spatial covariances, triu_size is kernel_size[0] * kernel_size[1]
    """
    def __init__(self, triu_size):
        super().__init__()
        self.triu_size = triu_size

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        scat_idx = triu[0] * self.triu_size + triu[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = triu.shape[1]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)


    def cov(self):
        R = self._tri_vec_to_mat(self.tri_vec, self.triu_size, self.scat_idx)
        return R.T @ R

    def sqrt(self):
        R = self._tri_vec_to_mat(self.tri_vec, self.triu_size, self.scat_idx)
        return R

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        r = torch.zeros((n*n), device=self.tri_vec.device,
                dtype=self.tri_vec.dtype).scatter_(0, scat_idx, vec).view(n, n)
        r = torch.diagonal_scatter(r, r.diagonal().exp_())
        return r

class LowRankCovariance(Covariance):
    """ Module for learnable low-rank covariance matrices
        Input: the size of the covariance matrix and the rank as a percentage
    """
    def __init__(self, triu_size, rank):
        super().__init__(triu_size)
        self.k = math.ceil(rank * triu_size)
        print(self.k)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] < self.k
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

class LowRank1stFullCovariance(Covariance):
    """ Module for learnable low-rank covariance matrices
        Input: the size of the covariance matrix and the rank as a percentage
    """
    def __init__(self, triu_size, rank):
        super().__init__(triu_size)
        if triu_size == 3:
            self.k = 3
            print(self.k)
        else:
            self.k = math.ceil(rank * triu_size)
        print(self.k)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] < self.k
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)


class LowRankPlusDiagCovariance(Covariance):
    """ Module for learning low-rank covariances plus the main diagonal
    input: size of covariance matrix and the percentage rank (rows kept in upper triangular covaraince)"""
    def __init__(self, triu_size, rank):
        super().__init__(triu_size)
        if triu_size == 3:
            self.k = 3
            print(self.k)
        else:
            self.k = math.ceil(rank * triu_size)
            print(self.k)
        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = (triu[0] < self.k)|(triu[0]==triu[1])
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

class LowRankK1Covariance(Covariance):
    """ Module for learnable low-rank covariance matrices where
        each layer has a rank of 1
        Input: the size of the covariance matrix 
    """
    def __init__(self, triu_size, rank):
        super().__init__(triu_size)

        self.k = 1
        print(self.k)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] < self.k
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)


class OffDiagCovariance(Covariance):
    """ Module for learning off-diagonal of weight covariance matrix
    input: size of covariance matrix"""
    def __init__(self, triu_size):
        super().__init__(triu_size)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = (triu[0]+1) == triu[1]
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

class DiagCovariance(Covariance):
    """ Module for learning diagonal of weight covariance matrix
    input: size of covariance matrix"""
    def __init__(self, triu_size):
        super().__init__(triu_size)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] == triu[1]
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)
