import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math

class Covariance(nn.Module):
    """ Module for learnable weight covariance matrices 
        Input: the size of the covariance matrix and the nonlinearity
        For channel covariances, triu_size is in_channels // groups
        For spatial covariances, triu_size is kernel_size[0] * kernel_size[1]
    """
    def __init__(self, triu_size, nonlinearity):
        super().__init__()
        self.triu_size = triu_size

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        scat_idx = triu[0] * self.triu_size + triu[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = triu.shape[1]
        tri_vec = torch.zeros((triu_len,))

        self.nl = nonlinearity
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]
        diag = triu[0] == triu[1]
        tri_vec[diag] = self.dict2[self.nl]
        self.tri_vec = Parameter(tri_vec)


    def cov(self):
        R = self._tri_vec_to_mat(self.tri_vec, self.triu_size, self.scat_idx)
        return R.T @ R

    def sqrt(self):
        R = self._tri_vec_to_mat(self.tri_vec, self.triu_size, self.scat_idx)
        return R

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        r = torch.zeros((n*n), device=self.tri_vec.device,
                dtype=self.tri_vec.dtype)
        r = r.view(n, n).fill_diagonal_(self.dict2[self.nl]).view(n*n)
        r = r.scatter_(0, scat_idx, vec).view(n, n)
        r = torch.diagonal_scatter(r, self.func(r.diagonal()))
        
        return r

class LowRankCovariance(Covariance):
    """ Module for learnable low-rank covariance matrices
        Input: the size of the covariance matrix and the rank as a percentage
        (Relative rank)
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
        (Relative rank)
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
    input: size of covariance matrix and the percentage rank (rows kept in upper triangular covaraince)
    (Relative rank)
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
        mask = (triu[0] < self.k)|(triu[0]==triu[1])
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

class LowRankK1Covariance(Covariance):
    """ Module for learnable low-rank covariance matrices where
        each layer has a rank integer
        Input: the size of the covariance matrix and the rank (int)
        (Absolute rank)
    """
    def __init__(self, triu_size, rank, nonlinearity):
        super().__init__(triu_size, nonlinearity)

        if triu_size == 3:
            self.k = 3
            print(self.k)
        elif triu_size < rank:
            self.k = triu_size
            print(self.k)
        else:
            self.k = rank
            print(self.k)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] < self.k
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))

        self.nl = nonlinearity
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]
        diag = triu[0][mask] == triu[1][mask]
        tri_vec[diag] = self.dict2[self.nl]

        self.tri_vec = Parameter(tri_vec)

class LowRankK1DiagCovariance(Covariance):
    """ Module for learnable low-rank covariance matrices where
        each layer has a rank integer plus full diagonal
        Input: the size of the covariance matrix and the rank (int)
        (Absolute rank)
    """
    def __init__(self, triu_size, rank, nonlinearity):
        super().__init__(triu_size, nonlinearity)

        if triu_size == 3:
            self.k = 3
            print(self.k)
        elif triu_size < rank:
            self.k = triu_size
            print(self.k)
        else:
            self.k = rank
            print(self.k)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = (triu[0] < self.k)|(triu[0]==triu[1])
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))

        self.nl = nonlinearity
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]

        diag = triu[0][mask] == triu[1][mask]
        tri_vec[diag] = self.dict2[self.nl]
        self.tri_vec = Parameter(tri_vec)

class OffDiagCovariance(Covariance):
    """ Module for learning off-diagonal of weight covariance matrix
    input: size of covariance matrix"""
    def __init__(self, triu_size, nonlinearity):
        super().__init__(triu_size, nonlinearity)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = (triu[0]+1) == triu[1]
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.nl = nonlinearity
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]
        tri_vec.fill_(self.dict2[self.nl])
        self.tri_vec = Parameter(tri_vec)

class DiagCovariance(Covariance):
    """ Module for learning diagonal of weight covariance matrix
    input: size of covariance matrix"""
    def __init__(self, triu_size, nonlinearity):
        super().__init__(triu_size, nonlinearity)

        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        mask = triu[0] == triu[1]
        scat_idx = triu[0][mask] * self.triu_size + triu[1][mask]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = scat_idx.shape[0]
        tri_vec = torch.zeros((triu_len,))
        self.nl = nonlinearity
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]
        tri_vec.fill_(self.dict2[self.nl])
        self.tri_vec = Parameter(tri_vec)


class DiagonallyDominantCovariance(Covariance):
    """ Module for learnable diagonally dominant weight covariance matrices 
        Input: the size of the covariance matrix
    """
    def __init__(self, triu_size):
        super().__init__(triu_size, "abs")
        self.triu_size = triu_size
        triu = torch.triu_indices(self.triu_size, self.triu_size,
                dtype=torch.long)
        scat_idx = triu[0] * self.triu_size + triu[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

        triu_len = triu.shape[1]
        tri_vec = torch.zeros((triu_len,))

        diag = triu[0] == triu[1]
        tri_vec[diag] = 1
        self.tri_vec = Parameter(tri_vec)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        r = torch.zeros((n*n), device=self.tri_vec.device,
                dtype=self.tri_vec.dtype)
        r = r.view(n, n).fill_diagonal_(1).view(n*n)
        r = r.scatter_(0, scat_idx, vec).view(n, n)
        r = torch.diagonal_scatter(r, torch.abs(torch.sum(r, dim=1)))
#        print(r) 
        return r

