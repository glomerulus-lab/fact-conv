import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math
""" 
The function below is copied directly from 
https://bonnerlab.github.io/ccn-tutorial/pages/analyzing_neural_networks.html
"""
def _contract(tensor, matrix, axis):
    """tensor is (..., D, ...), matrix is (P, D), returns (..., P, ...)."""
    t = torch.moveaxis(tensor, source=axis, destination=-1)  # (..., D)
    r = t @ matrix.T  # (..., P)
    return torch.moveaxis(r, source=-1, destination=axis)  # (..., P, ...)


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


class LowRankK1DiagCovariance(Covariance):
    """ Module for learnable low-rank covariance matrices where
        each layer has a rank integer plus full diagonal
        Input: the size of the covariance matrix and the rank (int)
        (Absolute rank)
    """
    def __init__(self, triu_size, rank):
        super().__init__(triu_size, "abs")

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

        self.nl = "abs"
        self.dict1 = {'abs': torch.abs, 'exp': torch.exp, 'x^2': torch.square,
                '1/x^2': lambda x: torch.div(1, torch.square(x)), 'relu':
                torch.relu, 'softplus': nn.Softplus(), 'identity': nn.Identity()}
        self.dict2 = {'abs': 1, 'exp': 0, 'x^2': 1, '1/x^2': 1, 'relu': 1, 
                'softplus': 0.5414, 'identity': 1}
        self.func = self.dict1[self.nl]

        diag = triu[0][mask] == triu[1][mask]
        tri_vec[diag] = self.dict2[self.nl]
        self.tri_vec = Parameter(tri_vec)

class LowRankDiagResamplingDoubleFactConv2d(nn.Conv2d):
    def __init__(
        self,
        channel_k: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # init as Conv2d
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.weight)

        new_weight = torch.empty_like(self.weight)
        self.register_buffer("resampling_weight", new_weight)
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = LowRankK1DiagCovariance(channel_triu_size, channel_k)
        self.spatial = Covariance(spatial_triu_size, "abs")

        self.state=0


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()

        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        x2 = self._conv_forward(input[input.shape[0]//2:], composite_weight, self.bias)

        if self.state == 1:
            return x2

        #nn.init.orthogonal_(self.resampling_weight)
        composite_resampling_weight = _contract(self.resampling_weight, U1.T, 1)
        composite_resampling_weight = _contract(
            torch.flatten(composite_resampling_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        x1 = self._conv_forward(input[:input.shape[0]//2], composite_resampling_weight, self.bias)
        return torch.cat([x1, x2], dim=0)


    def resample(self):
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

    def ref_resample(self):
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.resampling_weight)



