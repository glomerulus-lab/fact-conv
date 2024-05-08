import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union

""" 
The function below is copied directly from 
https://bonnerlab.github.io/ccn-tutorial/pages/analyzing_neural_networks.html
"""
def _contract(tensor, matrix, axis):
    """tensor is (..., D, ...), matrix is (P, D), returns (..., P, ...)."""
    t = torch.moveaxis(tensor, source=axis, destination=-1)  # (..., D)
    r = t @ matrix.T  # (..., P)
    return torch.moveaxis(r, source=-1, destination=axis)  # (..., P, ...)


class FactConv2d(nn.Conv2d):
    def __init__(
        self,
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
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                      device=self.weight.device,
                                      dtype=torch.long)
        scat_idx1 = triu1[0]*self.in_channels//self.groups + triu1[1]
        self.register_buffer("scat_idx1", scat_idx1, persistent=False)

        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                      device=self.weight.device,
                                      dtype=torch.long)
        scat_idx2 = triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
        self.register_buffer("scat_idx2", scat_idx2, persistent=False)

        triu1_len = triu1.shape[1]
        triu2_len = triu2.shape[1]

        tri1_vec = self.weight.new_zeros((triu1_len,))
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = self.weight.new_zeros((triu2_len,))
        self.tri2_vec = Parameter(tri2_vec)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                                  self.groups, self.scat_idx1)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

class DiagFactConv2d(nn.Conv2d):
    def __init__(
        self,
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
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                      device=self.weight.device,
                                      dtype=torch.long)
        mask = triu1[0] == triu1[1]
        scat_idx1 = triu1[0][mask]*self.in_channels//self.groups + triu1[1][mask]
        self.register_buffer("scat_idx1", scat_idx1, persistent=False)

        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                      device=self.weight.device,
                                      dtype=torch.long)
        mask = triu2[0] == triu2[1]
        scat_idx2 = triu2[0][mask]*self.kernel_size[0]*self.kernel_size[1] + triu2[1][mask]
        
        self.register_buffer("scat_idx2", scat_idx2, persistent=False)

        triu1_len = scat_idx1.shape[0]
        triu2_len = scat_idx2.shape[0]

        tri1_vec = self.weight.new_zeros((triu1_len,))
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = self.weight.new_zeros((triu2_len,))
        self.tri2_vec = Parameter(tri2_vec)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                                  self.groups, self.scat_idx1)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

class DiagChanFactConv2d(nn.Conv2d):
    def __init__(
        self,
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
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                      device=self.weight.device,
                                      dtype=torch.long)
        mask = triu1[0] == triu1[1]
        scat_idx1 = triu1[0][mask]*self.in_channels//self.groups + triu1[1][mask]
        self.register_buffer("scat_idx1", scat_idx1, persistent=False)

        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                      device=self.weight.device,
                                      dtype=torch.long)
        scat_idx2 = triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
        self.register_buffer("scat_idx2", scat_idx2, persistent=False)

        triu1_len = scat_idx1.shape[0]
        triu2_len = triu2.shape[1]

        tri1_vec = self.weight.new_zeros((triu1_len,))
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = self.weight.new_zeros((triu2_len,))
        self.tri2_vec = Parameter(tri2_vec)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                                  self.groups, self.scat_idx1)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

