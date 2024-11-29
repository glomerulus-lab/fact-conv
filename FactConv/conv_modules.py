import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
from cov import Covariance, LowRankCovariance, LowRankPlusDiagCovariance,\
LowRankK1Covariance, OffDiagCovariance, DiagCovariance,\
LowRankK1DiagCovariance, DiagonallyDominantCovariance

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
        nonlinearity,
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
        
        # self.in_features = self.in_channels // self.groups * \
        #   self.kernel_size[0] * self.kernel_size[1]

        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = Covariance(channel_triu_size, nonlinearity)
        self.spatial = Covariance(spatial_triu_size, nonlinearity)

    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()

        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)


class LowRankFactConv2d(nn.Conv2d):
    def __init__(
        self,
        spatial_k,
        channel_k,
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
        dtype=None,
    ) -> None:
        # init as Conv2d
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        self.spatial_k = spatial_k
        self.channel_k = channel_k
        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = LowRankCovariance(channel_triu_size, self.channel_k)
        self.spatial = LowRankCovariance(spatial_triu_size, self.spatial_k)
        #self.spatial = Covariance(spatial_triu_size)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()

        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class LowRankPlusDiagFactConv2d(nn.Conv2d):
    def __init__(
        self,
        channel_k,
        nonlinearity,
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
        dtype=None,
    ) -> None:
        # init as Conv2d
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        self.channel_k = channel_k
        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = LowRankK1DiagCovariance(channel_triu_size, self.channel_k, nonlinearity)
        self.spatial = Covariance(spatial_triu_size, nonlinearity)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()
        
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class LowRankK1FactConv2d(nn.Conv2d):
    def __init__(
        self,
        channel_k,
        nonlinearity,
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
        dtype=None,
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
        
        self.channel_k = channel_k
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = LowRankK1Covariance(channel_triu_size, self.channel_k,\
                nonlinearity)
        self.spatial = Covariance(spatial_triu_size, nonlinearity)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()
        
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class OffDiagFactConv2d(nn.Conv2d):
    def __init__(
        self,
        nonlinearity,
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
        dtype=None,
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
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = OffDiagCovariance(channel_triu_size, nonlinearity)
        self.spatial = Covariance(spatial_triu_size, nonlinearity)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()
        
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class DiagFactConv2d(nn.Conv2d):
    def __init__(
        self,
        nonlinearity,
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
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = DiagCovariance(channel_triu_size, nonlinearity)
        self.spatial = DiagCovariance(spatial_triu_size, nonlinearity)


    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()

        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class DiagChanFactConv2d(nn.Conv2d):
    def __init__(
        self,
        nonlinearity,
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
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = DiagCovariance(channel_triu_size, nonlinearity)
        self.spatial = Covariance(spatial_triu_size, nonlinearity)

    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)

class DiagDomFactConv2d(nn.Conv2d):
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
        
        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = DiagonallyDominantCovariance(channel_triu_size)
        self.spatial = Covariance(spatial_triu_size, "abs")

    def forward(self, input: Tensor) -> Tensor:
        U1 = self.channel.sqrt()
        U2 = self.spatial.sqrt()
        # flatten over filter dims and contract
        composite_weight = _contract(self.weight, U1.T, 1)
        composite_weight = _contract(
            torch.flatten(composite_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        return self._conv_forward(input, composite_weight, self.bias)
