import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union

class LearnableLinearCov(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.factory_kwargs = factory_kwargs

        self.register_buffer("weight",
                             torch.empty((out_features, in_features),
                                         **factory_kwargs))

        triu_len = torch.triu_indices(in_features, in_features).shape[1]
        self.tri_vec = Parameter(torch.empty((triu_len,), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.tri_vec, 0.)
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        U = torch.zeros((self.in_features, self.in_features),
                        **self.factory_kwargs)
        U[torch.triu_indices(self.in_features, self.in_features).tolist()] \
            = self.tri_vec
        exp_diag = torch.exp(torch.diagonal(U))
        U[range(self.in_features), range(self.in_features)] = exp_diag
        composite_weight = self.weight @ U
        
        return F.linear(input, composite_weight, self.bias)

class LearnableCovConv2d(nn.Conv2d):
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
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=None
    ) -> None:
        # init as Conv2d
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)

        factory_kwargs = {'device': device, 'dtype': dtype}
        print("Device: ", device)
        self.factory_kwargs = factory_kwargs

        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        self.weight = Parameter(torch.randn(self.weight.shape, **factory_kwargs,
                                  requires_grad=False))
        nn.init.kaiming_normal_(self.weight)
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu_len = torch.triu_indices(self.in_features,
                                      self.in_features).shape[1]
        self.tri_vec = Parameter(torch.zeros((triu_len,), **factory_kwargs))
        
    def forward(self, input: Tensor) -> Tensor:
        U = torch.zeros((self.in_features, self.in_features),
                        **self.factory_kwargs)
        U[torch.triu_indices(self.in_features, self.in_features).tolist()] \
            = self.tri_vec
        exp_diag = torch.exp(torch.diagonal(U))
        U[range(self.in_features), range(self.in_features)] = exp_diag
        
        matrix_shape = (self.out_channels, self.in_features)
        composite_weight = torch.reshape(
            torch.reshape(self.weight, matrix_shape) @ U,
            self.weight.shape
        )
        
        return self._conv_forward(input, composite_weight, self.bias)