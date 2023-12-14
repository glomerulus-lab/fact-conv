import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import time
import copy

class FactLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=None
    ) -> None:
        # init as Linear
        super().__init__(
               in_features, out_features, bias, device, dtype) 

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = factory_kwargs
        
        # weight shape: (out_features, in_features)
        weight_shape = self.weight.shape
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", torch.empty(weight_shape, **factory_kwargs))
        nn.init.kaiming_normal_(self.weight)

        self.in_features = in_features
        self.out_features = out_features
        
        triu1 = torch.triu_indices(self.in_features, self.in_features,
                **factory_kwargs)
        triu_len = triu1.shape[1]
        tri_vec = torch.zeros((triu_len,), **factory_kwargs)
        self.tri_vec = Parameter(tri_vec)

        self.scat_idx = triu1[0] * self.in_features + triu1[1]

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

#        self.reset_parameters()

#    def reset_parameters(self) -> None:
#        nn.init.constant_(self.tri_vec, 0.)
#        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#        # https://github.com/pytorch/pytorch/issues/57109
#        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
#        if self.bias is not None:
#            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#            nn.init.uniform_(self.bias, -bound, bound)
#
    def forward(self, input: Tensor) -> Tensor:
        U = self._tri_vec_to_mat(self.tri_vec, self.in_features, self.scat_idx)
        U = self._exp_diag(U)

        composite_weight = self.weight @ U
        
        return F.linear(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = torch.zeros((n* n),
                **self.factory_kwargs).scatter_(0,scat_idx,vec).view(n,n)
        U = torch.diagonal_scatter(U,U.diagonal().exp_())
        return U
    
    def _exp_diag(self, mat):
        exp_diag = torch.exp(torch.diagonal(mat))
        n = mat.shape[0]
        mat[range(n), range(n)] = exp_diag
        return mat
