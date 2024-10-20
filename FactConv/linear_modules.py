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
        device=None,
        dtype=None
    ) -> None:
        # init as Linear
        super().__init__(
               in_features, out_features, bias, device, dtype)  
        # weight shape: (out_features, in_features)
        weight_shape = self.weight.shape
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)

        self.in_features = in_features
        self.out_features = out_features
        
        triu1 = torch.triu_indices(self.in_features, self.in_features, device=self.weight.device, dtype=torch.long)
        triu_len = triu1.shape[1]
        tri_vec = self.weight.new_zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

        scat_idx = triu1[0] * self.in_features + triu1[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        U = self._tri_vec_to_mat(self.tri_vec, self.in_features, self.scat_idx)

        composite_weight = self.weight @ U
        
        return F.linear(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

class ResamplingFactLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        # init as Linear
        super().__init__(
               in_features, out_features, bias, device, dtype)  
        # weight shape: (out_features, in_features)
        weight_shape = self.weight.shape
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.weight)

        self.in_features = in_features
        self.out_features = out_features
        
        triu1 = torch.triu_indices(self.in_features, self.in_features, device=self.weight.device, dtype=torch.long)
        triu_len = triu1.shape[1]
        tri_vec = self.weight.new_zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

        scat_idx = triu1[0] * self.in_features + triu1[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        U = self._tri_vec_to_mat(self.tri_vec, self.in_features, self.scat_idx)
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.weight)

        composite_weight = self.weight @ U
        
        return F.linear(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U


class ResamplingDoubleFactLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        # init as Linear
        super().__init__(
               in_features, out_features, bias, device, dtype)  
        # weight shape: (out_features, in_features)
        weight_shape = self.weight.shape
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", new_weight)
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.weight)
        new_weight_og = torch.empty_like(self.weight)
        self.register_buffer("weight_og", new_weight_og)
        nn.init.kaiming_normal_(self.weight_og)
        #nn.init.orthogonal_(self.weight_og)
        #ideal to have kaiming_normal
        #need to explore assumptions on scale

        self.in_features = in_features
        self.out_features = out_features
        
        triu1 = torch.triu_indices(self.in_features, self.in_features, device=self.weight.device, dtype=torch.long)
        triu_len = triu1.shape[1]
        tri_vec = self.weight.new_zeros((triu_len,))
        self.tri_vec = Parameter(tri_vec)

        scat_idx = triu1[0] * self.in_features + triu1[1]
        self.register_buffer("scat_idx", scat_idx, persistent=False)

    def forward(self, input: Tensor) -> Tensor:
        U = self._tri_vec_to_mat(self.tri_vec, self.in_features, self.scat_idx)
        composite_weight = self.weight @ U
        x1 = F.linear(input, composite_weight, self.bias)
        composite_weight_og = self.weight_og @ U
        x2 = F.linear(input, composite_weight_og, self.bias)
        #return torch.cat([x1, x2], dim=0)
        return torch.cat([x1, x2], dim=1)

    def resample(self):
        #nn.init.orthogonal_(self.weight)
        nn.init.kaiming_normal_(self.weight)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

