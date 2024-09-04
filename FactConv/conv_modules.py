import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import math
from cov_refactor import Covariance, LowRankK1Covariance,\
LowRankK1DiagCovariance

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
        
        # self.in_features = self.in_channels // self.groups * \
        #    self.kernel_size[0] * self.kernel_size[1]

        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = Covariance(channel_triu_size, "abs")
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


class GMMFactConv2d(nn.Conv2d):
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
        dtype=None,
        k=1
    ) -> None:
        # init as Conv2d
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, True, padding_mode, device, dtype)
        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        new_weight = torch.empty_like(self.weight)
        del self.weight # remove Parameter, create buffer

        new_bias = torch.empty_like(self.bias)
        #del self.bias # remove Parameter, create buffer
        self.logits = Parameter(torch.ones((k, 1)))

        self.k = k
        for i in range(0, k):
            print("HI")
            self.register_buffer("weight_{}".format(i), new_weight)
            nn.init.kaiming_normal_(self.get_buffer("weight_{}".format(i)))#self.weight)
            
            self.in_features = self.in_channels // self.groups * \
                self.kernel_size[0] * self.kernel_size[1]
            triu1 = torch.triu_indices(self.in_channels // self.groups,
                                           self.in_channels // self.groups,
                                          device=new_bias.device,
                                          dtype=torch.long)
            scat_idx1 = triu1[0]*self.in_channels//self.groups + triu1[1]
            self.register_buffer("scat_idx1".format(i), scat_idx1, persistent=False)

            triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                           self.kernel_size[0]
                                           * self.kernel_size[1],
                                          device=new_bias.device,
                                          dtype=torch.long)
            scat_idx2 = triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
            self.register_buffer("scat_idx2".format(i), scat_idx2, persistent=False)

            triu1_len = triu1.shape[1]
            triu2_len = triu2.shape[1]

            #tri1_vec = self.weight.new_zeros((triu1_len,))
            tri1_vec = new_weight.new_zeros((triu1_len,))
            diag1 = triu1[0] == triu1[1]
            tri1_vec[diag1] = 1.0
            #self.tri1_vec = Parameter(tri1_vec)
            self.register_parameter("tri1_vec_{}".format(i),Parameter(
                tri1_vec))

            #tri2_vec = self.weight.new_zeros((triu2_len,))
            tri2_vec = new_weight.new_zeros((triu2_len,))
            diag2 = triu2[0] == triu2[1]
            tri2_vec[diag2] = 1.0
            self.register_parameter("tri2_vec_{}".format(i),
                    Parameter(tri2_vec))
            #self.tri2_vec = Parameter(tri2_vec)
            self.register_parameter("bias_{}".format(i),
                        Parameter(new_bias.new_zeros((new_bias.shape))))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.get_buffer("weight_{}".format(i)))
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.get_parameter("bias_{}".format(i)), -bound, bound)



    def forward(self, input: Tensor) -> Tensor:
        reps = []
        for i in range(0, self.k):
            U1   = self._tri_vec_to_mat(self.get_parameter("tri1_vec_{}".format(i)), self.in_channels //
                                      self.groups, self.scat_idx1)
            U2 = self._tri_vec_to_mat(self.get_parameter("tri2_vec_{}".format(i)), self.kernel_size[0] * self.kernel_size[1],
                    self.scat_idx2)
            # flatten over filter dims and contract
            composite_weight = _contract(self.get_buffer("weight_{}".format(i)), U1.T, 1)
            composite_weight = _contract(
                torch.flatten(composite_weight, -2, -1), U2.T, -1
            ).reshape(self.weight_0.shape)
            reps.append(self._conv_forward(input, composite_weight,
                    self.get_parameter("bias_{}".format(i))))
        reps = torch.stack(reps, dim=0)
        logits = self.logits.softmax(0)
        reps = (logits * reps.reshape(self.k, -1)).reshape(reps.shape)
        return reps.sum(0)

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.get_buffer("weight_0").new_zeros((n*n)).view(n, n).fill_diagonal_(1.0).view(n*n).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, torch.abs(U.diagonal()))# .exp_()
        return U



class ResamplingDoubleFactConv2d(nn.Conv2d):
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
        #nn.init.orthogonal_(self.weight)

        new_weight = torch.empty_like(self.weight)
        self.register_buffer("resampling_weight", new_weight)
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

        channel_triu_size = self.in_channels // self.groups
        spatial_triu_size = self.kernel_size[0] * self.kernel_size[1]

        self.channel = Covariance(channel_triu_size, "abs")
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
        x2 = self._conv_forward(input, composite_weight, self.bias)
        if self.state == 1:
            return x2

        #nn.init.orthogonal_(self.resampling_weight)
        composite_resampling_weight = _contract(self.resampling_weight, U1.T, 1)
        composite_resampling_weight = _contract(
            torch.flatten(composite_resampling_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        x1 = self._conv_forward(input, composite_resampling_weight, self.bias)
        return torch.cat([x1, x2], dim=1)


    def resample(self):
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

    def ref_resample(self):
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.resampling_weight)


class LowRankResamplingDoubleFactConv2d(nn.Conv2d):
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

        self.channel = LowRankK1Covariance(channel_triu_size, channel_k, "abs")
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
        x2 = self._conv_forward(input, composite_weight, self.bias)
        if self.state == 1:
            return x2

        #nn.init.orthogonal_(self.resampling_weight)
        composite_resampling_weight = _contract(self.resampling_weight, U1.T, 1)
        composite_resampling_weight = _contract(
            torch.flatten(composite_resampling_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        x1 = self._conv_forward(input, composite_resampling_weight, self.bias)
        return torch.cat([x1, x2], dim=1)


    def resample(self):
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

    def ref_resample(self):
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.resampling_weight)


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
        x2 = self._conv_forward(input, composite_weight, self.bias)
        if self.state == 1:
            return x2

        #nn.init.orthogonal_(self.resampling_weight)
        composite_resampling_weight = _contract(self.resampling_weight, U1.T, 1)
        composite_resampling_weight = _contract(
            torch.flatten(composite_resampling_weight, -2, -1), U2.T, -1
        ).reshape(self.weight.shape)
        x1 = self._conv_forward(input, composite_resampling_weight, self.bias)
        return torch.cat([x1, x2], dim=1)


    def resample(self):
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)

    def ref_resample(self):
        nn.init.kaiming_normal_(self.weight)
        #nn.init.orthogonal_(self.resampling_weight)


class FactProjConv2d(nn.Conv2d):
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
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                                  self.groups, self.scat_idx1)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
        U = torch.kron(U1, U2)
        #U = self._exp_diag(U)
        print(U.shape, self.weight.shape)
               
        matrix_shape = (self.out_channels, self.in_features)
        print(torch.reshape(self.weight, matrix_shape).shape)

                                
 


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

class OffFactConv2d(nn.Conv2d):
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
        mask = triu1[0] != triu1[1]
        scat_idx1 = triu1[0][mask]*self.in_channels//self.groups + triu1[1][mask]
        self.register_buffer("scat_idx1", scat_idx1, persistent=False)

        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                      device=self.weight.device,
                                      dtype=torch.long)
        mask = triu2[0] != triu2[1]
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


class NewResamplingDoubleFactConv2d(nn.Conv2d):
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
        #nn.init.orthogonal_(self.weight)

        new_weight = torch.empty_like(self.weight)
        self.register_buffer("resampling_weight", new_weight)
        nn.init.kaiming_normal_(self.resampling_weight)
        #nn.init.orthogonal_(self.resampling_weight)
        
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
        diag1 = triu1[0] == triu1[1]
        tri1_vec[diag1] = 1.0
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = self.weight.new_zeros((triu2_len,))
        diag2 = triu2[0] == triu2[1]
        tri2_vec[diag2] = 1.0
        self.tri2_vec = Parameter(tri2_vec)
        self.state=0



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
        x2 = self._conv_forward(input[input.shape[0]//2:], composite_weight, self.bias)

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



    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = self.weight.new_zeros((n*n)).view(n, n).fill_diagonal_(1.0).view(n*n).scatter_(0, scat_idx, vec).view(n, n)
        U = torch.diagonal_scatter(U, torch.abs(U.diagonal()))# .exp_()
        return U
