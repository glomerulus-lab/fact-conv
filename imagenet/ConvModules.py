import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
import time
import copy

class FactConv2dPostExp(nn.Conv2d):
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
        self.factory_kwargs = factory_kwargs

        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        weight_shape = self.weight.shape
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", torch.empty(weight_shape, **factory_kwargs))
        nn.init.kaiming_normal_(self.weight)
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups)
        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1])
        triu1_len = triu1.shape[1]
        triu2_len = triu2.shape[1]
        tri1_vec = torch.zeros((triu1_len,),
            **factory_kwargs)
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = torch.zeros((triu2_len,), **factory_kwargs)
        self.tri2_vec = Parameter(tri2_vec)
        
        
    def forward(self, input: Tensor) -> Tensor:
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels // self.groups)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1])
        U = torch.kron(U1, U2)
        U = self._exp_diag(U)
        
        matrix_shape = (self.out_channels, self.in_features)
        composite_weight = torch.reshape(
            torch.reshape(self.weight, matrix_shape) @ U,
            self.weight.shape
        )
        
        return self._conv_forward(input, composite_weight, self.bias)

    def _tri_vec_to_mat(self, vec, n):
        U = torch.zeros((n, n), **self.factory_kwargs)
        U[torch.triu_indices(n, n, **self.factory_kwargs).tolist()] = vec
        return U

    def _exp_diag(self, mat):
        exp_diag = torch.exp(torch.diagonal(mat))
        n = mat.shape[0]
        mat[range(n), range(n)] = exp_diag
        return mat
class FactConv2dPreExp(nn.Conv2d):
    def __init__(
        self,
        in_channels:   int,
        out_channels:  int,
        kernel_size:   _size_2_t,
        stride:        _size_2_t = 1,
        padding:       Union[str, _size_2_t] = 0,
        dilation:      _size_2_t = 1,
        groups:        int = 1,
        bias:          bool = True,
        padding_mode:  str = 'zeros',  # TODO: refine this type
        device         = None,
        dtype          = None
    ) -> None:
        # init as Conv2d
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                         dilation, groups, bias, padding_mode, device, dtype)

        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        weight = self.weight.clone()
        del self.weight                # remove Parameter, create buffer
        self.register_buffer("weight", weight)
        nn.init.kaiming_normal_(self.weight)
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                       device=weight.device)
        #self.scat_idx1 = triu1[0]*self.in_channels//self.groups + triu1[1]
        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                       device=weight.device)

        #self.scat_idx2 = triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
        triu1_len = triu1.shape[1]
        triu2_len = triu2.shape[1]
        
        tri1_vec = torch.zeros((triu1_len,), device=weight.device, dtype=weight.dtype)
        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = torch.zeros((triu2_len,), device=weight.device, dtype=weight.dtype)
        self.tri2_vec = Parameter(tri2_vec)


        self.register_buffer("scat_idx1", triu1[0]*self.in_channels//self.groups + triu1[1], persistent=False)
        self.register_buffer("scat_idx2", triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1], persistent=False)

    def construct_Us(self):
        self.tri1_vec = Parameter(self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                self.groups,self.scat_idx1))
        self.tri2_vec = Parameter(self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2))

        
    def forward(self, input: Tensor) -> Tensor:
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                self.groups,self.scat_idx1)
#        print(self.in_channels//self.groups)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
#
        U = torch.kron(U1, U2) 
        #s1 = time.time()
        #U = torch.kron(U1, U2) 
        #torch.cuda.synchronize()
        #print("torch.kron",time.time()-s1)
        #U = self._kron(U1, U2)
        #s1 = time.time()
        #U = self._kron(U1, U2)
        #torch.cuda.synchronize()
        #print("self.kron", time.time()-s1)
 
        matrix_shape = (self.out_channels, self.in_features)
        composite_weight = torch.reshape(
            torch.reshape(self.weight, matrix_shape) @ U,
            self.weight.shape
        )
        output = self._conv_forward(input, composite_weight, self.bias)
        return output

    def _kron(self, a, b):
        a_shape = a.shape
        b_shape = b.shape
        c_shape = (a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])
        
        a = a.reshape(-1, 1)
        b = b.reshape(1, -1)
        
        product = a@b
        product = product.reshape(a_shape[0], a_shape[1], b.shape[0], b.shape[1])
        product = product.permute(0, 2, 1, 3)
        product = product.reshape(c_shape[0], c_shape[1])
        return product

    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = torch.zeros((n*n), device=vec.device, dtype=vec.dtype).scatter_(0,scat_idx,vec).view(n,n)
        U = torch.diagonal_scatter(U, U.diagonal().exp_())
        return U

class FactConv2dPreExpOld(nn.Conv2d):
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

        #device = self.weight.device
        print("Device: ", device)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = factory_kwargs

        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        weight_shape = self.weight.shape
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", torch.empty(weight_shape, **factory_kwargs))
        nn.init.kaiming_normal_(self.weight)
        
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]
        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                       **factory_kwargs)
        self.scat_idx1=triu1[0]*self.in_channels//self.groups + triu1[1]
        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                       **factory_kwargs)

        self.scat_idx2=triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
        triu1_len = triu1.shape[1]
        triu2_len = triu2.shape[1]
        tri1_vec = torch.zeros((triu1_len,),
            **factory_kwargs)

        self.tri1_vec = Parameter(tri1_vec)

        tri2_vec = torch.zeros((triu2_len,), **factory_kwargs)
        self.tri2_vec = Parameter(tri2_vec)

    def construct_Us(self):
        self.tri1_vec = Parameter(self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                self.groups,self.scat_idx1))
        self.tri2_vec = Parameter(self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2))

        
    def forward(self, input: Tensor) -> Tensor:
        #U1 = self.tri1_vec
        #U2 = self.tri2_vec
        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                self.groups,self.scat_idx1)
#        print(self.in_channels//self.groups)
        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
#
        U = torch.kron(U1, U2) 
        #s1 = time.time()
        #U = torch.kron(U1, U2) 
        #torch.cuda.synchronize()
        #print("torch.kron",time.time()-s1)
        #U = self._kron(U1, U2)
        #s1 = time.time()
        #U = self._kron(U1, U2)
        #torch.cuda.synchronize()
        #print("self.kron", time.time()-s1)
 
        matrix_shape = (self.out_channels, self.in_features)
        composite_weight = torch.reshape(
            torch.reshape(self.weight, matrix_shape) @ U,
            self.weight.shape
        )
        output = self._conv_forward(input, composite_weight, self.bias)
        return output

    def _kron(self, a, b):
        a_shape = a.shape
        b_shape = b.shape
        c_shape = (a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])
        
        a = a.reshape(-1, 1)
        b = b.reshape(1, -1)
        
        product = a@b
        product = product.reshape(a_shape[0], a_shape[1], b.shape[0], b.shape[1])
        product = product.permute(0, 2, 1, 3)
        product = product.reshape(c_shape[0], c_shape[1])
        return product



    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = torch.zeros((n* n),
                **self.factory_kwargs).scatter_(0,scat_idx,vec).view(n,n)
        #U[torch.triu_indices(n, n, **self.factory_kwargs).tolist()] = vec
        U = torch.diagonal_scatter(U,U.diagonal().exp_())
        #self._exp_diag(U)
        return U

    #def _tri_vec_to_mat(self, vec, n):
    #    U = torch.zeros((n, n), **self.factory_kwargs)
    #    U[torch.triu_indices(n, n, **self.factory_kwargs).tolist()] = vec
    #    U = self._exp_diag(U)
    #    return U


    def _exp_diag(self, mat):
        exp_diag = torch.exp(torch.diagonal(mat))
        n = mat.shape[0]
        mat[range(n), range(n)] = exp_diag
        return mat

#        
#    def forward(self, input: Tensor) -> Tensor:
#        s1 = time.time()
#        U1 = self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
#                self.groups,self.scat_idx1)
#        print(time.time()-s1)
#        print(self.in_channels//self.groups)
#        U2 = self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
#                self.scat_idx2)
#        U = torch.kron(U1, U2) 
#        matrix_shape = (self.out_channels, self.in_features)
#        composite_weight = torch.reshape(
#            torch.reshape(self.weight, matrix_shape) @ U,
#            self.weight.shape
#        )
#        output = self._conv_forward(input, composite_weight, self.bias)
#        return output
#
#    def _tri_vec_to_mat(self, vec, n, scat_idx):
#        U = torch.zeros((n* n),
#                **self.factory_kwargs).scatter_(0,scat_idx,vec).view(n,n)
#        #U[torch.triu_indices(n, n, **self.factory_kwargs).tolist()] = vec
#        U = torch.diagonal_scatter(U,U.diagonal().exp_())
#        #self._exp_diag(U)
#        return U
#
#
class FactConv2dK(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        k: int = 1,
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
        self.factory_kwargs = factory_kwargs
        self.k = k
        # weight shape: (out_channels, in_channels // groups, *kernel_size)
        weight_shape = self.weight.shape
        del self.weight # remove Parameter, create buffer
        self.register_buffer("weight", torch.empty(weight_shape, **factory_kwargs))
        nn.init.kaiming_normal_(self.weight)
         
        self.in_features = self.in_channels // self.groups * \
            self.kernel_size[0] * self.kernel_size[1]

        triu1 = torch.triu_indices(self.in_channels // self.groups,
                                       self.in_channels // self.groups,
                                       **factory_kwargs)
        self.scat_idx1=triu1[0]*self.in_channels//self.groups + triu1[1]
        triu2 = torch.triu_indices(self.kernel_size[0] * self.kernel_size[1],
                                       self.kernel_size[0]
                                       * self.kernel_size[1],
                                       **factory_kwargs)

        self.scat_idx2=triu2[0]*self.kernel_size[0]*self.kernel_size[1] + triu2[1]
        triu1_len = triu1.shape[1]
        triu2_len = triu2.shape[1]
        tri1_vec = torch.zeros((triu1_len,), **factory_kwargs)
        tri2_vec = torch.zeros((triu2_len,), **factory_kwargs)

        self.tri1_vecs\
        = nn.ParameterList([nn.Parameter(copy.deepcopy(tri1_vec)) for i in
                range(self.k)])
        self.tri2_vecs\
        = nn.ParameterList([nn.Parameter(copy.deepcopy(tri2_vec)) for i in
                range(self.k)])

        for param in self.tri1_vecs:
            nn.init.trunc_normal_(param, std=0.02)
        for param in self.tri2_vecs:
            nn.init.trunc_normal_(param, std=0.02)


    def construct_Us(self):
        self.tri1_vec = Parameter(self._tri_vec_to_mat(self.tri1_vec, self.in_channels //
                self.groups,self.scat_idx1))
        self.tri2_vec = Parameter(self._tri_vec_to_mat(self.tri2_vec, self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2))

        
    def forward(self, input: Tensor) -> Tensor:
        krons = []
        comp_weights = []
        for i in range(self.k):
            U1 = self._tri_vec_to_mat(self.tri1_vecs[i], self.in_channels //
                self.groups,self.scat_idx1)
            U2 = self._tri_vec_to_mat(self.tri2_vecs[i], self.kernel_size[0] * self.kernel_size[1],
                self.scat_idx2)
            U = torch.kron(U1, U2) 
            krons.append(U)

            matrix_shape = (self.out_channels, self.in_features)
            comp_weight = torch.reshape(
                    torch.reshape(self.weight, matrix_shape) @ U,
                    self.weight.shape)
            comp_weights.append(comp_weight)

        arr = torch.stack(krons, dim=0)
        U = torch.mean(arr, dim=0)
        arr2 = torch.stack(comp_weights, dim=0)
        composite_weight = torch.mean(arr2, dim=0)

        # matrix_shape = (self.out_channels, self.in_features)
        # composite_weight = torch.reshape(
        #     torch.reshape(self.weight, matrix_shape) @ U,
        #     self.weight.shape
        # )
        
        output = self._conv_forward(input, composite_weight, self.bias)
        return output

    def _kron(self, a, b):
        a_shape = a.shape
        b_shape = b.shape
        c_shape = (a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])
        
        a = a.reshape(-1, 1)
        b = b.reshape(1, -1)
        
        product = a@b
        product = product.reshape(a_shape[0], a_shape[1], b.shape[0], b.shape[1])
        product = product.permute(0, 2, 1, 3)
        product = product.reshape(c_shape[0], c_shape[1])
        return product



    def _tri_vec_to_mat(self, vec, n, scat_idx):
        U = torch.zeros((n* n),
                **self.factory_kwargs).scatter_(0,scat_idx,vec).view(n,n)
        #U[torch.triu_indices(n, n, **self.factory_kwargs).tolist()] = vec
        U = torch.diagonal_scatter(U,U.diagonal().exp_())
        #self._exp_diag(U)
        return U


    def _exp_diag(self, mat):
        exp_diag = torch.exp(torch.diagonal(mat))
        n = mat.shape[0]
        mat[range(n), range(n)] = exp_diag
        return mat
