from rainbow import calc_svd
import torch
import torch.nn as nn
import tensorly

import numpy as np

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + 1e-4)

class SVD(torch.autograd.Function):
    """Low-rank SVD with manually re-implemented gradient.

    This function calculates the low-rank SVD decomposition of an arbitary
    matrix and re-implements the gradient such that we can regularize the
    gradient.

    Parameters
    ----------
    A : tensor
        Input tensor with at most 3 dimensions. Usually is 2 dimensional. if
        3 dimensional the svd is batched over the first dimension.

    size : int
        Slightly over-estimated rank of A.
    """

    @staticmethod
    def forward(self, A, size):
        #U, S, V = torch.linalg.svd(A)#, False)
        U, S, V = torch.svd_lowrank(A, size, 2)
        #V=V.T
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA, None



class NewAlignment(nn.Module):
    """Procurstes Alignment module used with rainbow networks when adapting for evaluation.

    This module splits the input along the channel/2nd dimension into the generated
    path and reference path, calculates the cross-covariance, and then
    calculates the alignment. This module also saves a running
    cross-covariance during train mode and uses a pre-calculated alignment
    matrix on evaluation mode.

    Parameters
    ----------
    size : int
        Tells the low-rank SVD solver what rank to calculate. Divided by 2. 
    rank : int
        Simply holds the unmodified size of what rank to calculate. rank = size*2
    state: int
        Notes if we are using the generated (0) or reference (1) path. Usually
        modified by recursive function.
    x : tensor
        Input tensor with at least 2 dimensions. If 4-dimensional we reshape
        the paths such that the channel dimension is in the 2nd dimension and
        all other dimensions (batch x spatial) are combined into the first dimension. 

    Buffers
    -------
    cov : tensor
        Cross-Covariance matrix. This is collected over multiple iterations on
        the training set. 
    alignment : tensor
        Alignment matrix which is calculated from cov and then stored.
    """
    def __init__(self, size, rank, mode='momentum', mom=1.0):
        super().__init__()
        self.svd = SVD.apply
        self.rank = rank
        if size < rank+1:
            self.size = size//2
        else:
            self.size = rank
        self.cov = torch.zeros((rank,rank)).cuda()
        self.total = 0
        self.mode=mode
        self.mom=mom
        alignment=torch.zeros((rank,rank)).cuda()
        self.register_buffer("alignment", alignment, persistent=True)

    def reset(self):
        self.cov = torch.zeros((self.rank,self.rank)).cuda()

    def forward(self, x):
        # changing path
        x1 = x[:, 0 : x.shape[1]//2, :, :]
        # fixed path
        x2 = x[:, x.shape[1]//2 : , :, :]
        
        if x.ndim == 4:
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape((-1, x1.shape[-1]))
            
            x2 = x2.permute(0, 2, 3, 1)
            x2 = x2.reshape((-1, x2.shape[-1]))

        if self.training:
            cov = x1.T@x2
            self.total += x1.shape[0]
            if self.mode == "average":
                self.cov = cov + self.cov.detach()
                temp_cov = self.cov
                temp_total = self.total
                U, S, V = self.svd(self.cov/self.total, self.size)

            elif self.mode == "momentum":
                #self.cov = (.1)*cov + (0.9)*self.cov.detach()
                self.cov = (self.mom)*cov + (1-self.mom)*self.cov.detach()
                U, S, V = self.svd(self.cov, self.size)

            V_h = V.T
            alignment = U  @ V_h
            self.alignment = alignment
        else:
            alignment = self.alignment
        x1 =  x1@alignment

        if x.ndim == 4:
            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
            x_2 = x2.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
        return aligned_x
#        return torch.cat([aligned_x, x_2], dim=1)

class NewAltAlignment(nn.Module):
    """Procurstes Alignment module.

    This module splits the input along the channel/2nd dimension into the generated
    path and reference path, calculates the cross-covariance, and then
    calculates the alignment. 

    Parameters
    ----------
    size : int
        Tells the low-rank SVD solver what rank to calculate. Divided by 2. 
    rank : int
        Simply holds the unmodified size of what rank to calculate. rank = size*2
    state: int
        Notes if we are using the generated (0) or reference (1) path. Usually
        modified by recursive function.
    x : tensor
        Input tensor with at least 2 dimensions. If 4-dimensional we reshape
        the paths such that the channel dimension is in the 2nd dimension and
        all other dimensions (batch x spatial) are combined into the first dimension. 
    mode : str
        "Momentum" if we are in momentum mode or "Average" if we are in
        average mode during training
    mom: float
        Momentum value
    """
    def __init__(self, size, rank, mode='momentum', mom=1.0):
        super().__init__()
        self.svd = SVD.apply
        self.rank = rank
        if size < rank+1:
            self.size = size//2
        else:
            self.size = rank
        self.cov = torch.zeros((self.rank,self.rank)).cuda()
        self.total = 0
        self.state=0
        self.mode=mode
        self.mom=mom

    def reset(self):
        self.cov = torch.zeros((self.rank,self.rank)).cuda()

    def forward(self, x):

        if self.state == 1:
            return x
        # changing path
        x1 = x[0 : x.shape[0]//2]
        # fixed path
        x2 = x[x.shape[0]//2 : ]

        #print(x1.shape, self.rank, self.size)
        
        if x.ndim == 4:
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape((-1, x1.shape[-1]))
            
            x2 = x2.permute(0, 2, 3, 1)
            x2 = x2.reshape((-1, x2.shape[-1]))
        if self.training:
            cov = x1.T@x2
            self.total += x1.shape[0]

            if self.mode == "average":
                print(self.cov.shape, cov.shape) 
                print(self.rank, self.size)
                self.cov = cov + self.cov.detach()
                temp_cov = self.cov
                temp_total = self.total
                U, S, V = self.svd(self.cov/self.total, self.size)

            elif self.mode == "momentum":
                #self.cov = (.1)*cov + (0.9)*self.cov.detach()
                self.cov = (self.mom)*cov + (1-self.mom)*self.cov.detach()
                U, S, V = self.svd(self.cov, self.size)

            V_h = V.T
            alignment = U  @ V_h
            self.alignment = alignment
        else:
            alignment = self.alignment
        x1 =  x1@alignment

        if x.ndim == 4:
            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
        return aligned_x

#unused
#class ANewAlignment(nn.Module):
#    def __init__(self, size, rank):
#        super().__init__()
#        self.svd = SVD.apply
#        self.rank = rank
#        if size < rank+1:
#            self.size = size//2
#        else:
#            self.size = rank
#        self.cov = torch.zeros((rank,rank)).cuda()
#        self.total = 0
#        self.first_time = True
#
#    def forward(self, x):
#        # changing path
#        x1 = x# = x[:, 0 : x.shape[1]//2, :, :]
#        # fixed path
#        
#        if x.ndim == 4:
#            x1 = x1.permute(0, 2, 3, 1)
#            x1 = x1.reshape((-1, x1.shape[-1]))
#            
#        alignment = self.alignment#.detach()
#        self.first_time=False
#        x1 =  x1@alignment
#
#        if x.ndim == 4:
#            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
#                x1.shape[-1]).permute(0, 3, 1, 2)
#        return aligned_x
#
