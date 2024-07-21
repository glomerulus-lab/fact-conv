from rainbow import calc_svd
import torch
import torch.nn as nn
import tensorly

import numpy as np

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + 1e-4)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A, size):
        #U, S, V = torch.linalg.svd(A)#, False)
        #V = V.T
        U, S, V = torch.svd_lowrank(A, size, 2)
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



class Alignment(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.svd = SVD.apply
        #self.bn1 = nn.BatchNorm2d(size)
        self.rank = rank
        if size < rank+1:
            #self.size = int(size*.50)
            self.size = size//2
        else:
            self.size = rank

    def forward(self, x):
        # changing path
        #self.bn1.eval()
        x1 = x[:, 0 : x.shape[1]//2, :, :]
        # fixed path
        x2 = x[:, x.shape[1]//2 : , :, :]
        
        if x.ndim == 4:
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape((-1, x1.shape[-1]))
            
            x2 = x2.permute(0, 2, 3, 1)
            x2 = x2.reshape((-1, x2.shape[-1]))
        #x1_mu = x1.mean(0, keepdims=True)
        #x2_mu = x2.mean(0, keepdims=True)
        #x1 = x1 - x1_mu
        #x2 = x2 - x2_mu
        #scale = (torch.norm(x2)/torch.norm(x1))
        #x1 = x1*scale
        cov = x1.T@x2
        #cov = cov/x1.shape[0]
        U, S, V = self.svd(cov, self.size)
        V_h = V.T
        alignment = U  @ V_h
        x1 =  x1@alignment

        #x1 = x1 +x2_mu - scale*x1_mu@alignment

        if x.ndim == 4:
            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
        #aligned_x = self.bn1(aligned_x)
        return aligned_x

