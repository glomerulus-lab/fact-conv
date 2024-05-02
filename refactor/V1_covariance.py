"""
The following code is copied from the Structured Random Features library,
https://github.com/glomerulus-lab/structured-random-features,
used under the following license:

The MIT License (MIT)
Copyright (c) 2021, Biraj Pandey

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

"""

import torch
from torch import Tensor
import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform


def V1_covariance_matrix(dim, size, spatial_freq, center, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with non-stationary 
    covariance. This matrix will be used to generate random 
    features inspired from the receptive-fields of V1 neurons.

    C(x, y) = exp(-|x - y|/(2 * spatial_freq))^2 * exp(-|x - m| / (2 * size))^2 * exp(-|y - m| / (2 * size))^2

    Parameters
    ----------

    dim : tuple of shape (2, 1)
        Dimension of random features.

    size : float
        Determines the size of the random weights 

    spatial_freq : float
        Determines the spatial frequency of the random weights  
    
    center : tuple of shape (2, 1)
        Location of the center of the random weights.

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    Returns
    -------

    C : array-like of shape (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """

    x = np.arange(dim[0])
    y = np.arange(dim[1])
    yy, xx = np.meshgrid(y, x)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    a = squareform(pdist(grid, 'sqeuclidean'))
    b = la.norm(grid - center, axis=1) ** 2
    c = b.reshape(-1, 1)
    C = np.exp(-a / (2 * spatial_freq ** 2)) * np.exp(-b / (2 * size ** 2)) * np.exp(-c / (2 * size ** 2)) \
        + 1e-5 * np.eye(dim[0] * dim[1])
    C *= scale * dim[0] * dim[1] / np.trace(C)
    return C


def V1_init(layer, size, spatial_freq, center, scale=1., bias=False, seed=None):
    '''
    Initialization for FactConv2d
    '''
    
    classname = layer.__class__.__name__
    assert classname.find('FactConv2d') != -1, 'This init only works for FactConv2d layers'
    assert center is not None, "center needed"

    out_channels, in_channels, xdim, ydim = layer.weight.shape
    dim = (xdim, ydim)
    
    C_patch = Tensor(V1_covariance_matrix(dim, size, spatial_freq, center, scale)).to(layer.weight.device)
    U_patch = torch.linalg.cholesky(C_patch, upper=True)
    n = U_patch.shape[0]
    # replace diagonal with logarithm for parameterization
    log_diag = torch.log(torch.diagonal(U_patch))
    U_patch[range(n), range(n)] = log_diag
    # form vector of upper triangular entries
    tri_vec = U_patch[torch.triu_indices(n, n, device=layer.weight.device).tolist()].ravel()
    with torch.no_grad():
        layer.tri2_vec.copy_(tri_vec)

    if bias == False:
        layer.bias = None
