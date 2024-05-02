"""
Parametric Scattering in Kymatio
=======================================
Here we demonstrate how to develop a novel scattering utilizing the Kymatio
code base. We then train a linear classifier and visualize the filters.
"""

###############################################################################
# Preliminaries
# -------------
# Kymatio can be used to develop novel scattering algorithms. We'll use class
# inheritance to massively reduce the amount of code we have to write. 
# Furthermore, we'll use Torch as we want differentiability and optimizability.
from kymatio.torch import Scattering2D
import torch

###############################################################################
# We'll also import various libraries needed for implementation

import types

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


###############################################################################
# Implementation of parametric scattering - wavelet generation
# -----------------------------------------------------------------------------
# We wish to develop a version of the scattering transform where the wavelet
# parameters can be optimized. Currently, in Kymatio, the filter generation 
# is defined using Numpy. Numpy does not give us differentiablity, so we 
# reimplement the morlet generation utilizing Torch.

def morlets(grid_or_shape, theta, xis, sigmas, slants, morlet=True, ifftshift=True, fft=True, return_gabors=False):
    """Creates morlet wavelet filters from inputs

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            theta -- global orientations of the wavelets
            xis -- frequency scales of the wavelets
            sigmas -- gaussian window scales of the wavelets
            slants -- slants of the wavelets
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)
        Returns:
            wavelets -- the wavelet filters

    """
    orientations = torch.cat((torch.cos(theta).unsqueeze(1),torch.sin(theta).unsqueeze(1)), dim =1)
    n_filters, ndim = orientations.shape
    wave_vectors = orientations * xis[:, np.newaxis]
    _, _, gauss_directions = torch.linalg.svd(orientations[:, np.newaxis])
    gauss_directions = gauss_directions / sigmas[:, np.newaxis, np.newaxis]
    indicator = torch.arange(ndim, device=slants.device) < 1
    slant_modifications = (1.0 * indicator + slants[:, np.newaxis] * ~indicator)
    gauss_directions = gauss_directions * slant_modifications[:, :, np.newaxis]
    if return_gabors:
        wavelets, gabors = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft, return_gabors=True)
    else:
        wavelets = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft)

    norm_factors = (2 * 3.1415 * sigmas * sigmas / slants).unsqueeze(1)

    if type(grid_or_shape) == tuple:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape[0]]).unsqueeze(2).repeat(1,1,grid_or_shape[1])
    else:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape.shape[1]]).unsqueeze(2).repeat(1,1,grid_or_shape.shape[2])

    wavelets = wavelets / norm_factors
    if return_gabors:
      return gabors/norm_factors

    return wavelets


def raw_morlets(grid_or_shape, wave_vectors, gaussian_bases, morlet=True, ifftshift=True, fft=True, return_gabors=False):
    """ Helper function for creating morlet filters

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            wave_vectors -- directions of the wave part of the morlet wavelet
            gaussian_bases -- bases of the gaussian part of the morlet wavelet
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)
        Returns:
            filters -- the wavelet filters before normalization
            
    """
    n_filters, n_dim = wave_vectors.shape
    assert gaussian_bases.shape == (n_filters, n_dim, n_dim)

    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)
    else:
        shape = grid_or_shape.shape[1:]
        grid = grid_or_shape

    waves = torch.exp(1.0j * torch.matmul(grid.T, wave_vectors.T).T)
    gaussian_directions = torch.matmul(grid.T, gaussian_bases.T.reshape(n_dim, n_dim * n_filters)).T
    gaussian_directions = gaussian_directions.reshape((n_dim, n_filters) + shape)
    radii = torch.norm(gaussian_directions, dim=0)
    gaussians = torch.exp(-0.5 * radii ** 2)
    signal_dims = list(range(1, n_dim + 1))
    gabors = gaussians * waves

    if morlet:
        gaussian_sums = gaussians.sum(dim=signal_dims, keepdim=True)
        gabor_sums = gabors.sum(dim=signal_dims, keepdim=True).real
        morlets = gabors - gabor_sums / gaussian_sums * gaussians
        filters = morlets
    else:
        filters = gabors
    if ifftshift:
        filters = torch.fft.ifftshift(filters, dim=signal_dims)
        gabors = torch.fft.ifftshift(gabors, dim=signal_dims)	
    if fft:
        filters = torch.fft.fftn(filters, dim=signal_dims)
        gabors = torch.fft.fftn(gabors, dim=signal_dims)	
    if return_gabors:
        return filters, gabors
    else:
        return filters


###############################################################################
# Implementation of parametric scattering - Updating dictionaires
# -----------------------------------------------------------------------------
# Kymatio utilizes dictionaries to store wavelets. These dictionaries are fed into 
# the core scattering algorithm, where the wavelets and meta information are used 
# to compute scattering coefficents. 
#
# We will need to update the dictionaries prior 
# to the start of training so that gradients can flow from wavelets to wavelet 
# parameters, as currently the wavelets that live in the dictionaries are numpy 
# arrays. Note also that every iteration of training we will update these 
# dictionaries. 
#
# We define a series of helper functions for this purpose.


def create_filters_params(J, L):
    """ Create reusable tight frame initialized filter parameters: orientations, xis, sigmas, sigmas     

        Parameters:
            J -- scale of the scattering
            L -- number of orientation for the scattering
        Returns:
            params -- list that contains the parameters of the filters

    """
    orientations = []
    xis = []
    sigmas = []
    slants = []

    for j in range(J):
        for theta in range(L):
            sigmas.append(0.8 * 2**j)
            t = ((int(L-L/2-1)-theta) * np.pi / L)
            xis.append(3.0 / 4.0 * np.pi /2**j)
            slant = 4.0/L
            slants.append(slant)
            orientations.append(t) 

    xis = torch.tensor(xis, requires_grad=True, dtype=torch.float32)
    sigmas = torch.tensor(sigmas, requires_grad=True, dtype=torch.float32)
    slants = torch.tensor(slants, requires_grad=True, dtype=torch.float32)
    orientations = torch.tensor(orientations, requires_grad=True, dtype=torch.float32)  
    params = [orientations, xis, sigmas, slants]

    return  params


def update_psi(J, psi, wavelets):
    """ Update the psi dictionnary with the new wavelets

        Parameters:
            J -- scale for the scattering
            psi -- dictionnary of filters
            wavelets -- wavelet filters

        Returns:
            psi -- dictionnary of filters
    """
    wavelets = wavelets.real.contiguous().unsqueeze(3)
    for i, d in enumerate(psi):
        for c in range(len(d['levels'])):
            d['levels'][c] = periodize_filter_fft(wavelets[i].squeeze(2), c).unsqueeze(2)
    return psi




def update_wavelets_phi(J, L, phi, shape, params_filters, equivariant=False):
        """ Create wavelets and update the psi dictionnary with the new wavelets

            Parameters:
                J -- scale for the scattering
                psi -- dictionnary of filters
                shape -- shape of the scattering (scattering.M_padded, scattering.N_padded,)
                params_filters -- the parameters used to create wavelets

            Returns:
                psi -- dictionnary of filters
                wavelets -- wavelets filters
        """
        wavelets  = morlets(shape, params_filters[0], params_filters[1],
                                   params_filters[2], params_filters[3], return_gabors=True)
        phi = update_phi(J, phi, wavelets)
        return phi, wavelets


def update_phi(J, phi, wavelets):
    """ Update the psi dictionnary with the new wavelets

        Parameters:
            J -- scale for the scattering
            psi -- dictionnary of filters
            wavelets -- wavelet filters

        Returns:
            psi -- dictionnary of filters
    """
    wavelets = wavelets.real.contiguous().unsqueeze(3)
    for i in range(len(phi['levels'])):
        phi['levels'][i] = periodize_filter_fft(wavelets[0].squeeze(2), i).unsqueeze(2)
    return phi



def periodize_filter_fft(x, res):
    """ Periodize the filter in fourier space

        Parameters:
            x -- signal to periodize in Fourier
            res -- resolution to which the signal is cropped


        Returns:
            periodized -- It returns a crop version of the filter, assuming that the convolutions
                          will be done via compactly supported signals.
    """
    s1, s2 = x.shape[0], x.shape[1]
    periodized = x.reshape(2**res, s1// 2**res, 2**res, s2//2**res).mean(dim=(0,2))
    return periodized


def update_wavelets_psi(J, L, psi, shape, params_filters):
        """ Create wavelets and update the psi dictionnary with the new wavelets

            Parameters:
                J -- scale for the scattering
                psi -- dictionnary of filters
                shape -- shape of the scattering (scattering.M_padded, scattering.N_padded,)
                params_filters -- the parameters used to create wavelets

            Returns:
                psi -- dictionnary of filters
                wavelets -- wavelets filters
        """
        wavelets  = morlets(shape, params_filters[0], params_filters[1],
                                   params_filters[2], params_filters[3])
        psi = update_psi(J, psi, wavelets)
        return psi, wavelets


###############################################################################
# Implementation of parametric scattering - Parametric scattering module
# -----------------------------------------------------------------------------
# We have four considerations to take into account 
#
# 1. We already have code that we can re-use; i.e. the core scattering algorithm, 
# inheritance of nn.Module. Thus we inherit from Scattering2DTorch.
#
# 2. With a parametric scattering, we want to optimize our wavelet parameters,
# so we want gradients to flow from wavelets, but we dont want to optimize 
# our wavelets. To solve this, we register the wavelet parameters as a nn.Parameter 
# wrapped together in a nn.ParameterList, and register the wavelets as module buffers.
#
# 3. At the end of every training iteration, the wavelets in our dictionaries will 
# be stale. We want to use our updated wavelet params to regenerate the wavelets 
# every training iteration. We utilize a forward_hook piror to every forward pass.
#
# 4. We want to reshape our representation to be fed into a 2D Batchnorm.
# For convinence, we hook at the end of each forward pass to reshape the output.

def _register_single_filter(self, v, n):
    self.register_buffer('tensor' + str(n), v)

class ParametricScattering2D(Scattering2D):
    """
    A learnable scattering nn.module 
    """

    def __init__(self, J, N, M, L=8, parametric=True):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
        """
        super(ParametricScattering2D, self).__init__(J=J, shape=(M, N), L=L)

        self.M_coefficient = self.shape[0]/(2**self.J)
        self.N_coefficient = self.shape[1]/(2**self.J)
        self.scatteringTrain = True

        self.n_coefficients =  L*L*J*(J-1)//2 + 1 + L*J  
        
        self.params_filters = create_filters_params(J, L) #kymatio init

        shape = (self._M_padded, self._N_padded,)
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)

        self.psi, _ = update_wavelets_psi(J, L, self.psi, shape, self.params_filters)
        self.params_filters_phi = [torch.tensor([0], requires_grad=True, dtype=torch.float32), torch.tensor([0], requires_grad=True, dtype=torch.float32),
                torch.tensor([0.8 * 2**(J-1)], requires_grad=True,
                    dtype=torch.float32), torch.tensor([1.0],
                        requires_grad=True, dtype=torch.float32)]

        self.phi, _ = update_wavelets_phi(J, L, self.phi, shape, self.params_filters_phi)

        #for c,phi in enumerate(self.phi['levels']):
        #    self.phi['levels'][c] = torch.from_numpy(phi).unsqueeze(-1).cuda()


        self.register_single_filter = types.MethodType(_register_single_filter, self)
        self.register_filters()

        for i in range(0, len(self.params_filters)):
            self.params_filters[i] = nn.Parameter(self.params_filters[i])
            self.register_parameter(name='scattering_params_'+str(i), param=self.params_filters[i])
        self.params_filters = nn.ParameterList(self.params_filters)

        for i in range(0, len(self.params_filters_phi)):
            self.params_filters_phi[i] = nn.Parameter(self.params_filters_phi[i])
            self.register_parameter(name='scattering_params_phi_'+str(i), param=self.params_filters_phi[i])
        self.params_filters_phi = nn.ParameterList(self.params_filters_phi)

        self.register_buffer(name='grid', tensor=grid)
        
        def updateFilters_hook(self, ip):
            """Update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (self.training or self.scatteringTrain):
                phi, psi = self.load_filters()
                wavelets = morlets(self.grid, 
                            self.scattering_params_0,
                            self.scattering_params_1,
                            self.scattering_params_2, 
                            self.scattering_params_3)
                wavelets_phi = morlets(self.grid,
                            self.scattering_params_phi_0,
                            self.scattering_params_phi_1,
                            self.scattering_params_phi_2,
                            self.scattering_params_phi_3, return_gabors=True)
                self.psi = update_psi(self.J, psi, wavelets)
                self.phi = update_phi(self.J, phi, wavelets_phi)
                self.register_filters()
                self.scatteringTrain = self.training
        if parametric:
            self.pre_hook = self.register_forward_pre_hook(updateFilters_hook)

        def reshape_hook(self, x, S):
            S = S[:,:, -self.n_coefficients:,:,:]
            S = S.reshape(S.size(0), -1, S.size(3), S.size(4))
            return S
        #self.register_forward_hook(reshape_hook)


###############################################################################
# We visualize our wavelets prior to training and after. We look at 
# the Littlewood Paley diagram. Dekha means "Look! See!" in Punjabi. 

def littlewood_paley_dekha(S, display=True):
    """Plots each wavelet in Fourier space, creating a 
    comprehensive view of the scattering filterbank."""
    wavelets = morlets(S.grid, S.params_filters[0], 
                                      S.params_filters[1], S.params_filters[2], 
                                        S.params_filters[3]).detach().numpy()
    gabors = morlets(S.grid, S.params_filters[0], 
                                      S.params_filters[1], S.params_filters[2], 
                                      S.params_filters[3], return_gabors=True).cpu().detach().numpy() 
    grid = S.grid
    lp = (np.abs(wavelets) ** 2).sum(0)
    fig, ax = plt.subplots()
    
    plt.imshow(np.fft.fftshift(lp))
    if display:
        grid = grid + 18
        for i in np.abs(gabors) ** 2:
            wave = np.fft.fftshift(i)
            ax.contour(grid[1], grid[0], wave, 1, colors='white')
    plt.show()
