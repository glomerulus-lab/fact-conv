import torch
import argparse
import os
from LC_models import V1_CIFAR10, V1_CIFAR100
from test_models_safety import PostExp, PreExp
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from distutils.util import strtobool

def tri_vec_to_mat(module, vec, n):
    U = torch.zeros((n, n), **module.factory_kwargs)
    U[torch.triu_indices(n, n, **module.factory_kwargs).tolist()] = vec
    U = exp_diag(module, U)
    return U
    
def exp_diag(module, mat):
    exp_diag = torch.exp(torch.diagonal(mat))
    n = mat.shape[0]
    mat[range(n), range(n)] = exp_diag
    return mat

def construct_spatial_cov(module):
    U2 = tri_vec_to_mat(module, module.tri2_vec, module.kernel_size[0] * module.kernel_size[1])
    U2 = exp_diag(module, U2)
    cov = U2.T @ U2
    if cov.requires_grad:
        cov = cov.cpu().detach().numpy()
    else:
        cov = cov.cpu().numpy()
    return cov 

def construct_channel_cov(module):
    U1 = tri_vec_to_mat(module, module.tri1_vec, module.in_channels //
            module.groups)
    U1 = exp_diag(module, U1)
    cov = U1.T @ U1
    if cov.requires_grad:
        cov = cov.cpu().detach().numpy()
    else:
        cov = cov.cpu().numpy()
    return cov

def get_eigens(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvalues, eigenvectors

def plot_eigenvalues(spatial_eigenvalues, channel_eigenvalues, name):
    fig, ax = plt.subplots(2, figsize=(12,8)) 
    ax[0].semilogy(np.flip(np.sort(spatial_eigenvalues)))
    ax[0].set_xlabel('Eigenvalues')
    ax[0].set_ylabel('Variance')
    ax[0].set_title("Spatial Eigenvalues")
    
    ax[1].semilogy(np.flip(np.sort(channel_eigenvalues)))
    ax[1].set_xlabel('Eigenvalues')
    ax[1].set_ylabel('Variance')
    ax[1].set_title("Channel Eigenvalues")
    plt.suptitle('Eigenvalues')
    fig.subplots_adjust(hspace=0.4)
    plt.savefig('{}_eigenvalues.png'.format(name))
    

def plot_covariance(spatial_cov, channel_cov, name):
    figure, ax = plt.subplots(2, figsize=(8,8))
    spatial_cov *= spatial_cov.shape[0] / np.trace(spatial_cov)
    fig = ax[0].imshow(spatial_cov)
    plt.colorbar(fig)
    ax[0].set_title("Spatial Covariance")

    channel_cov *= channel_cov.shape[0] / np.trace(channel_cov)
    fig = ax[1].imshow(channel_cov)
    plt.colorbar(fig)
    figure.subplots_adjust(hspace=0.4)
    ax[1].set_title("Channel Covariance")
    plt.savefig('{}_covariances.png'.format(name))


def plot_eigenvectors(eigenvalues, eigenvectors, name):
    eigenvalues = np.flip(eigenvalues)
    # calling np.flip() on sorted eigenvectors affects the matshow images
    # so instead we reverse the order in which they are plotted
    n_vectors = len(eigenvalues)
    reshape_dim = int(np.sqrt(n_vectors))
    fig, ax = plt.subplots(nrows=reshape_dim, ncols=reshape_dim, figsize=(12,8), layout='constrained')
    idx=0
    for row in ax:
        for col in row:
            col.axis('off')
            col.imshow(eigenvectors[:, -idx-1].reshape(reshape_dim,
                reshape_dim))
            col.set_title(str(eigenvalues[idx]), fontsize=7)
            idx += 1
    fig.suptitle("Spatial Eigenvectors")
    #fig.subplots_adjust(wspace=0.1, hspace=0.5, top=0.88)
    plt.savefig('{}_spatial_eigenvectors.png'.format(name))

def get_modules(model):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            modules.append(module)
    return modules

def plot_bins(model, module_num, filename):
    n_bins = 50
    fig, ax = plt.subplots(2, figsize=(6,8))
    modules = get_modules(model)
    module = modules[module_num]
    tri1vec = module.tri1_vec
    tri2vec = module.tri2_vec
    ax[0].hist(tri1vec.cpu().detach().numpy(), bins=n_bins)
    ax[0].set_title("Tri1 Hist")
    ax[1].hist(tri2vec.cpu().detach().numpy(), bins=n_bins)
    ax[1].set_title("Tri2 Hist")
    fig.subplots_adjust(hspace=0.4)
    plt.suptitle(filename)
    plt.savefig("{}_histograms.png".format(filename))

def plot_tri(model, module_num, filename):
    fig, ax = plt.subplots(2, figsize=(6,8))
    modules = get_modules(model)
    module = modules[module_num]
    tri1vec = module.tri1_vec
    tri2vec = module.tri2_vec
    n = module.in_channels // module.groups
    m = module.kernel_size[0] * module.kernel_size[1]
    U1 = tri_vec_to_mat(module, tri1vec, n)

    U2 = tri_vec_to_mat(module, tri2vec, m)
    plot1 = ax[0].matshow(U1.cpu().detach().numpy())
    fig.colorbar(plot1)
    ax[0].set_title("Tri1 Matrix")
    plot2 = ax[1].matshow(U2.cpu().detach().numpy())
    fig.colorbar(plot2)
    ax[1].set_title("Tri2 Matrix")
    plt.suptitle(filename)
    fig.subplots_adjust(hspace=0.4)
    plt.savefig("{}_matrices.png".format(filename))

def plot_all(model, module_num, filename):
    modules = get_modules(model)

    # get the module we are analyzing
    module = modules[module_num]

    spatial_cov = construct_spatial_cov(module)
    channel_cov = construct_channel_cov(module)
    
    spatial_evals, spatial_evecs = get_eigens(spatial_cov)
    channel_evals, channel_evecs = get_eigens(channel_cov)

    plot_eigenvalues(spatial_evals, channel_evals, filename)
    plot_covariance(spatial_cov, channel_cov, filename)
    plot_eigenvectors(spatial_evals, spatial_evecs, filename)
#
    plot_bins(model, module_num, filename)
    plot_tri(model, module_num, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    parser.add_argument('--freeze_spatial', dest='freeze_spatial', 
                        type=lambda x: bool(strtobool(x)), default=True, 
                        help="freeze spatial filters for LearnableCov models")
    parser.add_argument('--freeze_channel', dest='freeze_channel', 
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="freeze channels for LearnableCov models")
    parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                        help="initialization for spatial filters for LearnableCov models")
    parser.add_argument('--model', type=str, default='CIFAR10', 
                        choices=['CIFAR10', 'CIFAR100', 'post', 'pre'])
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--model_name', type=str, default="", 
                        help="name of saved model to load")
    parser.add_argument('--filename', type=str, default="ModelImgs",
                        help="filenames for saved image dir")
    parser.add_argument('--layer_num', type=int, default=0,
                        help="which layer num to analyze")
 
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
    
    if args.model == "post":
        model = PostExp(args.s, args.f, args.scale, args.bias,
                args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
        print("Post Exp Resnet")
    elif args.model == "pre":
        model = PreExp(args.s, args.f, args.scale, args.bias, 
                args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
        print("Pre Exp model")
        print("Pre Exp Resnet")
    elif args.model == "CIFAR10":
        model = V1_CIFAR10(100, args.s, args.f, args.scale, args.bias,
                args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
        print("CIFAR10 Model")
    elif args.model == "CIFAR100":
        model = V1_CIFAR100(100, args.s, args.f, args.scale, args.bias,
                args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
        print("CIFAR100 Model")

    src_dir = "../saved-models/muawi_models/" + args.model_name
    model_init = torch.load(src_dir)
    print("Model Loaded: ", src_dir)
    #postexp_LSLC = torch.load("../saved-models/muawi_models/postexp_LSLC.pt")
    #preexp_LSLC = torch.load("../saved-models/muawi_models/preexp_LSLC.pt")
     
    image_src = os.path.join("saved-images/", args.filename)
    os.makedirs(image_src, exist_ok=True)
    os.chdir(image_src)
 
    model.load_state_dict(model_init)
    plot_all(model, args.layer_num, args.filename)
