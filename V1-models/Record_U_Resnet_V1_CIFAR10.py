#Model w/ weights initialized according to V1 receptive fields

import torch
import torch.optim
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from datetime import datetime
import os
from distutils.util import strtobool
import argparse
import V1_models
import LC_models
from test_models_safety import PostExp, PreExp
import numpy as np
import random
from hooks import wandb_forwards_hook, wandb_backwards_hook
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

   
if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--name', type=str, default='Resnet_V1', 
                        help='filename for saved model')
    parser.add_argument('--trial', type=int, default=1, help='trial number')
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
    parser.add_argument('--net', type=str, default='post', choices=['post',
        'pre'], 
                        help="which convmodule to use")
    parser.add_argument('--seed', type=int, default=0, help='seed')
 
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    start = datetime.now()
    set_seeds(args.seed)

    if args.net == "post":
        model = PostExp(args.s, args.f, args.scale, args.bias, args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)
    elif args.net == "pre":
        model = PreExp(args.s, args.f, args.scale, args.bias, args.freeze_spatial, args.freeze_channel, args.spatial_init).to(device)

def _tri_vec_to_mat(module, vec, n):
        U = torch.zeros((n, n), **module.factory_kwargs)
        U[torch.triu_indices(n, n, **module.factory_kwargs).tolist()] = vec
        return U

def visualize_covariance(module, name):    
    U1 = _tri_vec_to_mat(module, module.tri1_vec, module.in_channels // module.groups)
    U2 = _tri_vec_to_mat(module, module.tri2_vec, module.kernel_size[0] * module.kernel_size[1])
    
    U2_cov = U2.T @ U2
    U2_eigenvalues, U2_eigenvectors = torch.linalg.eig(U2_cov)
    
    U2_eigenvalues = U2_eigenvalues.cpu().numpy()
    U2_eigenvectors = U2_eigenvectors.cpu().numpy()
    U2_cov = U2_cov.cpu().numpy()

    plt.semilogy(np.flip(np.sort(U2_eigenvalues)))
    plt.savefig('{}_U2_eigenvalues.png'.format(name))
    
    U2_cov *= U2_cov.shape[0] / np.trace(U2_cov)
    plt.matshow(U2_cov, aspect='auto')
    plt.savefig('{}_U2_covariance.png'.format(name))
  
    indices = np.argsort(U2_eigenvalues)
    sorted_U2_eigenvalues = [U2_eigenvalues[i] for i in indices]
    sorted_U2_eigenvalues = np.flip(sorted_U2_eigenvalues)
    sorted_U2_eigenvectors = [U2_eigenvectors[i] for i in indices]
    
    fig, ax = plt.subplots(nrows=1, ncols=9) 
    for idx, col in enumerate(ax):
        col.axis('off')
        col.matshow(np.real(sorted_U2_eigenvectors[8-idx].reshape((3,3))))
        col.set_title(str(np.real(sorted_U2_eigenvalues[idx])), fontsize=7)
    plt.savefig('{}_sorted_U2_eigenvectors.png'.format(name))

model_src = "../saved-models/muawi_models/postexp.pt"
model.load_state_dict(torch.load(model_src))

module_list = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d): 
        module_list.append(module)

module1 = module_list[-1]
print(module1)

visualize_covariance(module1, "last_module_pretrain")
