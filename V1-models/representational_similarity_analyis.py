# -*- coding: utf-8 -*-
from __future__ import annotations
"""representational_similarity_analyis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-L27kDZfNHDGtzj8JNZGT8WagPRTAz_A
"""

import matplotlib.pyplot as plt
import argparse
import wandb
import os
from ConvModules import FactConv2dPreExp
import torch
import torch.nn as nn
from resnet import ResNet18
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms

import contextlib
from typing import Callable, Optional

from torch import Tensor, nn


from functools import partial
from typing import Callable, Literal

from torch.nn import functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import contextlib
from typing import Callable, Optional

parser = argparse.ArgumentParser(description="CCA Colab")
parser.add_argument("--model_dir", type=str, default="conv_fact_final", help="Model file to load")
parser.add_argument("--model_type", type=str, default='both', choices=['conv', 'fact',
    'both'], help="which models to look at")
parser.add_argument("--hook_size", type=int, default=2, help='size for hooks')
parser.add_argument("--hook_type", type=str, default='fft', choices=['fft', 'avgpool'],
        help='method used in hook is either fft or adaptive avg pooling')
parser.add_argument("--distance", type=str, default='svcca', choices=['pwcca', 'svcca', 'linear_cka',
    'procrustes'])
parser.add_argument('--singleton', dest='singleton', 
                    type=lambda x: bool(strtobool(x)), default=False, 
                    help="is this script running on the twin studies or the\
                    singleton studies")

#this should be an import statement
def replace_layers_keep_weight(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_keep_weight(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding,
                    bias=True if module.bias is not None else False)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)


#this should be an import statement
def replace_layers_grab_names(model, name='', array=[]):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_grab_names(module, name + n if name == "" else name + "." + n,  array)
        if isinstance(module, nn.Conv2d):
            ## simple module
            array.append(name + "." + n if name !="" else name + n)

args = parser.parse_args() 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
conv_model = ResNet18().to(device)
fact_model = deepcopy(conv_model).to(device)
replace_layers_keep_weight(fact_model)

save_dir\
= "/network/scratch/v/vivian.white/v1-models/saved-models/correct_twins/{}".format(args.model_dir)
print("Save dir: ", save_dir)
conv_model.load_state_dict(torch.load('{}/conv_model.pt'.format(save_dir)))
fact_model.load_state_dict(torch.load('{}/fact_model.pt'.format(save_dir)))
print("Loaded models!")
fact_model.to(device)
conv_names = []
replace_layers_grab_names(conv_model, array=conv_names)
print(conv_names)
fact_names = []
replace_layers_grab_names(fact_model, array=fact_names)
print(fact_names)

if args.model_type == 'conv':
    ref_model = conv_model
    compare_model = conv_model
elif args.model_type == 'fact':
    ref_model = fact_model
    compare_model = fact_model
else:
    ref_model = conv_model
    compare_model = fact_model 

run_name = "{}_{}_{}".format(args.model_type, args.hook_type, args.distance)
print("Run name: ", run_name)

wandb_dir = "/home/mila/v/vivian.white/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
#run_name = "OGVGG"

run = wandb.init(project="random_project", config=args,
        group="pytorch_cifar_cca", name=run_name, dir=wandb_dir)
#wandb.watch(net, log='all', log_freq=1)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=3000, shuffle=False, num_workers=8, drop_last=True)

criterion = nn.CrossEntropyLoss()

def test(net, epoch=0):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(correct/total)

test(conv_model)
test(fact_model)

iterate = iter(testloader)
batch = next(iterate)
batch = batch[0].to(device)
batch.shape
#this should be an import statement
handle_list = []
def replace_layers_attach_hook(model, hook, ref_name='', name=''):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_attach_hook(module, hook, ref_name, name + n if name == "" else name + "." + n)
        if isinstance(module, nn.Conv2d):
            ## simple module
            if (name + n if name == "" else name + "." + n) == ref_name:
              handle_list.append(module.register_forward_hook(hook))
    return_hook(ref_name)

def _svd(input: torch.Tensor
         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch.svd style
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, S, V

def _zero_mean(input: Tensor,
               dim: int
               ) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def _check_shape_equal(x: Tensor,
                       y: Tensor,
                       dim: int
                       ):
    if x.size(dim) != y.size(dim):
        raise ValueError(f'x.size({dim}) == y.size({dim}) is expected, but got {x.size(dim)=}, {y.size(dim)=} instead.')


def cca_by_svd(x: Tensor,
               y: Tensor
               ) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using only SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = _svd(x)
    u_2, s_2, v_2 = _svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = _svd(uu)
    # a @ (1 / s_1).diag() @ u, without creating s_1.diag()
    a = v_1 @ (1 / s_1[:, None] * u)
    b = v_2 @ (1 / s_2[:, None] * v)
    return a, b, diag


def cca_by_qr(x: Tensor,
              y: Tensor
              ) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using QR and SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    q_1, r_1 = torch.linalg.qr(x)
    q_2, r_2 = torch.linalg.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = _svd(qq)
    # a = r_1.inverse() @ u, but it is faster and more numerically stable
    a = torch.linalg.solve(r_1, u)
    b = torch.linalg.solve(r_2, v)
    return a, b, diag


def cca(x: Tensor,
        y: Tensor,
        backend: str
        ) -> tuple[Tensor, Tensor, Tensor]:
    """ Compute CCA, Canonical Correlation Analysis

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    _check_shape_equal(x, y, 0)

    if x.size(0) < x.size(1):
        raise ValueError(f'x.size(0) >= x.size(1) is expected, but got {x.size()=}.')

    if y.size(0) < y.size(1):
        raise ValueError(f'y.size(0) >= y.size(1) is expected, but got {y.size()=}.')

    if backend not in ('svd', 'qr'):
        raise ValueError(f'backend is svd or qr, but got {backend}')

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == 'svd' else cca_by_qr(x, y)


def _svd_reduction(input: Tensor,
                   accept_rate: float
                   ) -> Tensor:
    left, diag, right = _svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      input.new_ones(1, dtype=torch.long),
                      input.new_zeros(1, dtype=torch.long)
                      ).sum()
    return input @ right[:, : num]


def svcca_distance(x: Tensor,
                   y: Tensor,
                   accept_rate: float = .99,
                   backend: str = "svd"
                   ) -> Tensor:
    """ Singular Vector CCA proposed in Raghu et al. 2017.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        accept_rate: 0.99
        backend: svd or qr

    Returns:

    """

    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def pwcca_distance(x: Tensor,
                   y: Tensor,
                   backend: str = "svd"
                   ) -> Tensor:
    """ Projection Weighted CCA proposed in Marcos et al. 2018.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        backend: svd or qr

    Returns:

    """

    a, b, diag = cca(x, y, backend)
    a, _ = torch.linalg.qr(a)  # reorthonormalize
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


def _debiased_dot_product_similarity(z: Tensor,
                                     sum_row_x: Tensor,
                                     sum_row_y: Tensor,
                                     sq_norm_x: Tensor,
                                     sq_norm_y: Tensor,
                                     size: int
                                     ) -> Tensor:
    return (z
            - size / (size - 2) * (sum_row_x @ sum_row_y)
            + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2)))


def linear_cka_distance(x: Tensor,
                        y: Tensor,
                        reduce_bias: bool=False
                        ) -> Tensor:
    """ Linear CKA used in Kornblith et al. 19

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        reduce_bias: debias CKA estimator, which might be helpful when D is limited

    Returns:

    """

    _check_shape_equal(x, y, 0)

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    dot_prod = (y.t() @ x).norm('fro').pow(2)
    norm_x = (x.t() @ x).norm('fro')
    norm_y = (y.t() @ y).norm('fro')

    if reduce_bias:
        size = x.size(0)
        # (x @ x.t()).diag()
        sum_row_x = torch.einsum('ij,ij->i', x, x)
        sum_row_y = torch.einsum('ij,ij->i', y, y)
        sq_norm_x = sum_row_x.sum()
        sq_norm_y = sum_row_y.sum()
        dot_prod = _debiased_dot_product_similarity(dot_prod, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size)
        norm_x = _debiased_dot_product_similarity(norm_x.pow(2), sum_row_x, sum_row_x, sq_norm_x, sq_norm_x, size
                                                  ).sqrt()
        norm_y = _debiased_dot_product_similarity(norm_y.pow(2), sum_row_y, sum_row_y, sq_norm_y, sq_norm_y, size
                                                  ).sqrt()
    return 1 - dot_prod / (norm_x * norm_y)


def orthogonal_procrustes_distance(x: Tensor,
                                   y: Tensor,
                                   ) -> Tensor:
    """ Orthogonal Procrustes distance used in Ding+21

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW

    Returns:

    """
    _check_shape_equal(x, y, 0)

    frobenius_norm = partial(torch.linalg.norm, ord="fro")
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")

    x = _zero_mean(x, dim=0)
    x /= frobenius_norm(x)
    y = _zero_mean(y, dim=0)
    y /= frobenius_norm(y)
    # frobenius_norm(x) = 1, frobenius_norm(y) = 1
    # 0.5*d_proc(x, y)
    return 1 - nuclear_norm(x.t() @ y)


def fft_shift(input: torch.Tensor,
              dims: Optional[tuple[int, ...]] = None
              ) -> torch.Tensor:
    """ PyTorch version of np.fftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    return torch.fft.fftshift(input, dims)


def ifft_shift(input: torch.Tensor,
               dims: Optional[tuple[int, ...]] = None
               ) -> torch.Tensor:
    """ PyTorch version of np.ifftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    return torch.fft.ifftshift(input, dims)


def _rfft(self: Tensor,
          signal_ndim: int,
          normalized: bool = False,
          onesided: bool = True
          ) -> Tensor:
    # old-day's torch.rfft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")

    m = torch.fft.rfftn if onesided else torch.fft.fftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    return torch.view_as_real(m(self, dim=dim, norm="ortho" if normalized else None))


def _irfft(self: Tensor,
           signal_ndim: int,
           normalized: bool = False,
           onesided: bool = True,
           ) -> Tensor:
    # old-day's torch.irfft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")
    if not torch.is_complex(self):
        self = torch.view_as_complex(self)

    m = torch.fft.irfftn if onesided else torch.fft.ifftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    out = m(self, dim=dim, norm="ortho" if normalized else None)
    return out.real if torch.is_complex(out) else out

def fft_stuffs(input, size):
        h = input.size(2)
        input_fft = _rfft(input, 2, normalized=True, onesided=False)
        freqs = torch.fft.fftfreq(h, 1 / h, device=input.device)
        idx = (freqs >= -size / 2) & (freqs < size / 2)
        # BxCxHxWx2 -> BxCxhxwx2
        input_fft = input_fft[..., idx, :][..., idx, :, :]
        input = _irfft(input_fft, 2, normalized=True, onesided=False)
        return input

# now i wanna hook into the model

ref_activation = []
def ref_hook(mod, input, output):
    if args.hook_type == 'fft':
        ref_activation.append(fft_stuffs(output, args.hook_size).flatten(1))
    elif args.hook_type == 'avgpool':
        ref_activation.append(F.adaptive_avg_pool2d(output,
            (args.hook_size,args.hook_size)).flatten(1))

compare_activation = []
def compare_hook(mod, input, output):
    if args.hook_type == 'fft':
        compare_activation.append(fft_stuffs(output, args.hook_size).flatten(1))
    elif args.hook_type == 'avgpool':
        compare_activation.append(F.adaptive_avg_pool2d(output,
            (args.hook_size,args.hook_size)).flatten(1))

def return_hook(name):
  def print_hook(mod, input, output):
    print(name)
  return print_hook

for i in range(0, len(handle_list)):
  handle_list[0].remove()
  del handle_list[0]

torch.cuda.empty_cache()

with torch.no_grad():
  rows = []
  # hook into reference model. define row. compute quantity of interest
  #for i in range(0, len(conv_names)-1):
  for i in range(0, len(conv_names)):
    row = []
    ref_name = conv_names[i]
    replace_layers_attach_hook(ref_model, ref_hook, ref_name)
    _ = ref_model(batch)
    ref_act = ref_activation[0]
    del ref_activation[0]
    handle_list[0].remove()
    del handle_list[0]


    # now hook into comparison model. compute quantity of interest
    #for j in range(i+1, len(conv_names)):
    for j in range(0, len(conv_names)):
      compare_name = conv_names[j]
      print(ref_name, compare_name)
      replace_layers_attach_hook(compare_model, compare_hook, compare_name)
      _ = compare_model(batch)
      compare_act = compare_activation[0]
      del compare_activation[0]
      handle_list[0].remove()
      del handle_list[0]
      # commpute cca
      if args.distance == 'pwcca':
        _cca = pwcca_distance(ref_act, compare_act)
        row.append(_cca)
      elif args.distance == 'svcca':
        _cca = svcca_distance(ref_act, compare_act)
        row.append(_cca)
      elif args.distance == 'linear_cka':
        _cca = linear_cka_distance(ref_act, compare_act)
        row.append(_cca)
      elif args.distance == 'procrustes':
        _cca = orthogonal_procrustes_distance(ref_act, compare_act)
        row.append(_cca)
    rows.append(row)

new_array = [torch.stack(row) for row in rows]
final = torch.stack(new_array).cpu()
final = final.numpy()
run.log({"final-array": wandb.Table(data=[[s] for s in final], columns=['values'])})
print("Final: ", final)

