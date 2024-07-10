import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pytorch_cifar_utils import set_seeds
from linear_modules import FactLinear, ResamplingFactLinear, ResamplingDoubleFactLinear
import time
import argparse
import os 
import copy
from rainbow import calc_svd
import wandb
import tensorly
#https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', type=str, default='alignment', help="which model to use")
parser.add_argument('--resample_freq', type=int, default=3, help="0 no resample, 1 resample once per epoch, 2 resample all the time")
parser.add_argument('--epochs', type=int, default=100, help="epoch")
parser.add_argument('--width', type=int, default=512, help="width")
parser.add_argument('--rank', type=int, default=200, help="width")
parser.add_argument('--layer', type=int, default=20, help="width")
parser.add_argument('--resume', type=int, default=0, help="width")
parser.add_argument('--niters', type=int, default=2, help="width")
parser.add_argument('--eps', type=float, default=0.0, help="exp")
parser.add_argument('--batchsize', type=int, default=1000, help="width")
parser.add_argument('--dataset', type=str, default="mnist", help="dataset")
parser.add_argument('--double', type=int, default=0, help="width")
parser.add_argument('--test_svd', type=int, default=0, help="width")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dataset == 'mnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('.', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('.', train=False,
                       transform=transform)


if args.dataset == 'kmnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.KMNIST('.', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.KMNIST('.', train=False,
                       transform=transform)

if args.dataset == 'fmnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST('.', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('.', train=False,
                       transform=transform)

model_save_dir  = "/home/mila/m/muawiz.chaudhary/scratch/svd_rainbow_testing_{}/".format(args.dataset)
set_seeds(0)
batchsize=6000000

train_loader = torch.utils.data.DataLoader(dataset1,
        batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize,
        shuffle=True)

for (x_train, y_train) in train_loader:
    x_train, y_train = x_train.to(device), y_train.to(device)

for (x_test, y_test) in test_loader:
    x_test, y_test = x_test.to(device), y_test.to(device)

dataset1 = torch.utils.data.TensorDataset(x_train, y_train)
dataset2 = torch.utils.data.TensorDataset(x_test, y_test)

batchsize=args.batchsize

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batchsize, shuffle=True)

set_seeds(0)
wandb_dir = "../../wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

print(args.layer)
def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1, ResamplingDoubleFactLinear):
            m1.resample()

name = args.net+"_{}_layer_{}_dataset_{}_rank_{}_batchsize_{}_niter_{}_eps_{}_double_{}_svdtesting_{}".format(args.resample_freq,
            args.layer, args.dataset, args.rank, args.batchsize, args.niters,
            args.eps, args.double, args.test_svd)

run = wandb.init(project="FactConv", config=args,
        group="svd_testing",
        name=name, dir=wandb_dir)


import numpy as np
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + args.eps)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        #U, S, V = torch.linalg.svd(A, False)
        U, S, V = torch.svd_lowrank(A, 200, 2)
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
        return dA


class Alignment(nn.Module):
    def __init__(self, rank=None, detach=False):
        super().__init__()
        self.detach = detach
        self.rank = rank
        self.svd = SVD.apply

    def forward(self, x):
        # changing path
        x1 = x[:, 0 : x.shape[-1]//2]
        # fixed path
        x2 = x[:, x.shape[-1]//2 : ]

        if self.detach:
            x2 = x2.detach()
        cov= x1.T@x2

        if self.rank is None:
            #U, S, V_h = torch.linalg.svd(cov +(args.eps)*torch.eye(cov.shape[0]).cuda(), full_matrices=False)
            U, S, V_h = SVD(cov)
        else:
            if args.test_svd:
                U, S, V = self.svd(cov)
            else:
                U, S, V = torch.svd_lowrank(cov, q=self.rank, niter=args.niters)
            V_h = V.T

        alignment = U  @ V_h
        return x1@alignment

class SimpleFactMLP(nn.Module):
    def __init__(self, rank=None, detach=False):
        super().__init__()
        k=args.width
        self.detach = detach
        mlp_layers = []
        for i in range(args.layer):
            mlp_layers.append(
                    nn.Sequential(
                    nn.BatchNorm1d(k, affine=True),
                    FactLinear(k, k, bias=True),
                    nn.ReLU(),
                    )
                    )

        self.layers = nn.Sequential(
                    nn.Flatten(),
                    FactLinear(28*28, k, bias=True),
                    nn.ReLU(),
                    nn.Sequential(*mlp_layers),
                    nn.BatchNorm1d(k, affine=True),
                    #nn.LayerNorm(k),#, affine=True),
                    nn.Linear(k, 10)
                    )

    def forward(self, x):
        return self.layers(x)

class SimpleResampledDoubleFactProMLP(nn.Module):
    def __init__(self, rank=None, detach=False):
        super().__init__()
        k=args.width
        self.detach = detach
        mlp_layers = []
        for i in range(args.layer):
            mlp_layers.append(
                    nn.Sequential(
                    Alignment(rank, detach),
                    nn.BatchNorm1d(k, affine=True),
                    ResamplingDoubleFactLinear(k, k, bias=True),
                    nn.ReLU(),
                    )
                    )



        self.layers = nn.Sequential(
                    nn.Flatten(),
                    ResamplingDoubleFactLinear(28*28, k, bias=True),
                    nn.ReLU(),
                    nn.Sequential(*mlp_layers),
                    Alignment(rank, detach),
                    nn.BatchNorm1d(k, affine=True),
                    nn.Linear(k, 10)
                    )

    def forward(self, x):
        return self.layers(x)


if "factmlp" in args.net:
    detach = True if "detach" in args.net else False
    rank = args.rank
    model = SimpleFactMLP(rank, detach)

if "full_rank" in args.net:
    detach = True if "detach" in args.net else False
    rank = None
    model = SimpleResampledDoubleFactProMLP(rank, detach)

if "alignment" in args.net:
    detach = True if "detach" in args.net else False
    rank = args.rank
    model = SimpleResampledDoubleFactProMLP(rank, detach)
model = model.to(device)
if args.double:
    model = model.double()
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def correct(output, target):
    predicted_digits = output.argmax(1)                            
    correct_ones = (predicted_digits == target).type(torch.float)  
    return correct_ones.sum().item()                               


logger = {}
def train(data_loader, model, criterion, optimizer, epoch):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        if args.double:
            data = data.double()
        target = target.to(device)
        if args.resample_freq >= 2:
            resample(model)
        
        # Do a forward pass
        output = model(data)
        
        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")
    logger["train_acc"] = accuracy
    logger["train_loss"] = train_loss


def test(test_loader, model, criterion, epoch, train_mode=True):
    model.eval()
    net = copy.deepcopy(model)
    if train_mode:
        net.train()
    else:
        net.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            if args.double:
                data = data.double()
            target = target.to(device)
            if args.resample_freq >= 3:
                resample(net)
        
            # Do a forward pass
            output = net(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}% average loss: {test_loss:>7f}")

    return {"acc": accuracy, "loss": test_loss}


def save_model(model, net, epoch):
    storage = {"sd":model.state_dict(), "optim_sd": optimizer.state_dict(),
            "epochs": epoch}
    model_save_path=model_save_dir+name+ "_weight/" 
    os.makedirs(model_save_path, exist_ok=True)
    path = model_save_path 
    model_save_path=model_save_path + str(epoch) + ".pt"

    torch.save(storage, model_save_path)

    if os.path.lexists(path + "latest"):
        os.unlink(path + "latest")
    os.symlink(model_save_path, path + "latest")

start_epochs  = 0
if args.resume:
    base_dir=model_save_dir+name+ "_weight/" + "latest"
    if os.path.isfile(base_dir):
        storage = torch.load(base_dir)
        sd = storage['sd']
        optim_sd = storage['optim_sd']
        start_epochs  = storage['epochs'] + 1
        model.load_state_dict(sd)
        optimizer.load_state_dict(optim_sd)
    else:
        print("NOT RESUMING")

epochs = args.epochs
for epoch in range(start_epochs, epochs):
    if args.resample_freq >= 1:
        resample(model)
    print(f"Epoch: {epoch+1}")
    #print("Resampling Performance")
    print(f"Training Model")
    train(train_loader, model, criterion, optimizer, epoch)
    print(f"Model Final Performance")

    after_resample_train = test(train_loader, model, criterion, epoch)
    after_resample_test = test(test_loader, model, criterion, epoch)
    logger['after_resample_train_acc'] = after_resample_train['acc']
    logger['after_resample_train_loss'] = after_resample_train['loss']
    logger['after_resample_test_acc'] = after_resample_test['acc']
    logger['after_resample_test_loss'] = after_resample_test['loss']

    after_resample_train = test(train_loader, model, criterion, epoch, False)
    after_resample_test = test(test_loader, model, criterion, epoch, False)
    logger['Eval_stats/after_resample_train_acc'] = after_resample_train['acc']
    logger['Eval_stats/after_resample_train_loss'] = after_resample_train['loss']
    logger['Eval_stats/after_resample_test_acc'] = after_resample_test['acc']
    logger['Eval_stats/after_resample_test_loss'] = after_resample_test['loss']

    run.log(logger)
    save_model(model, args.net, epoch)
