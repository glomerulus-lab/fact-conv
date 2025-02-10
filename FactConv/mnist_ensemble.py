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
import numpy as np
#https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net', type=str, default='alignmlp', help="which model to use")
parser.add_argument('--nonlin', type=str, default='sigmoid', help="which nonlin to use")
parser.add_argument('--resample_freq', type=int, default=3, help="0 no resample, 1 resample once per epoch, 2 resample all the time")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--epochs', type=int, default=100, help="epoch")
parser.add_argument('--width', type=int, default=512, help="width")
parser.add_argument('--rank', type=int, default=256, help="width")
parser.add_argument('--layer', type=int, default=2, help="width")
parser.add_argument('--resume', type=int, default=0, help="width")
parser.add_argument('--niters', type=int, default=2, help="width")
parser.add_argument('--eps', type=float, default=0.0, help="exp")
parser.add_argument('--batchsize', type=int, default=1000, help="width")
parser.add_argument('--dataset', type=str, default="mnist", help="width")
parser.add_argument('--align_rank', type=int, default=2048)
args = parser.parse_args()

nonlins = {'relu':nn.ReLU(), 'sigmoid':nn.Sigmoid()}
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

    model_save_dir = "/home/mila/v/vivian.white/scratch/sci4dl/rainbow_testing/"

if args.dataset == 'kmnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.KMNIST('.', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.KMNIST('.', train=False,
                       transform=transform)
    model_save_dir = "/home/mila/v/vivian.white/scratch/sci4dl/rainbow_testing_kmnist/"

if args.dataset == 'fmnist':
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST('.', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('.', train=False,
                       transform=transform)
    model_save_dir = "/home/mila/v/vivian.white/scratch/sci4dl/rainbow_testing_fmnist/"

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
wandb_dir = "/home/mila/v/vivian.white/scratch/wandb/"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)

print(args.layer)
def resample(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            resample(m1)
        if isinstance(m1, ResamplingDoubleFactLinear):
            m1.resample()

run = wandb.init(project="FactConv", config=args,
        #group="eval_mnists_final_sci4dl",
        group="eval_mnist_ensembles",
        name="{}_{}_{}_{}_{}_{}_align{}".format(args.dataset,args.nonlin,
            args.net, args.width, args.layer, args.seed, args.align_rank), dir=wandb_dir)

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
    """
    def __init__(self, size, rank):
        super().__init__()
        self.svd = SVD.apply
        self.rank = rank
        if size < rank+1:
            #self.size = min(size//2, 1024)
            self.size = size//2
        else:
            self.size = rank
        self.state=0

    def forward(self, x):
        #torch.cuda.empty_cache()
        if self.state == 1:
            return x
        # changing path
        x1 = x[:, 0 : x.shape[1]//2]#, :, :]
        # fixed path
        x2 = x[:, x.shape[1]//2 : ]#, :, :]

        
        if x.ndim == 4:
            x1 = x1.permute(0, 2, 3, 1)
            x1 = x1.reshape((-1, x1.shape[-1]))
            
            x2 = x2.permute(0, 2, 3, 1)
            x2 = x2.reshape((-1, x2.shape[-1]))
        cov = x1.T@x2
        U, S, V = self.svd(cov, self.size)
        V_h = V.T
        alignment = U  @ V_h
        x1 =  x1@alignment

        if x.ndim == 4:
            aligned_x = x1.reshape(-1, x.shape[2], x.shape[3],
                x1.shape[-1]).permute(0, 3, 1, 2)
        else:
            aligned_x = x1
        #torch.cuda.empty_cache()
        return aligned_x




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
    def __init__(self, size, rank, mode='average', mom=1.0):
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
        x1 = x[:, 0 : x.shape[1]//2]#, :, :]
        # fixed path
        x2 = x[:, x.shape[1]//2 :]# , :, :]
        
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
        else:
            aligned_x = x1
        return aligned_x
 
def realign(model):
    for (n1, m1) in model.named_children():
        if len(list(m1.children())) > 0:
            realign(m1)
        if isinstance(m1, Alignment):
            setattr(model, n1, NewAlignment(m1.rank, m1.rank))
        if isinstance(m1, NewAlignment):
            setattr(model, n1, NewAlignment(m1.rank, m1.rank))

   
class SimpleMLP(nn.Module):
    def __init__(self, rank=None, nonlin='relu'):
        super().__init__()
        nonlin = nonlins[nonlin]
        k=args.width
        mlp_layers = []
        for i in range(args.layer):
            mlp_layers.append(
                    nn.Sequential(
                    nn.BatchNorm1d(k, affine=True),
                    nn.Linear(k, k, bias=True),
                    nonlin,
                    )
                    )

        self.layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28*28, k, bias=True),
                    nonlin,
                    nn.Sequential(*mlp_layers),
                    nn.BatchNorm1d(k, affine=True),
                    #nn.LayerNorm(k),#, affine=True),
                    nn.Linear(k, 10)
                    )

    def forward(self, x):
        return self.layers(x)


class SimpleFactMLP(nn.Module):
    def __init__(self, rank=None, nonlin='relu'):
        super().__init__()
        nonlin = nonlins[nonlin]
        k=args.width
        mlp_layers = []
        for i in range(args.layer):
            mlp_layers.append(
                    nn.Sequential(
                    nn.BatchNorm1d(k, affine=True),
                    FactLinear(k, k, bias=True),
                    nonlin,
                    )
                    )

        self.layers = nn.Sequential(
                    nn.Flatten(),
                    FactLinear(28*28, k, bias=True),
                    nonlin,
                    nn.Sequential(*mlp_layers),
                    nn.BatchNorm1d(k, affine=True),
                    nn.Linear(k, 10)
                    )

    def forward(self, x):
        return self.layers(x)

class SimpleResampledDoubleFactProMLP(nn.Module):
    def __init__(self, rank=None, nonlin='relu'):
        super().__init__()
        nonlin = nonlins[nonlin]
        k=args.width
        mlp_layers = []
        for i in range(args.layer):
            mlp_layers.append(
                    nn.Sequential(
                    Alignment(rank, rank),
                    nn.BatchNorm1d(k, affine=True),
                    ResamplingDoubleFactLinear(k, k, bias=True),
                    nonlin,
                    )
                    )



        self.layers = nn.Sequential(
                    nn.Flatten(),
                    ResamplingDoubleFactLinear(28*28, k, bias=True),
                    nonlin,
                    nn.Sequential(*mlp_layers),
                    Alignment(rank, rank),
                    nn.BatchNorm1d(k, affine=True),
                    nn.Linear(k, 10)
                    )

    def forward(self, x):
        return self.layers(x)

if "factmlp" in args.net:
    rank = args.width
    model = SimpleFactMLP(rank, args.nonlin)
if "linearmlp" in args.net:
    rank = args.width
    model = SimpleMLP(rank, args.nonlin)
if "alignmlp" in args.net:
    rank = args.width
    rank = args.align_rank
    model = SimpleResampledDoubleFactProMLP(rank, args.nonlin)

model = model.to(device)#.double()
print(model)
#print(model(next(iter(train_loader))[0].to(device)).shape)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def correct(output, target):
    predicted_digits = output.argmax(1)                            
    correct_ones = (predicted_digits == target).type(torch.float)  
    return correct_ones.sum().item()                               

logger = {}
def test(test_loader, model, criterion, epoch, train_mode=False,
mode="minibatch_"):
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
    base_dir=model_save_dir+args.net+"_{}_nonlin_{}_layer_{}_width_{}_seed_{}_align{}".format(args.resample_freq,
            args.nonlin, args.layer, args.width, args.seed, args.align_rank)+ "_weight/ensemble/"

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)#.double()
            target = target.to(device)
            output=None
            # Do a forward pass
            for i in range(0,5):
                model_save_path = path +mode + str(i) + ".pt"
                sd = torch.load(model_save_path)
                net.load_state_dict(sd)
                if output is None:
                    output = net(data)
                else:
                    output += net(data)
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}% average loss: {test_loss:>7f}")

    #logger["test_acc"] = accuracy
    #logger["test_loss"] = test_loss
    return {"acc": accuracy, "loss": test_loss}
logger = {}
def train(data_loader, model, criterion, optimizer, epoch):
    model.eval()
    if True:
        def train_align(net):                                            
            for (n1, m1) in net.named_children():                      
                if len(list(m1.children())) > 0:                         
                    train_align(m1)                                          
                if isinstance(m1, NewAlignment): 
                    m1.train()
        train_align(model)


    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            # Copy data and targets to GPU
            data = data.to(device)#.double()
            target = target.to(device)
            
            # Do a forward pass
            output = model(data)
            
            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss

            # Count number of correct digits
            total_correct += correct(output, target)
        
        # Backpropagation
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")
    logger["train_acc"] = accuracy
    logger["train_loss"] = train_loss



start_epochs  = 0
#if args.resume:
base_dir=model_save_dir+args.net+"_{}_nonlin_{}_layer_{}_width_{}_seed_{}_align{}".format(args.resample_freq,
            args.nonlin, args.layer, args.width, args.seed, args.align_rank)+ "_weight/" + "99.pt"
if os.path.isfile(base_dir):
    storage = torch.load(base_dir)
    sd = storage['sd']
    optim_sd = storage['optim_sd']
    start_epochs  = storage['epochs'] + 1
    model.load_state_dict(sd)
    optimizer.load_state_dict(optim_sd)
else:
    print(base_dir)
    print("NOT RESUMING")
epoch=0
#sample 5 samples. save. evaluate each sample together. 
#calculate trainset alignment over each sample. save. evaluate.
base_dir=model_save_dir+args.net+"_{}_nonlin_{}_layer_{}_width_{}_seed_{}_align{}".format(args.resample_freq,
            args.nonlin, args.layer, args.width, args.seed, args.align_rank)+ "_weight/ensemble/"
os.makedirs(base_dir, exist_ok=True)
path = base_dir
print(path)


for i in range(0, 5):
    resample(model)
    net = copy.deepcopy(model)
    sd = net.state_dict()
    model_save_path = path + "minibatch_" + str(i) + ".pt"
    torch.save(sd, model_save_path)
    realign(net)
    train(train_loader, net, criterion, optimizer, epoch)
    sd = net.state_dict()
    model_save_path = path + "trainset_" + str(i) + ".pt"
    torch.save(sd, model_save_path)

post_resample_test = test(test_loader, model, criterion, epoch,
mode="minibatch_")
logger['minibatch_test_acc'] = post_resample_test['acc']
logger['minibatch_test_loss'] = post_resample_test['loss']

realign(model)
post_resample_test = test(test_loader, model, criterion, epoch,
mode="trainset_")
logger['trainset_test_acc'] = post_resample_test['acc']
logger['trainset_test_loss'] = post_resample_test['loss']

run.log(logger)

#pre_resample_test = test(test_loader, model, criterion, epoch)
#logger['pre_test_acc'] = pre_resample_test['acc']
#logger['pre_test_loss'] = pre_resample_test['loss']
#print(f"Training Model")
#realign(model)
#
#train(train_loader, model, criterion, optimizer, epoch)
#print(f"Model Final Performance")
#
#
