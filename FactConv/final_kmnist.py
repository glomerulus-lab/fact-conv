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
parser.add_argument('--niters', type=int, default=2, help="width")
parser.add_argument('--resume', type=int, default=0, help="width")
parser.add_argument('--eps', type=float, default=0.0, help="exp")
parser.add_argument('--batchsize', type=int, default=1000, help="width")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

#args.epochs = 100
dataset1 = datasets.KMNIST('.', train=True, download=True,
                   transform=transform)
dataset2 = datasets.KMNIST('.', train=False,
                   transform=transform)
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

run = wandb.init(project="FactConv", config=args,
        group="final_double_testing_kmnist",
        name=args.net+"_{}_layer_{}_rank_{}_batchsize_{}_niter_{}_eps_{}".format(args.resample_freq,
            args.layer,args.rank, args.batchsize, args.niters, args.eps), dir=wandb_dir)
class Alignment(nn.Module):
    def __init__(self, rank=None, detach=False):
        super().__init__()
        self.detach = detach
        self.rank = rank

    def forward(self, x):
        # changing path
        x1 = x[:, 0 : x.shape[-1]//2]
        # fixed path
        x2 = x[:, x.shape[-1]//2 : ]

        if self.detach:
            x2 = x2.detach()
        cov= x1.T@x2

        if self.rank is None:
            U, S, V_h = torch.linalg.svd(cov +(args.eps)*torch.eye(cov.shape[0]).cuda(), full_matrices=False)
        else:
            U, S, V = torch.svd_lowrank(cov+(args.eps)*torch.eye(cov.shape[0]).cuda(), q=self.rank, niter=args.niters)
            V_h = V.T

        alignment = U  @ V_h  # (C_in_reference, C_in_generated)
        return x1@alignment#(x @ V) #+ (x_mean @ V)


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
model = model.to(device).double()
print(model)
#print(model(next(iter(train_loader))[0].to(device)).shape)


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
        data = data.to(device).double()
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
            data = data.to(device).double()
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

    #logger["test_acc"] = accuracy
    #logger["test_loss"] = test_loss
    return {"acc": accuracy, "loss": test_loss}

model_save_dir = "/home/mila/m/muawiz.chaudhary/scratch/rainbow_testing_kmnist/"

def save_model(model, net, epoch):
    storage = {"sd":model.state_dict(), "optim_sd": optimizer.state_dict(),
            "epochs": epoch}
    model_save_path=model_save_dir+args.net+"_{}_layer_{}_rank_{}_batchsize_{}_niter_{}_eps_{}".format(args.resample_freq,
            args.layer,args.rank, args.batchsize, args.niters,
            args.eps)+ "_weight/" 
    os.makedirs(model_save_path, exist_ok=True)
    path = model_save_path 
    model_save_path=model_save_path + str(epoch) + ".pt"

    torch.save(storage, model_save_path)

    if os.path.lexists(path + "latest"):
        os.unlink(path + "latest")
    os.symlink(model_save_path, path + "latest")
    #os.rename(path+".tmp", path+"")

start_epochs  = 0
if args.resume:
    base_dir="/home/mila/m/muawiz.chaudhary/scratch/rainbow_testing_kmnist/"+args.net+"_{}_layer_{}_rank_{}_batchsize_{}_niter_{}_eps_{}".format(args.resample_freq,
            args.layer,args.rank, args.batchsize, args.niters,
            args.eps) + "_weight/" + "latest"
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
    #before_resample_train = test(train_loader, model, criterion, epoch)
    #before_resample_test = test(test_loader, model, criterion, epoch)
    #logger['before_resample_train_acc'] = before_resample_train['acc']
    #logger['before_resample_train_loss'] = before_resample_train['loss']
    #logger['before_resample_test_acc'] = before_resample_test['acc']
    #logger['before_resample_test_loss'] = before_resample_test['loss']

    #before_resample_train = test(train_loader, model, criterion, epoch, False)
    #before_resample_test = test(test_loader, model, criterion, epoch, False)
    #logger['Eval_stats/before_resample_train_acc'] = before_resample_train['acc']
    #logger['Eval_stats/before_resample_train_loss'] = before_resample_train['loss']
    #logger['Eval_stats/before_resample_test_acc'] = before_resample_test['acc']
    #logger['Eval_stats/before_resample_test_loss'] = before_resample_test['loss']


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
