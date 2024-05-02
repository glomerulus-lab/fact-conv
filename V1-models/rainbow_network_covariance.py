'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
import torchvision.transforms as transforms
import time 
import os
import argparse
import copy

from pytorch_cifar_utils import progress_bar, set_seeds
from test_models_safety import PostExp, PreExp
import wandb
from distutils.util import strtobool
from resnet import ResNet18
from vgg import VGG
from test_models_safety import PostExp, PreExp
from  layers_model import ThreeLayer_CIFAR10, Sequential_ThreeLayer_CIFAR10
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.matmul.allow_tf32 = True

    
def save_model(args, model):
    src = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_rainbow_models/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    
    torch.save(model.state_dict(), model_dir+ "/model.pt")
    torch.save(args, model_dir+ "/args.pt")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--net', type=str, default='vgg', choices=['vgg', 'vggbn',
    'resnet', 'factnetv1', 'factnetdefault', 'vggfact', 'vggbnfact'], help="which convmodule to use")
parser.add_argument('--spatial_init', type=str, default='V1', choices=['default', 'V1'], 
                    help="initialization for spatial filters for LearnableCov models")
parser.add_argument('--name', type=str, default='TESTING_VGG', 
                        help='filename for saved model')
parser.add_argument('--affine', type=lambda x: bool(strtobool(x)), 
                        default=True, help='Batch Norm affine True or False')
parser.add_argument('--fact', type=lambda x: bool(strtobool(x)), 
                        default=True, help='FactNet True or False')
parser.add_argument('--width', default=1, type=float, help='width')
parser.add_argument('--sampling', type=str, default='rainbow',
        choices=['rainbow', 'random'], help="which sampling to use")
args = parser.parse_args()
if args.width == 1.0:
    args.width = 1
if args.width == 2.0:
    args.width = 2
if args.width == 4.0:
    args.width = 4


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
    root='./data', train=True, download=True, transform=transform_train)#transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
from ConvModules import FactConv2dPreExp

def replace_layers_agnostic(model, scale=1):
    prev_out_ch = 0
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_agnostic(module,scale)
        if isinstance(module, nn.Conv2d):
            if module.in_channels == 3:
                in_scale = 1 
            else:
                in_scale = scale
            ## simple module
            new_module = nn.Conv2d(
                    in_channels=int(module.in_channels*in_scale),
                    out_channels=int(module.out_channels*scale),
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    groups = module.groups,
                    bias=True if module.bias is not None else False)
            setattr(model, n, new_module)
            prev_out_ch = new_module.out_channels
        if isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(prev_out_ch)
            setattr(model, n, new_module)
        if isinstance(module, nn.Linear):
            new_module = nn.Linear(int(512 * scale), 10)
            setattr(model, n, new_module)

def replace_affines(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_affines(module)
        if isinstance(module, nn.BatchNorm2d):
            ## simple module
            new_module = nn.BatchNorm2d(
                    num_features=module.num_features,
                    eps=module.eps, momentum=module.momentum,
                    affine=False,
                    track_running_stats=module.track_running_stats)
            setattr(model, n, new_module)
def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module)
        if isinstance(module, nn.Conv2d):
            ## simple module
            new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False)
            setattr(model, n, new_module)


#net=ResNet18()#VGG("VGG11", args.bn_on)
net=Sequential_ThreeLayer_CIFAR10(100,False)
net=ResNet18()#VGG("VGG11", args.bn_on)
net.to(device)
replace_layers_agnostic(net, args.width)
if args.fact:
    replace_layers(net)
#replace_affines(net)
if not args.affine:
    replace_affines(net)
#sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/three-layer-models-sequential/TESTING_3Layer_final/{}_model.pt".format("fact"
#    if args.fact else "conv"))
if args.fact and args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/fact_model.pt".format(args.width))
elif not args.fact and args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_1/{}scale_final/conv_model.pt".format(args.width))
elif args.fact and not args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_2/{}scale_final/fact_model.pt".format(args.width))
elif not args.fact and not args.affine:
    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/affine_2/{}scale_final/conv_model.pt".format(args.width))
net.load_state_dict(sd)
net.to(device)
net_new = copy.deepcopy(net)
net_new.to(device)
#if args.fact:
#    replace_layers(net_new)


#replace_layers(net_new)

@torch.no_grad()
def recurse_layers(model, new_model):
    for (n1, m1), (n2, m2) in zip(model.named_children(), new_model.named_children()):
        if len(list(m1.children())) > 0:
            recurse_layers(m1, m2)
        if isinstance(m1, nn.Conv2d):
            if isinstance(m2, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
                    in_channels=m2.in_channels,
                    out_channels=m2.out_channels,
                    kernel_size=m2.kernel_size,
                    stride=m2.stride, padding=m2.padding, 
                    bias=True if m2.bias is not None else False)
            else:
                new_module = nn.Conv2d(
                    in_channels=int(m2.in_channels),
                    out_channels=int(m2.out_channels),
                    kernel_size=m2.kernel_size,
                    stride=m2.stride, padding=m2.padding, 
                    groups = m2.groups,
                    bias=True if m2.bias is not None else False).to(device)


            old_sd = m1.state_dict()
            new_sd = new_module.state_dict()
            og_copy = new_sd['weight']

            if m1.bias is not None:
                new_sd['bias'] = old_sd['bias']
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
 

            #old_weight = old_sd['weight']
            #u, s, v = torch.linalg.svd(old_weight.reshape(old_weight.shape[0], -1), full_matrices=False)
            #white_gaussian = torch.randn_like(u)
            #colored_gaussian = white_gaussian @ (s[..., None] * v)
            #new_sd['weight'] = colored_gaussian.reshape(old_weight.shape)

            copy_weight = old_sd['weight']
            copy_weight_gen = og_copy #new_sd['weight']
            copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
            copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
            weight_cov = (copy_weight_gen@copy_weight)##.T
            u, s, vh = torch.linalg.svd(
                   weight_cov, full_matrices=False
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
            alignment = u  @ vh  # (C_in_reference, C_in_generated)
            new_weight= og_copy#new_sd['weight'] #generated_model[j].weight
            new_weight = new_weight.reshape(new_weight.shape[0], -1)
            new_weight = new_weight@alignment # C_in_reference to C_in_generated
            new_weight = new_weight.reshape(old_sd['weight'].shape)
            new_module.register_buffer("weight_align", alignment)
            new_sd['weight_align'] = alignment
            # Set the new weights in the generated model.
            # NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
            # As such, if evaluated as is, its accuracy would be that of a random guess.
            new_sd['weight'] = new_weight

            if m1.in_channels != 3:
                activation = []
                other_activation = []

                hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                        activation.append(inputs[0].permute(0,2,3,1).reshape(-1,
                            inputs[0].shape[1])))
                hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                        other_activation.append(inputs[0].permute(0,2,3,1).reshape(-1,
                            inputs[0].shape[1])))

                covar = None
                total = 0
                print("starting covariance_calc")
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs1 = net(inputs)
                    outputs2 = net_new(inputs)
                    total+= inputs.shape[0]
                    if covar is None:
                        covar = activation[0].T @ other_activation[0]
                    else: 
                        covar += activation[0].T @ other_activation[0]
                    activation = []
                    other_activation = []
                covar /= total
                print("done with covariance_calc")
                hook_handle_1.remove()
                hook_handle_2.remove()
                u, s, v = torch.linalg.svd(covar, full_matrices=False)
                align = u @ v
                new_module.register_buffer("input_align", align)
                new_sd['input_align'] = align
                def return_hook():
                    def hook(mod, inputs):
                        shape = inputs[0].shape
                        inputs_permute = inputs[0].permute(1,0,2,3).reshape(inputs[0].shape[1], -1)
                        reshape = (mod.input_align@inputs_permute).reshape(shape[1],
                                shape[0], shape[2],
                                shape[3]).permute(1, 0, 2, 3)
                        return reshape 
                    return hook
                hook_handle_pre_forward = new_module.register_forward_pre_hook(return_hook())

                #else:
                #    new_weight = new_sd['weight']
                #    new_weight = torch.moveaxis(new_weight, source=1,
                #            destination=-1)
                #    print(align.shape)
                #    print(new_weight.shape)
                #    new_weight = new_weight@align
                #    new_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                #            destination=1)        
            #Set the new weights in the generated model.
            #NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
            #As such, if evaluated as is, its accuracy would be that of a random guess.
            new_module.load_state_dict(new_sd)
            setattr(new_model, n1, new_module)
        if isinstance(m1, nn.Linear):
            new_module = nn.Linear(m1.in_features, m1.out_features, bias=True
                    if m1.bias is not None else False).to(device)
            old_sd = m1.state_dict()
            new_sd = new_module.state_dict()
            new_sd['weight'] = old_sd['weight']
            if m1.bias is not None:
                new_sd['bias'] = old_sd['bias']
            activation = []
            other_activation = []

            hook_handle_1 = m1.register_forward_hook(lambda mod, inputs, outputs:
                    activation.append(inputs[0]))

            hook_handle_2 = m2.register_forward_hook(lambda mod, inputs, outputs:
                    other_activation.append(inputs[0]))
            covar = None
            total = 0
            print("starting covariance_calc")
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net(inputs)
                outputs2 = net_new(inputs)
                total+= inputs.shape[0]
                if covar is None:
                    covar = activation[0].T @ other_activation[0]
                else: 
                    covar += activation[0].T @ other_activation[0]
                activation = []
                other_activation = []
            covar /= total
            hook_handle_1.remove()
            hook_handle_2.remove()
            u, s, v = torch.linalg.svd(covar, full_matrices=False)
            align = u @ v
            new_weight = new_sd['weight']
            new_weight = torch.moveaxis(new_weight, source=1,
                    destination=-1)
            print(align.shape)
            print(new_weight.shape)
            new_weight = new_weight@align
            new_sd['weight'] = torch.moveaxis(new_weight, source=-1,
                    destination=1)        
            new_module.load_state_dict(new_sd)
            setattr(new_model, n1, new_module)
 
 
            print("linear")
        if isinstance(m1,nn.BatchNorm2d):
            print("BatchieNormie")
            m1.train()
            m2.train()
            m1.reset_running_stats()
            m2.reset_running_stats()
            m1.to(device)
            m2.to(device)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net(inputs)
                outputs2 = net_new(inputs)
            m1.eval()
            m2.eval()
 

        #    ## compound module, go inside it
        #    recurse_layers(module, new_model)
        #if isinstance(module, nn.Conv2d):
        #    ## simple module
        #    print("HI")
#
#if args.fact:
#    replace_layers(net)
#if args.fact:
#    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_scale/{}scale_final/fact_model.pt".format(args.width))
#else:
#    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_scale/{}scale_final/conv_model.pt".format(args.width))
##if args.bn_on and args.fact:
##    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_fixed/compare_vggbn_final/fact_model.pt")
##elif args.bn_on and not args.fact:
##    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_fixed/compare_vggbn_final/conv_model.pt")
##elif not args.bn_on and args.fact:
##    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_fixed/compare_vgg_final/fact_model.pt")
##elif not args.bn_on and not args.fact:
##    sd=torch.load("/network/scratch/v/vivian.white/v1-models/saved-models/comparing_fixed/compare_vgg_final/conv_model.pt")
#net.load_state_dict(sd)
#net.to(device)

#assert False
@torch.no_grad()
def rainbow_sampling(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            rainbow_sampling(module)
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, nn.Conv2d):
            ## simple module
            if isinstance(module, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            else:
                new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            print("old sd:",torch.norm(old_sd['weight']))
            print("new sd:",torch.norm(new_sd['weight']))
            #new_sd['weight'] *= torch.norm(old_sd['weight'])/torch.norm(new_sd['weight'])
            print("new sd:",torch.norm(new_sd['weight']))
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            copy_weight = old_sd['weight']
            copy_weight_gen = new_sd['weight']
            copy_weight = copy_weight.reshape(copy_weight.shape[0], -1)
            copy_weight_gen = copy_weight_gen.reshape(copy_weight_gen.shape[0], -1).T
            weight_cov = (copy_weight_gen@copy_weight)##.T
            u, s, vh = torch.linalg.svd(
                   weight_cov, full_matrices=False
            )  # (C_in_reference, R), (R,), (R, C_in_generated)
            alignment = u  @ vh  # (C_in_reference, C_in_generated)
            new_weight= new_sd['weight'] #generated_model[j].weight
            new_weight = new_weight.reshape(new_weight.shape[0], -1)
            new_weight = new_weight@alignment # C_in_reference to C_in_generated
            new_weight = new_weight.reshape(module.weight.shape)
          # Set the new weights in the generated model.
          # NOTE: this an intermediate model, as sampling the j-th layer means that the j+1-th layer is no longer aligned.
          # As such, if evaluated as is, its accuracy would be that of a random guess.
            new_sd['weight'] = new_weight
            #new_sd['weight'] *= torch.norm(old_sd['weight'])/torch.norm(new_sd['weight'])
            new_module.load_state_dict(new_sd)
            print("new sd:", torch.norm(new_sd['weight']))
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            print()
            setattr(model, n, new_module)

@torch.no_grad()
def random_sampling(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            random_sampling(module)
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
        if isinstance(module, nn.Conv2d):
            ## simple module
            if isinstance(module, FactConv2dPreExp):
                new_module = FactConv2dPreExp(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)
            else:
                new_module = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    bias=True if module.bias is not None else False,
                    groups=module.groups).to(device)

            old_sd = module.state_dict()
            new_sd = new_module.state_dict()
            print(torch.mean(torch.pow(old_sd['weight']-new_sd['weight'], 2)))
            print()
            if "tri1_vec" in old_sd.keys():
                new_sd['tri1_vec']=old_sd['tri1_vec']
                new_sd['tri2_vec']=old_sd['tri2_vec']
            if module.bias is not None:
                new_sd['bias'] = old_sd['bias']
            new_module.load_state_dict(new_sd)
            setattr(model, n, new_module)

def turn_off_grads(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            turn_off_grads(module)
        else:
            if isinstance(module, nn.Linear) and module.out_features == 10:
                grad=True
            else:
                grad=False
            for param in module.parameters():
                param.requires_grad = grad

set_seeds(1)
#net=VGG("VGG11", args.bn_on)
criterion = nn.CrossEntropyLoss()
def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net):
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    print("accuracy:", acc)
    return acc, test_loss
    #run.log({"accuracy":acc})

print("testing Res{}Net18 with width of {}".format("Fact" if args.fact else "Conv", args.width))
pretrained_acc, og_loss = test(0, net)

set_seeds(1)
s=time.time()
if args.sampling == "rainbow":
    rainbow_sampling(net_new)
    net_new.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_new(inputs)
else: 
    recurse_layers(net, net_new)

run_name= "{}_Res{}Net18_Width_{}_{}_affine".format(args.sampling.capitalize(), "Fact"
        if args.fact else "Conv", str(args.width), "No" if not args.affine else
        "Yes")


print("TOTAL TIME:", time.time()-s)
turn_off_grads(net_new)
##
##
## 
#
optimizer = optim.SGD(filter(lambda param: param.requires_grad, net_new.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#print("testing rainbow sampling")
print("testing {} sampling at width {}".format(args.sampling, args.width))
net_new.eval()

run_name= "correctly_saved_network_{}_Res{}Net18_Width_{}_{}_affine".format(args.sampling.capitalize(), "Fact"
        if args.fact else "Conv", str(args.width), "No" if not args.affine else
        "Yes")

args.name = run_name
save_model(args, net_new)
sampled_acc, sampled_loss = test(0, net_new)
#print("training rainbow sampling classifier head for 10 epochs")
accs = []
test_losses= []
print("training classifier head of {} sampled model for {} epochs".format(args.sampling, args.epochs))
for i in range(0, args.epochs):
    net_new.train()
    train(i, net_new)
    net_new.eval()
    acc, loss_test =test(i, net_new)
    test_losses.append(loss_test)
    accs.append(acc)
logger ={"pretrained_acc": pretrained_acc, "sampled_acc": sampled_acc,
        "first_epoch_acc":accs[0], "third_epoch_acc": accs[2],
        "tenth_epoch_acc":accs[args.epochs-1], 'width':args.width,
        "og_loss":og_loss, "sampled_loss":sampled_loss,
        "first_epoch_loss":test_losses[0], "third_epoch_loss": test_losses[2],
        "tenth_epoch_loss":test_losses[args.epochs-1], 'width':args.width}


wandb_dir = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/wandb"
os.makedirs(wandb_dir, exist_ok=True)
os.chdir(wandb_dir)
run = wandb.init(project="random_project", config=args,
        group="correct_saving_align_sota_covariance_rainbow_sampling", name=run_name, dir=wandb_dir)
run.log(logger)


args.name += "_trained_classifier_head"
save_model(args, net_new)
#wandb.watch(net, log='all', log_freq=1)
