#Model w/ weights initialized according to V1 receptive fields

import torch
import torch.optim
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from datetime import datetime
import os
from distutils.util import strtobool
import argparse
from numpy.random import RandomState
import numpy as np
import V1_models
import LC_models
from test_models_safety import PostExp, PreExp
import numpy as np
import random
   
    
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)



def save_model(args, model, loss, accuracy):
    src  = "/home/mila/m/muawiz.chaudhary/scratch/v1-models/saved-models/CIFAR10_50_Samples/"
    model_dir =  src + args.name
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(args.trial)
    os.makedirs(trial_dir, exist_ok=True)
    os.chdir(trial_dir)
    
    torch.save(loss, trial_dir+"/loss.pt")
    torch.save(accuracy, trial_dir+ "/accuracy.pt")
    torch.save(model.state_dict(), trial_dir+ "/model.pt")
    torch.save(args, trial_dir+ "/args.pt")

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='samples per class')
    parser.add_argument('--learning_schedule_multi', type=int, default=10,
                        help='samples per class')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for dataset subselection')
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--name', type=str, default='BN_V1_V1_Linear', 
                        help='filename for saved model')
    parser.add_argument('--trial', type=int, default=1, help='trial number')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), 
                        default=False, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, 
                        help="which device to use (0 or 1)")

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
    #parser.add_argument('--seed', type=int, default=0, help='seed')
 
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

    set_seeds(args.seed)
    print(model)
    print("Num params total: ", sum(p.numel() for p in model.parameters()))
    print("Num params grad: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
     # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    #####cifar data
    cifar_data = datasets.CIFAR10(
            root=scattering_datasets.get_dataset_dir('CIFAR'), 
            train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    # Extract a subset of X samples per class
    prng = RandomState(args.seed)
    random_permute = prng.permutation(np.arange(0, 5000))[0:args.num_samples]
    indx = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute] for classe in range(0, 10)])

    cifar_data.data, cifar_data.targets = cifar_data.data[indx], list(np.array(cifar_data.targets)[indx])
    train_loader = torch.utils.data.DataLoader(cifar_data,
                                               batch_size=32, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=scattering_datasets.get_dataset_dir('CIFAR'), 
            train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    test_loss = []
    test_accuracy = []
    
    # Optimizer
    for epoch in range(0, 90):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005, nesterov=True)
            args.lr*=0.2

        V1_models.train(model, device, train_loader, optimizer, epoch+1)
        loss, accuracy = V1_models.test(model, device, test_loader, epoch+1)
        test_loss.append(loss)
        test_accuracy.append(accuracy)   
            
    end = datetime.now()
    print("Trial {} time (HH:MM:SS): {}".format(args.trial, end-start))
    print("Learning rate: {}".format(initial_lr))
    
    save_model(args, model, test_loss, test_accuracy) 

