import torch
import torch.optim
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from datetime import datetime
import os
from distutils.util import strtobool
import argparse
import control_models
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets

def save_model(args, model, loss, accuracy):
    src = "/research/harris/vivian/v1-models/saved-models/CIFAR100/"
    model_dir =  src + args.name
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = model_dir + "/trial_" + str(args.trial)
    if not os.path.exists(trial_dir): 
        os.makedirs(trial_dir)
    os.chdir(trial_dir)
    
    torch.save(loss, "loss.pt")
    torch.save(accuracy, "accuracy.pt")
    torch.save(model.state_dict(), "model.pt")
    torch.save(args, "args.pt")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')
    parser.add_argument('--width', type=int, default=2,help='width factor for resnet')
    parser.add_argument('--trial', type=int, default=1, help='num trial')
    parser.add_argument('--name', type=str, default='Scattering_Linear_Control', help='filesave name')
    parser.add_argument('--device', type=int, default=0, help='gpu: 0 or 1')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(32, 32), max_order=1)
        K = 17*3
    else:
        scattering = Scattering2D(J=2, shape=(32, 32))
        K = 81*3
    if use_cuda:
        scattering = scattering.to(device)
        
    model = control_models.Scattering_Linear_CIFAR100().to(device)

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    test_loss = []
    test_accuracy = []
    
    # Optimizer
    lr = 0.1
    for epoch in range(0, 90):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=0.0005)
            lr*=0.2

        control_models.train_scatter(model, device, train_loader, optimizer, epoch+1, scattering)
        loss, accuracy = control_models.test_scatter(model, device, test_loader, epoch+1, scattering)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        
    save_model(args, model, test_loss, test_accuracy) 