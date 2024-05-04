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


def save_model(args, model, loss, accuracy):
    src = "/research/harris/vivian/v1-models/saved-models/MNIST/"
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
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions in model')
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=2, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1, help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    parser.add_argument('--name', type=str, default='BN_V1_V1_Linear', help='filename for saved model')
    parser.add_argument('--trial', type=int, default=1, help='trial number')
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), default=True, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    start = datetime.now()

    model = V1_models.V1_MNIST(args.hidden_dim, args.s, args.f, args.scale, args.bias).to(device)

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=scattering_datasets.get_dataset_dir('MNIST'), train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=scattering_datasets.get_dataset_dir('MNIST'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    test_loss = []
    test_accuracy = []

    for epoch in range(0, args.num_epoch):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005, nesterov=True)
            args.lr*=0.2

        V1_models.train(model, device, train_loader, optimizer, epoch+1)
        loss, accuracy = V1_models.test(model, device, test_loader, epoch+1)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    end = datetime.now()
    print("Trial {} time (HH:MM:SS): {}".format(args.trial, end-start))
    print("Hidden dim: {}\t Learning rate: {}".format(args.hidden_dim, initial_lr))
    
    save_model(args, model, test_loss, test_accuracy)    
    