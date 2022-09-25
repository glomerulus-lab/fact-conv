#Control model with uniform weight distribution on conv2d layers
import torch
import torch.optim
from torchvision import datasets, transforms
import kymatio.datasets as scattering_datasets
from datetime import datetime
import os
from distutils.util import strtobool
import argparse
import control_models

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden dimensions in model')
    parser.add_argument('--num_epoch', type=int, default=90, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--name', type=str, default="Uniform_Control", help="file name")
    parser.add_argument('--trial', type=int, default=1, help="trial #")
    parser.add_argument('--bias', dest='bias', type=lambda x: bool(strtobool(x)), default=True, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0, help="which device to use (0 or 1)")
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")


    model = control_models.Uniform_CIFAR100(args.hidden_dim, args.bias).to(device)
 
    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=scattering_datasets.get_dataset_dir('CIFAR100'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
 
    test_loss = []
    test_accuracy = []
    
    start = datetime.now()

    for epoch in range(0, args.num_epoch):
        if epoch%20==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005, nesterov=True)
            args.lr*=0.2

        control_models.train(model, device, train_loader, optimizer, epoch+1)
        loss, accuracy = control_models.test(model, device, test_loader, epoch+1)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
    
    end = datetime.now()
    print("Trial {} time (HH:MM:SS): {}".format(args.trial, end-start))
    print("Hidden dim: {}\t Learning rate: {}".format(args.hidden_dim, initial_lr))
    
    save_model(args, model, test_loss, test_accuracy) 
