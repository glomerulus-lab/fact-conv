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
    # src = "/research/harris/vivian/v1-models/saved-models/classification/CIFAR10/"
    # model_dir =  src + args.name
    model_dir = os.path.join(args.dir, args.name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    #os.chdir(model_dir)
    
    #saves loss & accuracy in the trial directory -- all trials
    trial_dir = os.path.join(model_dir, f"trial_{args.trial}")
    if not os.path.exists(trial_dir): 
        os.makedirs(trial_dir)
    os.chdir(trial_dir)
    
    torch.save(loss, "loss.pt")
    torch.save(accuracy, "accuracy.pt")
    torch.save(model.state_dict(), "model.pt")
    torch.save(args, "args.pt")


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='number of hidden dimensions in model')
    parser.add_argument('--num_epoch', type=int, default=90,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=1, help='V1 size')
    parser.add_argument('--f', type=float, default=0.1,
                        help='V1 spatial frequency')
    parser.add_argument('--scale', type=int, default=1, help='V1 scale')
    # TODO(kamdh): change to filename
    parser.add_argument('--name', type=str, default='BN_V1_V1_Linear',
                        help='subdirectory for this simulation')
    # TODO(kamdh): deprecate
    parser.add_argument('--trial', type=int, default=1, help='trial number')
    parser.add_argument('--bias', dest='bias',
                        type=lambda x: bool(strtobool(x)),
                        default=True, help='bias=True or False')
    parser.add_argument('--device', type=int, default=0,
                        help="which device to use (0 or 1)")
    parser.add_argument('--penalty', type=float, default=0.,
                        help="regularization term")
    # TODO(kamdh): deprecate
    parser.add_argument('--dir', type=str,
                        default="../saved-models/classification/CIFAR10",
                        help="directory to save in")
    args = parser.parse_args()
    initial_lr = args.lr

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    start = datetime.now()

    model = V1_models.BN_V1_V1_LinearLayer_CIFAR10(
        args.hidden_dim, args.s, args.f, args.scale, args.bias).to(device)
    model_init = V1_models.BN_V1_V1_LinearLayer_CIFAR10(
        args.hidden_dim, args.s, args.f, args.scale, args.bias).to(device)
    model_init.load_state_dict(model.state_dict())


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
        datasets.CIFAR10(
            root="/research/harris/vivian/v1-models/datasets/new_CIFAR10",
            #root="$SLURM_TMPDIR",
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
        batch_size=128, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="/research/harris/vivian/v1-models/datasets/new_CIFAR10",
            #root="$SLURM_TMPDIR",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=128, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)


    test_loss = []
    test_accuracy = []

    for epoch in range(0, args.num_epoch):
        if epoch % 20 == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=0.0005,
                                        nesterov=True)
            args.lr *= 0.2

        V1_models.train(model, model_init, args.penalty, device, train_loader, optimizer, epoch+1)
        loss, accuracy = V1_models.test(model, device, test_loader, epoch+1)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    end = datetime.now()
    print("Trial {} time (HH:MM:SS): {}".format(args.trial, end-start))
    print("Hidden dim: {}\t Learning rate: {}".format(args.hidden_dim, initial_lr))
    
    save_model(args, model, test_loss, test_accuracy)    
    
