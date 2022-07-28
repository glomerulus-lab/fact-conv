from os import getcwd, makedirs
from os.path import abspath, join, exists

import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/whitev4/research/structured_random_features/')
from src.models.networks import V1_mnist_RFNet, classical_RFNet
from src.data.load_dataset import load_mnist

data_dir = abspath(join(getcwd(), '../../'))
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# load dataset and move to GPU
mnist_dir = data_dir + '/data/processed/'
train_set = datasets.MNIST(mnist_dir, train=True, download=True)
test_set = datasets.MNIST(mnist_dir, train=False, download=True)
train, y_train = train_set.data.float().to(device), train_set.targets.to(device)
test, y_test = test_set.data.float().to(device), test_set.targets.to(device)

# normalize and reshape
X_train = (train - train.mean()) / train.std()
X_test = (test - train.mean()) / train.std()
X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

# training params
scale = 2/784 # since we do a cholesky before generating weights
num_epochs = 20
num_trials = 1
loss_fn = F.cross_entropy

# params to iterate over
hidden_size_list = [50, 100, 400, 1000, 2000, 5000]
lr_list = [1E-1, 1E-2, 1E-3]

# V1 params
compatible = {'s': 5, 'f':2}
incompatible = {'s': 0.5, 'f':0.5}

s, f = compatible['s'], compatible['f']
v1_train_loss = {h: {lr: {'mean': [], 'std': []} for lr in lr_list} for h in hidden_size_list}
v1_test_accuracy = {h: {lr: {'mean': [], 'std': []} for lr in lr_list} for h in hidden_size_list}

for h in hidden_size_list:
    for lr in lr_list:
        train_loss = np.zeros((num_trials, num_epochs))
        test_accuracy = np.zeros((num_trials, num_epochs))
        for trial in range(num_trials):
            # declare model and optimizer
            model = V1_mnist_RFNet(h, s, f, scale=scale, center=None).to(device)
            model.v1_layer.weight.requires_grad = True
            optimizer = optim.SGD(model.parameters(), lr=lr)
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                output = model(X_train)
                loss = loss_fn(output, y_train)
                loss.backward()
                optimizer.step()
                train_loss[trial, epoch] = loss.item()

                # accuracy
                pred = torch.argmax(model(X_test), axis=1)
                test_accuracy[trial, epoch] = torch.sum(pred == y_test) / len(X_test)

                if epoch % 1000 == 0:
                    print('Accuracy: {:.6f}'.format(test_accuracy[trial, epoch]))
    
        # train error
        v1_train_loss[h][lr]['mean'] = np.mean(train_loss, axis=0)
        v1_train_loss[h][lr]['std'] = np.std(train_loss, axis=0) / np.sqrt(num_trials)
        # test error
        v1_test_accuracy[h][lr]['mean'] = np.mean(test_accuracy, axis=0)
        v1_test_accuracy[h][lr]['std'] = np.std(test_accuracy, axis=0) / np.sqrt(num_trials)


