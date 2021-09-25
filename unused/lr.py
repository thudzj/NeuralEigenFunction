import os
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='MLP on MNIST')
parser.add_argument('--data-path', type=str,
                    default='/Users/dengzhijie/Desktop/automl-one/automl-one/data') # '/data/LargeData/Regular')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', default=16, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=1e-3, type=float)


parser.add_argument('--num-layers', default=1, type=int)
parser.add_argument('--hidden-size', default=256, type=int)
parser.add_argument('--num-classes', default=5, type=int)


parser.add_argument('--num-pcs', default=4, type=int)
parser.add_argument('--eigengame-lr', default=0.1, type=float)
parser.add_argument('--eigengame-lr-schedule', default=None, type=str)
parser.add_argument('--eigengame-momentum', default=0.9999, type=float)
parser.add_argument('--riemannian-projection', action='store_true')
parser.add_argument('--num-steps-per-ite', default=1, type=int)

def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = load_mnist(args)
    
    # for eigengame_lr in [0.1, 0.01, 0.001, 0.0001]:
    #     for eigengame_momentum in [0.9, 0.99, 0.999, 0.9999]:
    #         for eigengame_lr_schedule in ['cos', None]:
    #             for riemannian_projection in [True, False]:
    #                 for num_steps_per_ite in [1, 5, 25]:

    model = MyModel(args, bias=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.decay)

    V = torch.ones(model.vectorized_weight.shape[0], args.num_pcs).to(device)
    V.div_(V.norm(dim=0))
    eigen_game = ParallelEigengame(V, args)

    # Train the model
    weights = []
    for epoch in range(args.epochs):
        losses = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = model.criterion(outputs, labels)
            losses += loss.item() * images.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            weights.append(model.vectorized_weight)
            for _ in range(args.num_steps_per_ite):
                eigen_game.step(model.vectorized_weight.unsqueeze(0))
        
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, 
                                                    losses / len(train_loader.dataset)))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = model.predict(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('     Accuracy of the model on the {} test images: {} %'.format(total, 100 * float(correct) / total))

    # weights = torch.stack(weights)

    # print("\n Eigenvalues calculate using the Eigengame are :\n", eigen_game.eigvals)
    # print("\n Eigenvalues calculate using the Eigengame are :\n", eigen_game.exact_eigvals(weights))
    # print("\n Eigenvectors calculated using the Eigengame are :\n", eigen_game.V)


    # p,q = torch.symeig(weights.T @ weights, eigenvectors=True)
    # print("\n Eigenvalues calculated using numpy are :\n",p[range(-1, -(args.num_pcs+1), -1)])
    # print("\n Eigenvectors calculated using numpy are :\n",q[:, range(-1, -(args.num_pcs+1), -1)])


    # print("\n Squared error in estimation of eigenvectors as compared to numpy :\n")
    # print(args.eigengame_lr, args.eigengame_momentum, args.eigengame_lr_schedule, args.riemannian_projection, args.num_steps_per_ite, torch.stack([q[:, -(i+1)] @ eigen_game.V[:, i] for i in range(args.num_pcs)]).data.numpy())
    print(V.T @ V)


class ParallelEigengame(object):
    def __init__(self, V, args):
        super(ParallelEigengame, self).__init__()
        self.lr = args.eigengame_lr
        self.momentum = args.eigengame_momentum
        self.num_ites = args.num_ites
        self.lr_schedule = args.eigengame_lr_schedule
        self.riemannian_projection = args.riemannian_projection

        self.V = V

        self.acc_a = None
        self.acc_XtXV = None

        self.ite = 0

    def step(self, X):
        lr = self.lr
        if self.num_ites is not None and self.lr_schedule is not None:
            if self.lr_schedule == 'cos':
                lr = self.lr * (1 + math.cos(math.pi * self.ite / self.num_ites)) / 2.

        XV = X @ self.V
        if self.acc_a is None:
            self.acc_a = XV.T @ XV
        else:
            self.acc_a.mul_(self.momentum).add_(XV.T @ XV, alpha = 1 - self.momentum)
        
        a = self.acc_a / (self.acc_a.diag()[None, :])
        a = torch.eye(a.shape[0], device=a.device) - a.tril(diagonal=-1)

        if self.acc_XtXV is None:
            self.acc_XtXV = X.T @ XV
        else:
            self.acc_XtXV.mul_(self.momentum).add_(X.T @ XV, alpha = 1 - self.momentum)

        g = self.acc_XtXV @ a.T

        if self.riemannian_projection:
            g.sub_((self.V*g).sum(0) * self.V)

        self.V.add_(g, alpha=lr)
        self.V.div_(self.V.norm(dim=0))

        self.ite += 1

    @property
    def eigvals(self):
        return self.acc_a.diag()

    def exact_eigvals(self, X):
        XV = X @ self.V
        return (XV.T @ XV).diag()

class MyModel(nn.Module):
    def __init__(self, args, bias=False):
        super(MyModel, self).__init__()
        self.input_size = args.input_size
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.bias = bias
        
        if self.num_layers == 1:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 1 if self.num_classes == 2 else self.num_classes, bias=bias))
        else:
            layers = [nn.Linear(self.input_size, self.hidden_size, bias=bias),
                      nn.ReLU(),
                      nn.Linear(self.hidden_size, 1 if self.num_classes == 2 else self.num_classes, bias=bias)]
            for _ in range(self.num_layers - 2):
                layers.insert(2, nn.ReLU())
                layers.insert(2, nn.Linear(self.hidden_size, self.hidden_size, bias=bias))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @property
    @torch.no_grad()
    def vectorized_weight(self):
        weight = []
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                if layer.weight is not None:
                    weight.append(layer.weight.data.view(-1))
                if layer.bias is not None:
                    weight.append(layer.bias.data.view(-1))
        return torch.cat(weight)

    def criterion(self, logits, targets):
        if self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())
        else:
            return F.cross_entropy(logits, targets)

    def predict(self, logits):
        if self.num_classes > 2:
            _, predicted = torch.max(logits, 1)
        else:
            predicted = (logits.squeeze() > 0.5).long()
        return predicted

def data_transform(x):
    return x.flatten().mul_(2).sub_(1)

def load_mnist(args):
    train_dataset = torchvision.datasets.MNIST(root=args.data_path,
                                               train=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(data_transform),
                                                ]),
                                               download=True)
    idx = (train_dataset.targets < args.num_classes)
    train_dataset.targets= train_dataset.targets[idx]
    train_dataset.data = train_dataset.data[idx]


    test_dataset = torchvision.datasets.MNIST(root=args.data_path,
                                              train=False,
                                              transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(data_transform),
                                                ])
                                              )
    idx = (test_dataset.targets < args.num_classes)
    test_dataset.targets= test_dataset.targets[idx]
    test_dataset.data = test_dataset.data[idx]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    args.input_size = 784
    args.num_ites = len(train_loader) * args.epochs

    return train_loader, test_loader

if __name__ == '__main__':
    main()
