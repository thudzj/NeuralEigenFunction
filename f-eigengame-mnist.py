import math
from functools import partial
import itertools
from timeit import default_timer as timer

from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import nystrom, build_mlp_given_config, init_NN, load_mnist

parser = argparse.ArgumentParser(description='Decompose the ConvNet kernel on MNIST')
parser.add_argument('--data-path', type=str,
                    default='/Users/dengzhijie/Desktop/automl-one/automl-one/data') # '/data/LargeData/Regular')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 1000)')

parser.add_argument('--arch', default='convnet1', type=str)
parser.add_argument('--bhs-r', default=16, type=int, help='base hidden size for random NNs')
parser.add_argument('--w-var-r', default=2., type=float, help='w_var for random NNs')
parser.add_argument('--b-var-r', default=0.01, type=float, help='b_var for random NNs')

parser.add_argument('--bhs', default=64, type=int, help='base hidden size for eigenfuncs')
parser.add_argument('--k', default=3, type=int)
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--optimizer-type', default='Adam', type=str)
parser.add_argument('--num-iterations', default=10000, type=int)
parser.add_argument('--num-samples', default=1000, type=int)
parser.add_argument('--riemannian-projection', action='store_true')
parser.add_argument('--max-grad-norm', default=10., type=float)
parser.add_argument('--momentum', default=0.9, type=float)
# parser.add_argument('--decay', default=1e-3, type=float)

class ConvNet(nn.Module):
    def __init__(self, arch, bhs, input_size, output_size):
        super(ConvNet, self).__init__()
        self.arch = arch
        self.bhs = bhs
        self.input_size = input_size
        self.output_size = output_size

        if self.arch == 'convnet1':
            self.model = torch.nn.Sequential(
    			nn.Conv2d(in_channels=input_size[0], out_channels=bhs, kernel_size=3, padding=1),
    			nn.BatchNorm2d(bhs),
                nn.ReLU(), nn.MaxPool2d(2),
    			nn.Conv2d(in_channels=bhs, out_channels=bhs*2, kernel_size=3),
    			nn.BatchNorm2d(bhs*2),
                nn.ReLU(), nn.MaxPool2d(2),
    			nn.Flatten(1),
    			nn.Linear(bhs*2*6*6, bhs*4), nn.ReLU(),
    			nn.Linear(bhs*4, output_size)
    		)
        else:
            raise NotImplementedError

    def forward(self, x):
        if len(x.shape[1:]) != len(self.input_size):
            x = x.view(-1, *self.input_size)
        return self.model(x)

class NeuralEigenFunctions(nn.Module):
    def __init__(self, k, arch, bhs, input_size, output_size=1):
        super(NeuralEigenFunctions, self).__init__()
        self.functions = nn.ModuleList()
        for i in range(k):
            function = ConvNet(arch, bhs, input_size, output_size)
            self.functions.append(function)

    def forward(self, x):
        return F.normalize(torch.cat([f(x) for f in self.functions], 1), dim=0)*math.sqrt(x.shape[0])

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
    X, Y = [], []
    for x, y in train_loader:
        X.append(x); Y.append(y)
    X, Y = torch.cat(X).to(device), torch.cat(Y).to(device)
    print(X.shape, Y.shape, X.max(), X.min())

    random_model = ConvNet(args.arch, args.bhs_r, input_size=[1, 28, 28], output_size=1).to(device)
    random_model.eval()
    samples = []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(args.num_samples):
                if _ % 50 == 0:
                    print("Have obtained {} samples of the ConvNet kernel".format(_))
                init_NN(random_model, args.w_var_r, args.b_var_r)
                samples.append(random_model(X))
    samples = torch.cat(samples, -1).T.float()

    start = timer()
    nef = NeuralEigenFunctions(args.k, args.arch, args.bhs, input_size=[1, 28, 28]).to(device)
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(nef.parameters(), lr=args.lr)
    elif args.optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(nef.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(nef.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

    eigenvalues_our = None
    for ite in range(args.num_iterations):
        idx = np.random.choice(X.shape[0], args.bs, replace=False)
        samples_batch = samples[:, idx]
        X_batch = X[idx]

        psis_X = nef(X_batch)
        with torch.no_grad():
            samples_batch_psis = samples_batch @ psis_X
            psis_K_psis = samples_batch_psis.T @ samples_batch_psis / args.num_samples
            mask = torch.eye(args.k, device=psis_X.device) - (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
            grad = samples_batch.T @ (samples_batch_psis @ mask / args.num_samples)

            if eigenvalues_our is None:
                eigenvalues_our = psis_K_psis.diag() / (args.bs**2)
            else:
                eigenvalues_our.mul_(0.9).add_(psis_K_psis.diag() / (args.bs**2), alpha = 0.1)

            if args.riemannian_projection:
                grad.sub_((psis_X*grad).sum(0) * psis_X)

            if ite % 50 == 0:
                print(ite, grad.norm(dim=0))
            
            clip_coef = args.max_grad_norm / (grad.norm(dim=0) + 1e-6)
            grad.mul_(clip_coef)

        optimizer.zero_grad()
        psis_X.backward(-grad)
        optimizer.step()
        # scheduler.step()
    end = timer()
    print("Our method consumes {}s".format(end - start))
    print(eigenvalues_our)

if __name__ == '__main__':
    main()
