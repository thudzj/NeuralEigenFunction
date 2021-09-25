'''
CUDA_VISIBLE_DEVICES=5 python f-eigengame-mnist.py  --data-path /data/LargeData/Regular/ --k 10 --num-samples 2000 --b-var-r 0.01 --job-id 2  --bhs-r 16 16 16 --bhs 32 64 128 --arch convnet2
	Training acc of the l-svc for data projected by pca: 0.7529833333333333
	Testing acc of the l-svc for data projected by pca: 0.7639
	Training acc of the lr for data projected by pca: 0.7377
	Testing acc of the lr for data projected by pca: 0.7433

	Training acc of the linear svc: 0.8076166666666666
	Testing acc of the linear svc: 0.8212
	Training acc of the lr: 0.8063166666666667
	Testing acc of the lr: 0.8235


run on g20
CUDA_VISIBLE_DEVICES=6 python f-eigengame-mnist.py  --data-path /data/LargeData/Regular --k 10 --num-samples 2000 --b-var-r 0.01 --job-id 6  --bhs-r 16 16 16 --bhs 32 64 128 --arch convnet1
	Training acc of the linear svc: 0.7791166666666667
	Testing acc of the linear svc: 0.7846
	Training acc of the lr: 0.7697666666666667
	Testing acc of the lr: 0.776

	Training acc of the l-svc for data projected by nystrom: 0.77825
	Testing acc of the l-svc for data projected by nystrom: 0.7886
	Training acc of the lr for data projected by nystrom: 0.7687
	Testing acc of the lr for data projected by nystrom: 0.7781

	Training acc of the l-svc for data projected by pca: 0.6986166666666667
	Testing acc of the l-svc for data projected by pca: 0.699
	Training acc of the lr for data projected by pca: 0.73265
	Testing acc of the lr for data projected by pca: 0.7442
'''
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

from sklearn import svm
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import SGDClassifier
from utils import nystrom, init_NN, load_mnist, ConvNet, ConvNetKernel

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
parser.add_argument('--bhs-r', default=[16, 16, 16], type=int, nargs='+', help='base hidden size for random NNs')
parser.add_argument('--w-var-r', default=2., type=float, help='w_var for random NNs')
parser.add_argument('--b-var-r', default=0.01, type=float, help='b_var for random NNs')

parser.add_argument('--bhs', default=[16, 32, 64], type=int, nargs='+', help='base hidden size for eigenfuncs')
parser.add_argument('--k', default=3, type=int)
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--optimizer-type', default='Adam', type=str)
parser.add_argument('--num-iterations', default=20000, type=int)
parser.add_argument('--num-samples', default=4000, type=int)
parser.add_argument('--riemannian-projection', action='store_true')
parser.add_argument('--max-grad-norm', default=None, type=float)
parser.add_argument('--momentum', default=0.9, type=float)

parser.add_argument('--job-id', default='', type=str)

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, arch, bhs, input_size, output_size=1, momentum=0.9, normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.functions = nn.ModuleList()
		for i in range(k):
			function = ConvNet(arch, bhs, input_size, output_size)
			self.functions.append(function)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = torch.cat([f(x) for f in self.functions], 1)
		if self.training:
			norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
			with torch.no_grad():
				if self.num_calls == 0:
					self.eigennorm.copy_(norm_.data)
				else:
					self.eigennorm.mul_(self.momentum).add_(norm_.data, alpha = 1-self.momentum)
				self.num_calls += 1
		else:
			norm_ = self.eigennorm
		return ret_raw / norm_

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
	X, Y = torch.cat(X).to(device), torch.cat(Y)
	# print(X.shape, Y.shape, X.max(), X.min())

	X_val, Y_val = [], []
	for x, y in test_loader:
		X_val.append(x); Y_val.append(y)
	X_val, Y_val = torch.cat(X_val).to(device), torch.cat(Y_val)


	# perform nystrom method on the nngp-kernel
	if args.arch == 'convnet1':
		kernel = ConvNetKernel('convnet1', [1, 28, 28], args.w_var_r, args.b_var_r).cuda()
		with torch.no_grad():
			_, eigenfuncs_nystrom, _ = nystrom(X[np.random.choice(X.shape[0], 800, replace=False)].contiguous(), args.k, kernel)
			X_projected_by_nystrom, X_val_projected_by_nystrom = [], []
			with torch.cuda.amp.autocast():
				for i in range(0, len(X), args.bs):
					X_projected_by_nystrom.append(eigenfuncs_nystrom(X[i: min(len(X), i+args.bs)]).cpu())
				for i in range(0, len(X_val), args.bs):
					X_val_projected_by_nystrom.append(eigenfuncs_nystrom(X_val[i: min(len(X_val), i+args.bs)]).cpu())
			X_projected_by_nystrom = torch.cat(X_projected_by_nystrom).float()
			X_val_projected_by_nystrom = torch.cat(X_val_projected_by_nystrom).float()
		clf = svm.LinearSVC()
		clf.fit(X_projected_by_nystrom, Y)
		print("Training acc of the l-svc for data projected by nystrom: {}".format(clf.score(X_projected_by_nystrom, Y)))
		print("Testing acc of the l-svc for data projected by nystrom: {}".format(clf.score(X_val_projected_by_nystrom, Y_val)))
		clf = SGDClassifier(loss='log')
		clf.fit(X_projected_by_nystrom, Y)
		print("Training acc of the lr for data projected by nystrom: {}".format(clf.score(X_projected_by_nystrom, Y)))
		print("Testing acc of the lr for data projected by nystrom: {}".format(clf.score(X_val_projected_by_nystrom, Y_val)))

	with torch.no_grad():
		p, q = torch.symeig(X.T @ X, eigenvectors=True)
		X_projected_by_pca = (X @ q[:, range(-1, -(args.k+1), -1)]).data.cpu()
		X_val_projected_by_pca = (X_val @ q[:, range(-1, -(args.k+1), -1)]).data.cpu()
	clf = svm.LinearSVC()
	clf.fit(X_projected_by_pca, Y)
	print("Training acc of the l-svc for data projected by pca: {}".format(clf.score(X_projected_by_pca, Y)))
	print("Testing acc of the l-svc for data projected by pca: {}".format(clf.score(X_val_projected_by_pca, Y_val)))
	clf = SGDClassifier(loss='log')
	clf.fit(X_projected_by_pca, Y)
	print("Training acc of the lr for data projected by pca: {}".format(clf.score(X_projected_by_pca, Y)))
	print("Testing acc of the lr for data projected by pca: {}".format(clf.score(X_val_projected_by_pca, Y_val)))

	# perform our method
	random_model = ConvNet(args.arch, args.bhs_r, input_size=[1, 28, 28], output_size=1).to(device)
	num_params = sum(p.numel() for p in random_model.parameters())
	print("Number of parameters:", num_params)
	random_model.eval()
	samples = []
	with torch.no_grad():
		with torch.cuda.amp.autocast(False):
			for _ in range(args.num_samples):
				if _ % 50 == 0:
					print("Have obtained {} samples of the ConvNet kernel".format(_))
				init_NN(random_model, args.w_var_r, args.b_var_r)
				samples.append(random_model(X).data.cpu())
	samples = torch.cat(samples, -1).T

	start = timer()
	nef = NeuralEigenFunctions(args.k, args.arch, args.bhs, input_size=[1, 28, 28]).to(device)
	if args.optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=args.lr)
	elif args.optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=args.lr, momentum=args.momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=args.lr, momentum=args.momentum)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_iterations)

	eigenvalues_our = None
	nef.train()
	for ite in range(args.num_iterations):
		idx = np.random.choice(X.shape[0], args.bs, replace=False)
		samples_batch = samples[:, idx].to(device)
		psis_X = nef(X[idx])
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
				grad.sub_((psis_X*grad).sum(0) * psis_X / args.bs)

			if ite % 50 == 0:
				print(ite, grad.norm(dim=0))
			if args.max_grad_norm is not None:
				clip_coef = args.max_grad_norm / (grad.norm(dim=0) + 1e-6)
				grad.mul_(clip_coef)

		optimizer.zero_grad()
		psis_X.backward(-grad)
		optimizer.step()
		scheduler.step()
	end = timer()
	print("Our method consumes {}s".format(end - start))
	print(eigenvalues_our)

	# dimension reduction
	nef.eval()
	X_projected_by_our, X_val_projected_by_our = [], []
	with torch.no_grad():
		with torch.cuda.amp.autocast():
			for i in range(0, len(X), args.bs):
				X_projected_by_our.append(nef(X[i: min(len(X), i+args.bs)]).data.cpu())
			for i in range(0, len(X_val), args.bs):
				X_val_projected_by_our.append(nef(X_val[i: min(len(X_val), i+args.bs)]).data.cpu())
		X_projected_by_our = torch.cat(X_projected_by_our).float()
		X_val_projected_by_our = torch.cat(X_val_projected_by_our).float()
		print(X_projected_by_our.shape, X_projected_by_our[: 5], X_val_projected_by_our.shape)

	# visualization
	idx = np.random.choice(X.shape[0], 1000, replace=False)
	colors = [plt.cm.tab10(i) for i in range(10)]
	cmap=matplotlib.colors.ListedColormap(colors)
	figure = plt.figure(figsize=(5, 5))
	ax = figure.add_subplot(111, projection='3d')
	# ax.set_title("Projected by our")
	ax.scatter3D(X_projected_by_our[idx, 0],
				 X_projected_by_our[idx, 1],
				 X_projected_by_our[idx, 2],
				 c=Y[idx], cmap=cmap, edgecolors='k')
	ax.grid(True)
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)
	plt.setp( ax.get_zticklabels(), visible=False)

	figure.tight_layout()
	figure.savefig('mnist_plots/{}_3d.pdf'.format(args.job_id), format='pdf', dpi=1000, bbox_inches='tight')

	# eigenvalues_our = eigenvalues_our.cpu()
	clf = svm.LinearSVC()
	clf.fit(X_projected_by_our, Y)
	print("Training acc of the linear svc: {}".format(clf.score(X_projected_by_our, Y)))
	print("Testing acc of the linear svc: {}".format(clf.score(X_val_projected_by_our, Y_val)))

	clf = SGDClassifier(loss='log')
	clf.fit(X_projected_by_our, Y)
	print("Training acc of the lr: {}".format(clf.score(X_projected_by_our, Y)))
	print("Testing acc of the lr: {}".format(clf.score(X_val_projected_by_our, Y_val)))

	# this is too slow
	# kernel = DotProduct() + WhiteKernel()
	# gpc = GaussianProcessClassifier(kernel=kernel, random_state=args.seed).fit(X_projected_by_our*eigenvalues_our.sqrt(), Y)
	# print("Training acc of the linear gpc: {}".format(gpr.score(X_projected_by_our*eigenvalues_our.sqrt(), Y)))
	# print("Testing acc of the linear gpc: {}".format(gpr.score(X_val_projected_by_our*eigenvalues_our.sqrt(), Y_val)))

if __name__ == '__main__':
	main()
