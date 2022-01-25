import warnings
import math
import os
import time
from timeit import default_timer as timer
from functools import partial

import numpy as np
import scipy
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import cnn_gp
try:
	import jax
	import neural_tangents as nt
except:
	print("Jax and neural_tangents not found")

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
# sns.set(style="darkgrid")

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise NanError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(5):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		raise e


def polynomial_kernel(degree, eta, nu, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	x1 = x1.flatten(1)
	x2 = x2.flatten(1)
	return (x1 @ x2.T * eta + nu) ** degree

def sigmoid_kernel(eta, nu, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	x1 = x1.flatten(1)
	x2 = x2.flatten(1)
	return ((x1.unsqueeze(1) * x2.unsqueeze(0)).sum(-1) * eta + nu).tanh()

def cosine_kernel(period, output_scale, length_scale, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	x1 = x1.flatten(1)
	x2 = x2.flatten(1)
	return (((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1).sqrt() * math.pi / period / length_scale).cos() * output_scale

def rbf_kernel(output_scale, length_scale, x1, x2=None):
	if x2 is None:
		x2 = x1

	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)

	x1 = x1.flatten(1)
	x2 = x2.flatten(1)

	#
	# (x1 ** 2).sum(-1).view(-1, 1) + (x2 ** 2).sum(-1).view(1, -1) - 2 * x1 @ x2.T
	return (- ((x1 ** 2).sum(-1).view(-1, 1) + (x2 ** 2).sum(-1).view(1, -1) - 2 * x1 @ x2.T) / 2. / length_scale).exp() * output_scale

def linear_kernel(x1, x2=None):
	if x2 is None:
		x2 = x1

	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)

	x1 = x1.flatten(1)
	x2 = x2.flatten(1)

	return x1 @ x2.T

def periodic_plus_rbf_kernel(period, output_scale1, length_scale1, output_scale2, length_scale2, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)

	x1 = x1.flatten(1)
	x2 = x2.flatten(1)

	out1 = (- (((x1.unsqueeze(1) - x2.unsqueeze(0)).abs().sum(-1) * math.pi / period).sin() ** 2) * 2. / length_scale1).exp() * output_scale1
	out2 = (- ((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1) / 2. / length_scale2).exp() * output_scale2
	return out1 + out2


def nystrom(X, k, kernel):
	start = timer()
	K = kernel(X)
	p, q = scipy.linalg.eigh(K.data.cpu().numpy(), subset_by_index=[K.shape[0]-k, K.shape[0]-1])
	p = torch.from_numpy(p).to(X.device).float()[range(-1, -(k+1), -1)]
	q = torch.from_numpy(q).to(X.device).float()[:, range(-1, -(k+1), -1)]
	# p, q = torch.symeig(K, eigenvectors=True)
	eigenvalues_nystrom = p / X.shape[0]
	eigenfuncs_nystrom = lambda x: kernel(x, X) @ q / p * math.sqrt(X.shape[0])
	end = timer()
	return eigenvalues_nystrom, eigenfuncs_nystrom, end - start


def oas(X):
	# shrinkage the covariance matrix
	n_samples, n_features = X.shape

	# emp_cov = X.T @ X / n_samples
	# tmp = torch.trace(emp_cov).item() / n_features

	mu = (X ** 2).mean().item()
	# assert np.isclose(mu, tmp), (mu, tmp)

	# formula from Chen et al.'s **implementation**
	# tmp = (emp_cov ** 2).mean().item()
	alpha = ((X @ X.T / n_features) ** 2).mean().item()

	# assert np.isclose(alpha, tmp), (alpha, tmp)

	num = alpha + mu ** 2
	den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

	shrinkage = 1. if den == 0 else min(num / den, 1.)
	return mu, shrinkage

class Erf(torch.nn.Module):
	def __init__(self):
		super(Erf, self).__init__()

	def forward(self, x):
		return x.erf()

class SinAndCos(torch.nn.Module):
	def __init__(self):
		super(SinAndCos, self).__init__()

	def forward(self, x):
		assert x.shape[1] % 2 == 0
		x1, x2 = x.chunk(2, dim=1)
		return torch.cat([torch.sin(x1), torch.cos(x2)], 1)

def build_mlp_given_config(**kwargs):
	if kwargs['nonlinearity'] == 'relu':
		nonlinearity=nn.ReLU
	elif 'lrelu' in kwargs['nonlinearity']:
		nonlinearity=partial(nn.LeakyReLU, float(kwargs['nonlinearity'].replace("lrelu", "")))
	elif kwargs['nonlinearity'] == 'erf':
		nonlinearity=Erf
	elif kwargs['nonlinearity'] == 'sin_and_cos':
		nonlinearity=SinAndCos
	else:
		raise NotImplementedError
	if kwargs['num_layers'] == 1:
		function = nn.Sequential(
			nn.Linear(kwargs['input_size'], kwargs['output_size'], bias=kwargs['bias']))
	else:
		layers = [nn.Linear(kwargs['input_size'], kwargs['hidden_size'], bias=kwargs['bias']),
				  nonlinearity(),
				  nn.Linear(kwargs['hidden_size'], kwargs['output_size'], bias=kwargs['bias'])]
		for _ in range(kwargs['num_layers'] - 2):
			layers.insert(2, nonlinearity())
			layers.insert(2, nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'], bias=kwargs['bias']))
		function = nn.Sequential(*layers)
	return function

def init_NN(model, w_var_list, b_var_list):
	if not isinstance(w_var_list, list):
		w_var_list = [w_var_list]
	if not isinstance(b_var_list, list):
		b_var_list = [b_var_list]
	i = 0
	for m in model.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			with torch.no_grad():
				fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
				m.weight.normal_(0, math.sqrt(w_var_list[i]/fan_in))
				if m.bias is not None:
					if math.sqrt(b_var_list[i]) > 0:
						m.bias.normal_(0, math.sqrt(b_var_list[i]))
					else:
						m.bias.fill_(0.)
				i += 1
				if i >= len(w_var_list):
					i = 0

		elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.running_mean, 0)
			nn.init.constant_(m.running_var, 1)

class ConvNet(nn.Module):
	def __init__(self, arch, hs, input_size, output_size):
		super(ConvNet, self).__init__()
		self.arch = arch
		self.input_size = input_size
		self.output_size = output_size

		if self.arch == 'convnet1':
			self.model = torch.nn.Sequential(
				nn.Conv2d(in_channels=input_size[0], out_channels=hs[0], kernel_size=3, stride=2, padding=1),
				nn.BatchNorm2d(hs[0]),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=hs[0], out_channels=hs[1], kernel_size=3, stride=2, padding=0),
				nn.BatchNorm2d(hs[1]),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=hs[1], out_channels=hs[2], kernel_size=6, stride=1, padding=0),
				nn.BatchNorm2d(hs[2]),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels=hs[2], out_channels=output_size, kernel_size=1, stride=1, padding=0),
				# nn.Conv2d(in_channels=input_size[0], out_channels=hs[0], kernel_size=1, stride=1, padding=0),
				# nn.ReLU(inplace=True),
				# nn.Conv2d(in_channels=hs[0], out_channels=hs[1], kernel_size=1, stride=1, padding=0),
				# nn.ReLU(inplace=True),
				# nn.Conv2d(in_channels=hs[1], out_channels=hs[2], kernel_size=1, stride=1, padding=0),
				# nn.ReLU(inplace=True),
				# nn.Conv2d(in_channels=hs[2], out_channels=output_size, kernel_size=1, stride=1, padding=0),
			)
		elif self.arch == 'convnet2':
			self.model = torch.nn.Sequential(
				nn.Conv2d(in_channels=input_size[0], out_channels=hs[0], kernel_size=5, padding=2),
				nn.BatchNorm2d(hs[0]),
				nn.ReLU(), nn.MaxPool2d(2),
				nn.Conv2d(in_channels=hs[0], out_channels=hs[1], kernel_size=5, padding=2),
				nn.BatchNorm2d(hs[1]),
				nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(1),
				nn.Linear(hs[1]*7*7, hs[2]), nn.ReLU(),
				nn.Linear(hs[2], output_size)
			)
		elif self.arch == 'convnet3':
			self.model = torch.nn.Sequential(
				nn.Conv2d(in_channels=input_size[0], out_channels=hs[0], kernel_size=3, padding=1),
				nn.BatchNorm2d(hs[0]),
				nn.ReLU(), nn.MaxPool2d(2),
				nn.Conv2d(in_channels=hs[0], out_channels=hs[1], kernel_size=3),
				nn.BatchNorm2d(hs[1]),
				nn.ReLU(), nn.MaxPool2d(2),
				nn.Conv2d(in_channels=hs[1], out_channels=hs[2], kernel_size=3),
				nn.BatchNorm2d(hs[2]),
				nn.ReLU(), nn.MaxPool2d(2),
				nn.Flatten(1),
				nn.Linear(hs[2]*2*2, output_size)
			)
		else:
			raise NotImplementedError

	def forward(self, x):
		return self.model(x.view(-1, *self.input_size)).view(x.shape[0], -1)

class ConvNetNT:
	def __init__(self, arch, hs, output_size):
		super(ConvNetNT, self).__init__()
		self.arch = arch
		self.output_size = output_size

		if self.arch == 'convnet1':
			from jax.experimental import stax
			import functools
			# from neural_tangents import stax
			Conv = functools.partial(stax.GeneralConv, ('NCHW', 'OIHW', 'NCHW'))
			init_fn, f = stax.serial(
			   Conv(hs[0], (3, 3), strides=(2,2), padding='SAME'),
			   stax.Relu,
			   Conv(hs[1], (3, 3), strides=(2,2)),
			   stax.Relu,
			   Conv(hs[2], (6, 6), strides=(1,1)),
			   stax.Relu,
			   Conv(output_size, (1, 1)),
			   stax.Flatten
			)
			self.init_fn = init_fn
			self.f = f
			self.kernel_fn = None #kernel_fn

		elif self.arch == 'convnet2':
			from jax.experimental import stax
			import functools
			Conv = functools.partial(stax.GeneralConv, ('NCHW', 'OIHW', 'NCHW'))
			init_fn, f = stax.serial(
			   Conv(hs[0], (3, 3), padding='SAME'),
			   stax.Relu,
			   stax.MaxPool((2, 2), strides=(2,2), spec='NCHW'),
			   Conv(hs[1], (3, 3)),
			   stax.Relu,
			   stax.MaxPool((2, 2), strides=(2,2), spec='NCHW'),
			   stax.Flatten,
			   stax.Dense(hs[2]),
			   stax.Relu,
			   stax.Dense(output_size),
			)
			self.init_fn = init_fn
			self.f = f
			self.kernel_fn = None
		else:
			raise NotImplementedError

		self.emp_ntk_fn = nt.empirical_ntk_fn(self.f, trace_axes=(-1,),
		                                      vmap_axes=0, implementation=1)
		self.params = None

	def random_init(self, input_size, seed=1):
		_, params = self.init_fn(jax.random.PRNGKey(1), input_size)
		self.params = params

	def ntk(self, x1, x2):
		if self.kernel_fn is None:
			return None
		else:
			return self.kernel_fn(x1, x2, 'ntk')

	def emp_ntk(self, x1, x2):
		if self.params is None:
			return None
		else:
			return self.emp_ntk_fn(x1, x2, self.params)



class ConvNetKernel(nn.Module):
	def __init__(self, arch, input_size, w_var, b_var):
		super(ConvNetKernel, self).__init__()
		self.arch = arch
		self.input_size = input_size

		if self.arch == 'convnet1':
			self.model = cnn_gp.Sequential(
				cnn_gp.Conv2d(kernel_size=3, stride=2, padding=1, var_weight=w_var, var_bias=b_var),
				cnn_gp.ReLU(),
				cnn_gp.Conv2d(kernel_size=3, stride=2, padding=0, var_weight=w_var, var_bias=b_var),
				cnn_gp.ReLU(),
				cnn_gp.Conv2d(kernel_size=6, padding=0, var_weight=w_var, var_bias=b_var),
				cnn_gp.ReLU(),
				cnn_gp.Conv2d(kernel_size=1, padding=0, var_weight=w_var, var_bias=b_var),
				# cnn_gp.Conv2d(kernel_size=1, stride=1, padding=0, var_weight=w_var, var_bias=b_var),
				# cnn_gp.ReLU(),
				# cnn_gp.Conv2d(kernel_size=1, stride=1, padding=0, var_weight=w_var, var_bias=b_var),
				# cnn_gp.ReLU(),
				# cnn_gp.Conv2d(kernel_size=1, padding=0, var_weight=w_var, var_bias=b_var),
				# cnn_gp.ReLU(),
				# cnn_gp.Conv2d(kernel_size=1, padding=0, var_weight=w_var, var_bias=b_var),
			)
		else:
			raise NotImplementedError

	def forward(self, x, x2=None):
		# print(x.shape, x.view(-1, *self.input_size).shape, x2)
		return self.model(x.view(-1, *self.input_size), None if x2 is None else x2.view(-1, *self.input_size))

class ParallelLinear(nn.Module):
	def __init__(self, in_features, out_features, num_copies):
		super(ParallelLinear, self).__init__()
		self.register_parameter('weight', nn.Parameter(torch.randn(num_copies, out_features, in_features)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(num_copies, out_features, 1)))

		for i in range(num_copies):
			nn.init.normal_(self.weight[i], 0, math.sqrt(2./in_features))
		nn.init.zeros_(self.bias)

	def forward(self, x):
		if x.dim() == 2:
			return torch.tensordot(self.weight, x, [[2], [1]]) + self.bias
		else:
			return self.weight @ x + self.bias

class ParallelMLP(nn.Module):
	def __init__(self, in_features, out_features, num_copies, num_layers, hidden_size=64, nonlinearity='relu'):
		super(ParallelMLP, self).__init__()

		if nonlinearity == 'relu':
			nonlinearity=nn.ReLU
		elif 'lrelu' in nonlinearity:
			nonlinearity=partial(nn.LeakyReLU, float(nonlinearity.replace("lrelu", "")))
		elif nonlinearity == 'erf':
			nonlinearity=Erf
		elif nonlinearity == 'sin_and_cos':
			nonlinearity=SinAndCos
		else:
			raise NotImplementedError

		if num_layers == 1:
			self.fn = nn.Sequential(
				ParallelLinear(in_features, out_features, num_copies))
		else:
			layers = [ParallelLinear(in_features, hidden_size, num_copies),
					  nonlinearity(),
					  ParallelLinear(hidden_size, out_features, num_copies)]
			for _ in range(num_layers - 2):
				layers.insert(2, nonlinearity())
				layers.insert(2, ParallelLinear(hidden_size, hidden_size, num_copies))
			self.fn = nn.Sequential(*layers)

	def forward(self, x):
		return self.fn(x).permute(2, 1, 0)

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
	if hasattr(args, 'num_classes'):
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
	if hasattr(args, 'num_classes'):
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
	return train_loader, test_loader


def dataset_with_indices(cls):
	"""
	Modifies the given Dataset class to return a tuple data, target, index
	instead of just data, target.
	"""

	def __getitem__(self, index):
		data, target = cls.__getitem__(self, index)
		return data, target, index

	return type(cls.__name__, (cls,), {
		'__getitem__': __getitem__,
	})


def load_cifar(args):
	if args.dataset == 'cifar10':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		dataset = torchvision.datasets.CIFAR10
	elif args.dataset == 'cifar100':
		mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
		dataset = torchvision.datasets.CIFAR100

	normalize = transforms.Normalize(mean=mean, std=std)

	train_loader = torch.utils.data.DataLoader(
		dataset(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	nef_collect_train_loader = torch.utils.data.DataLoader(
		dataset(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.nef_batch_size_collect, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	nef_train_loader = torch.utils.data.DataLoader(
		dataset_with_indices(dataset)(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.nef_batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		dataset(root=args.data_dir, train=False,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	ood_loader = torch.utils.data.DataLoader(
		torchvision.datasets.SVHN(root=args.data_dir, split='test',
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	return train_loader, nef_collect_train_loader, nef_train_loader, val_loader, ood_loader, 10 if args.dataset == 'cifar10' else 100

class _ECELoss(torch.nn.Module):
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

		bin_boundaries_plot = torch.linspace(0, 1, 11)
		self.bin_lowers_plot = bin_boundaries_plot[:-1]
		self.bin_uppers_plot = bin_boundaries_plot[1:]

	def forward(self, confidences, predictions, labels, title=None):
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=confidences.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		accuracy_in_bin_list = []
		for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			accuracy_in_bin = 0
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean().item()
			accuracy_in_bin_list.append(accuracy_in_bin)

		if title:
			fig = plt.figure(figsize=(8,6))
			p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
			p2 = plt.plot([0,1], [0,1], '--', color='gray')

			plt.ylabel('Accuracy', fontsize=18)
			plt.xlabel('Confidence', fontsize=18)
			#plt.title(title)
			plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.xlim(left=0,right=1)
			plt.ylim(bottom=0,top=1)
			plt.grid(True)
			#plt.legend((p1[0], p2[0]), ('Men', 'Women'))
			plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=18)
			fig.tight_layout()
			plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')

		return ece

def binary_classification_given_uncertainty(uncs_id, uncs_ood, file_name, reverse=False):
	y = np.concatenate([np.zeros((uncs_id.shape[0],)), np.ones((uncs_ood.shape[0],))])
	if reverse:
		y = 1 - y
	x = torch.cat([uncs_id, uncs_ood]).data.cpu().numpy()
	fpr, tpr, thresholds = metrics.precision_recall_curve(y, x)
	auroc = metrics.average_precision_score(y, x)

	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(1, 1, 1)
	sns.kdeplot(uncs_id.data.cpu().numpy(), shade=True, color="r", label='In-distribution')
	sns.kdeplot(uncs_ood.data.cpu().numpy(), shade=True, color="b", label='Out-of-distribution')
	ax.text(0.3, 0.7, 'AUPR: {:.4f}'.format(auroc), fontsize=18, transform=ax.transAxes)
	if 'ntkunc' in file_name:
		plt.legend(loc='center right')
	plt.savefig(file_name, format='pdf', dpi=600, bbox_inches='tight')

	print("\tAUPR is {:.4f}".format(auroc))
	return auroc

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2

        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)

        return conv

    else:
        return False

def fuse_bn_recursively(model):
    previous_name = None

    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization

        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()

        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

        previous_name = module_name

    return model

def load_imagenet(args):
	# Data loading code
	traindir = os.path.join(args.data, 'train')
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_dataset = torchvision.datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))
	idx = np.array(train_dataset.targets) < args.num_classes
	train_dataset.samples = [s for i, s in enumerate(train_dataset.samples) if idx[i]]
	train_dataset.targets = [s[1] for s in train_dataset.samples]

	train_dataset_noaug = torchvision.datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))
	idx = np.array(train_dataset_noaug.targets) < args.num_classes
	train_dataset_noaug.samples = [s for i, s in enumerate(train_dataset_noaug.samples) if idx[i]]
	train_dataset_noaug.targets = [s[1] for s in train_dataset_noaug.samples]

	train_dataset_noaug_with_indices = dataset_with_indices(torchvision.datasets.ImageFolder)(
		traindir,
		transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))
	idx = np.array(train_dataset_noaug_with_indices.targets) < args.num_classes
	train_dataset_noaug_with_indices.samples = [s for i, s in enumerate(train_dataset_noaug_with_indices.samples) if idx[i]]
	train_dataset_noaug_with_indices.targets = [s[1] for s in train_dataset_noaug_with_indices.samples]

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)
	train_loader_no_aug = torch.utils.data.DataLoader(
		train_dataset_noaug, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
	train_loader_no_aug_with_indices = torch.utils.data.DataLoader(
		train_dataset_noaug_with_indices, batch_size=args.nef_batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_dataset = torchvision.datasets.ImageFolder(
		valdir,
		transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))
	idx = np.array(val_dataset.targets) < args.num_classes
	val_dataset.samples = [s for i, s in enumerate(val_dataset.samples) if idx[i]]
	val_dataset.targets = [s[1] for s in val_dataset.samples]

	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	print('# of training data:', len(train_dataset.samples),
		  '\n# of testing data:', len(val_dataset.samples),
		  '\ntraining classes:', train_dataset.classes[:args.num_classes])
	return train_loader, train_loader_no_aug, train_loader_no_aug_with_indices, val_loader

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

if __name__ == '__main__':
	import random
	import numpy as np
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(0)
		torch.cuda.manual_seed_all(0)
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	X = torch.randn(128, 784).cuda()
	X2 = torch.randn(64, 784).cuda()

	# from nngpk import NNGPKernel
	# kernel = NNGPKernel(kernel_type='relu', w_var_list=[2.,2.,2.,2.], b_var_list=[0.01,0.01,0.01,0.01])
	# k1_m = kernel(X)
	# k2_m = kernel(X, X2)

	kernel = ConvNetKernel('convnet1', [1, 28, 28], 2., 0.01).cuda()
	k1_0 = kernel(X)
	k2_0 = kernel(X, X2)

	# print(k1_0[:5, :5], k1_m[:5, :5], k1_0.shape, k1_m.shape)

	# print(torch.dist(k1_0, k1_m))
	# print(torch.dist(k2_0, k2_m))

	random_model = ConvNet('convnet1', [16, 16, 16], input_size=[1, 28, 28], output_size=1).cuda()
	random_model.eval()
	samples = []
	with torch.no_grad():
		with torch.cuda.amp.autocast(False): # this is important!!! Debug for one whole day!!!
			for _ in range(10000):
				# if _ % 50 == 0:
				# 	print("Have obtained {} samples of the ConvNet kernel".format(_))
				init_NN(random_model, 2., 0.01)
				samples.append(random_model(torch.cat([X, X2])))
		samples = torch.cat(samples, -1)
		# print(samples.shape)
	k1_1 = samples[:X.shape[0]] @ samples[:X.shape[0]].T / samples.shape[-1]
	k2_1 = samples[:X.shape[0]] @ samples[X.shape[0]:].T / samples.shape[-1]

	print(k1_0[:5, :5], k1_1[:5, :5], k1_0.shape, k1_1.shape)
	print(torch.dist(k1_0, k1_1))
	print(torch.dist(k2_0, k2_1))
