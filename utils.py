import warnings
import math
from timeit import default_timer as timer
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

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
	return ((x1.unsqueeze(1) * x2.unsqueeze(0)).sum(-1) * eta + nu) ** degree

def sigmoid_kernel(eta, nu, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	return ((x1.unsqueeze(1) * x2.unsqueeze(0)).sum(-1) * eta + nu).tanh()

def cosine_kernel(period, output_scale, length_scale, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	return (((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1).sqrt() * math.pi / period / length_scale).cos() * output_scale

def rbf_kernel(output_scale, length_scale, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	return (- ((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1) / 2. / length_scale).exp() * output_scale

def periodic_plus_rbf_kernel(period, output_scale1, length_scale1, output_scale2, length_scale2, x1, x2=None):
	if x2 is None:
		x2 = x1
	if x1.dim() == 1:
		x1 = x1.unsqueeze(-1)
	if x2.dim() == 1:
		x2 = x2.unsqueeze(-1)
	out1 = (- (((x1.unsqueeze(1) - x2.unsqueeze(0)).abs().sum(-1) * math.pi / period).sin() ** 2) * 2. / length_scale1).exp() * output_scale1
	out2 = (- ((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1) / 2. / length_scale2).exp() * output_scale2
	return out1 + out2


def nystrom(X, k, kernel):
	start = timer()
	K = kernel(X)
	p, q = torch.symeig(K, eigenvectors=True)
	eigenvalues_nystrom = p[range(-1, -(k+1), -1)] / X.shape[0]
	eigenfuncs_nystrom = lambda x: kernel(x, X) @ q[:, range(-1, -(k+1), -1)] \
									 / p[range(-1, -(k+1), -1)] * math.sqrt(X.shape[0])
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

def build_mlp_given_config(**kwargs):
	if kwargs['nonlinearity'] == 'relu':
		nonlinearity=nn.ReLU
	elif 'lrelu' in kwargs['nonlinearity']:
		nonlinearity=partial(nn.LeakyReLU, float(kwargs['nonlinearity'].replace("lrelu", "")))
	elif kwargs['nonlinearity'] == 'erf':
		nonlinearity=Erf
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
