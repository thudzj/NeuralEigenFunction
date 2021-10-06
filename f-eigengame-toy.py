import math
from functools import partial
import itertools
from timeit import default_timer as timer

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 18})
import pandas as pd
import seaborn as sns

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from utils import nystrom, psd_safe_cholesky, rbf_kernel, \
	polynomial_kernel, periodic_plus_rbf_kernel

class PolynomialEigenFunctions(nn.Module):
	def __init__(self, k, r, momentum=0.9, normalize_over=[0]):
		super(PolynomialEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.register_parameter('w', nn.Parameter(torch.ones(r + 1, k) * 1e-3))
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		with torch.no_grad():
			results = [torch.ones(x.shape[0], device=x.device)]
			for _ in range(1, self.w.shape[0]):
				results.append(results[-1] * x.squeeze())
			results = torch.stack(results, 1)
		ret_raw = results @ self.w
		if self.training:
			norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
						np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
			with torch.no_grad():
				if self.num_calls == 0:
					self.eigennorm.copy_(norm_.data)
				else:
					self.eigennorm.mul_(self.momentum).add_(
						norm_.data, alpha = 1-self.momentum)
				self.num_calls += 1
		else:
			norm_ = self.eigennorm
		return ret_raw / norm_

def our(X, x_dim, x_range, k, kernel, kernel_type, riemannian_projection, 
		max_grad_norm):
	optimizer_type = 'Adam'
	lr = 1e-3
	num_iterations = 2000
	num_samples = 2000
	B = min(128, X.shape[0])
	K = kernel(X)

	# perform our method
	start = timer()
	nef = PolynomialEigenFunctions(k, 5 if kernel_type != 'periodic_plus_rbf' else 5)
	if optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
	elif optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

	nef.train()
	eigenvalues_our = None
	for ite in range(num_iterations):

		idx = np.random.choice(X.shape[0], B, replace=False)
		# samples_batch = samples[:, idx]
		X_batch = X[idx]
		psis_X = nef(X_batch)
		with torch.no_grad():
			K_psis = K[idx][:, idx] @ psis_X
			psis_K_psis = psis_X.T @ K_psis
			mask = torch.eye(k, device=psis_X.device) - \
				(psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
			grad = K_psis @ mask
			if eigenvalues_our is None:
				eigenvalues_our = psis_K_psis.diag() / (B**2)
			else:
				eigenvalues_our.mul_(0.9).add_(psis_K_psis.diag() / (B**2), alpha = 0.1)
			if riemannian_projection:
				grad.sub_((psis_X*grad).sum(0) * psis_X / B)
			if max_grad_norm is not None:
				clip_coef = max_grad_norm / (grad.norm(dim=0) + 1e-6)
				grad.mul_(clip_coef)

		optimizer.zero_grad()
		psis_X.backward(-grad)
		optimizer.step()
		scheduler.step()
	end = timer()
	return eigenvalues_our, nef, end - start

def plot_efs(ax, k, X_val, eigenfuncs_eval_nystrom, eigenfuncs_eval_our=None, 
			 k_lines=3, xlim=[-2., 2.], ylim=[-2., 2.]):

	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	sns.color_palette()
	for i in range(k_lines):
		data = eigenfuncs_eval_nystrom[:, i] \
			if eigenfuncs_eval_nystrom[1300:1400, i].mean() > 0 else -eigenfuncs_eval_nystrom[:, i]
		ax.plot(X_val.view(-1), data, alpha=1, label='$\hat\psi_{}$ (Nyström)'.format(i+1))

	if eigenfuncs_eval_our is not None:
		plt.gca().set_prop_cycle(None)
		for i in range(k_lines):
			data = eigenfuncs_eval_our[:, i] \
				if eigenfuncs_eval_our[1300:1400, i].mean() > 0 else -eigenfuncs_eval_our[:, i]
			ax.plot(X_val.view(-1), data, linestyle='dashdot', label='$\hat\psi_{}$ (our)'.format(i+1))

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_xlim(xlim[0], xlim[1])
	ax.set_ylim(ylim[0], ylim[1])

	ax.spines['bottom'].set_color('gray')
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.spines['left'].set_color('gray')
	ax.set_axisbelow(True)
	ax.grid(axis='y', color='lightgray', linestyle='--')
	ax.grid(axis='x', color='lightgray', linestyle='--')


def main():
	# set random seed
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# general settings
	x_dim = 1
	x_range = [-2, 2]
	k = 3
	riemannian_projection = False
	max_grad_norm = None
	for kernel_type in ['rbf', 'polynomial', 'periodic_plus_rbf']: #
		if kernel_type == 'rbf':
			kernel = partial(rbf_kernel, 1, 1)
			ylim = [-2., 2.]
		elif kernel_type == 'periodic_plus_rbf':
			kernel = partial(periodic_plus_rbf_kernel, 1.5, 1, 1, 1, 1)
			x_range = [-1., 1.]
			ylim = [-2., 2.]
		elif kernel_type == 'cosine':
			kernel = partial(cosine_kernel, 4, 1, 1)
			ylim = [-2., 2.]
		elif kernel_type == 'polynomial':
			kernel = partial(polynomial_kernel, 4, 1, 1.5)
			x_range = [-1., 1.]
			ylim = [-3., 3.]
		elif kernel_type == 'sigmoid':
			kernel = partial(sigmoid_kernel, 1, 2)
			x_range = [-1, 1]
			ylim = [-2., 2.]

		X_val = torch.arange(x_range[0], x_range[1], 
					(x_range[1] - x_range[0]) / 2000.).view(-1, 1)
		eigenvalues_nystrom_list, eigenfuncs_nystrom_list, cost_nystrom_list = [], [], []
		eigenvalues_our_list, nefs_our_list, cost_our_list = [], [], []
		NS = [64, 256, 1024, 4096]
		for N in NS:
			X = torch.empty(N, x_dim).uniform_(x_range[0], x_range[1])

			eigenvalues_nystrom, eigenfuncs_nystrom, c = nystrom(X, k, kernel)
			eigenvalues_nystrom_list.append(eigenvalues_nystrom)
			eigenfuncs_nystrom_list.append(eigenfuncs_nystrom)
			cost_nystrom_list.append(c)

			eigenvalues_our, nef, c = our(X, x_dim, x_range, k, kernel, kernel_type, 
										  riemannian_projection, max_grad_norm)
			eigenvalues_our_list.append(eigenvalues_our)
			nefs_our_list.append(nef)
			cost_our_list.append(c)

			print("-------------------" + str(N) + "-------------------")
			print("Eigenvalues estimated by nystrom method:")
			print(eigenvalues_nystrom_list[-1])
			print("Eigenvalues estimated by our method:")
			print(eigenvalues_our_list[-1])
			print("Time comparison {} vs. {}".format(cost_nystrom_list[-1], cost_our_list[-1]))

		[nef.eval() for nef in nefs_our_list]
		# plots
		fig = plt.figure(figsize=(25, 4.5))
		ax = fig.add_subplot(151)
		with torch.no_grad():
			plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[0](X_val), 
					 nefs_our_list[0](X_val), k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.legend(ncol=2, columnspacing=1.2, handletextpad=0.5)
			ax.text(-1.5, -2.2, '$\\kappa(x, x\')=(x^\\top x\' + 1.5)^4$', rotation=90, fontsize=18)
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[0]), pad=20)
		else:
			ax.text(-3.1, -2., '$\\kappa(x, x\')=exp(-||x - x\'||^2/2)$', rotation=90, fontsize=18)
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(152)
		with torch.no_grad():
			plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[1](X_val), 
					 nefs_our_list[1](X_val), k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[1]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(153)
		with torch.no_grad():
			plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[2](X_val), 
					 nefs_our_list[2](X_val), k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[2]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		# compare eigenfunctions
		ax = fig.add_subplot(154)
		with torch.no_grad():
			plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[3](X_val), 
					 nefs_our_list[3](X_val), k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[3]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(155)
		ax.tick_params(axis='y', which='major', labelsize=12)
		ax.tick_params(axis='y', which='minor', labelsize=12)
		ax.tick_params(axis='x', which='major', labelsize=12)
		ax.tick_params(axis='x', which='minor', labelsize=12)
		# sns.color_palette()
		ax.plot(range(1, len(NS) + 1), cost_nystrom_list, label='Nyström', color='k')
		ax.plot(range(1, len(NS) + 1), cost_our_list, label='Our', linestyle='dashdot', color='k')
		ax.set_xlim(1, len(NS) + 0.2)
		ax.set_xticks(range(1, len(NS) + 1))
		ax.set_xticklabels(NS)
		# ax.set_ylim(ylim[0], ylim[1])
		ax.set_xlabel('Number of samples')
		ax.set_ylabel('Training time (s)')
		ax.spines['bottom'].set_color('gray')
		ax.spines['top'].set_color('gray')
		ax.spines['right'].set_color('gray')
		ax.spines['left'].set_color('gray')
		# ax.spines['right'].set_visible(False)
		# ax.spines['top'].set_visible(False)
		ax.set_axisbelow(True)
		ax.grid(axis='y', color='lightgray', linestyle='--')
		ax.grid(axis='x', color='lightgray', linestyle='--')
		if kernel_type != 'rbf':
			ax.legend()
		if kernel_type != 'rbf':
			ax.set_title('Training time comparison', pad=20)
		else:
			ax.set_title(' ', pad=20)
		fig.tight_layout()
		fig.savefig('toy_plots/eigen_funcs_comp_{}.pdf'.format(kernel_type), 
					format='pdf', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
	main()
