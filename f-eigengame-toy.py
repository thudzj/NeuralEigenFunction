import math
from typing import List, Tuple
from functools import partial
import copy
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

import tensorflow as tf
import spectral_inference_networks as spin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.distributions import MultivariateNormal

from utils import nystrom, psd_safe_cholesky, rbf_kernel, \
	polynomial_kernel, periodic_plus_rbf_kernel, build_mlp_given_config, ParallelMLP

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

# class PolynomialEigenFunctions(nn.Module):
	# def __init__(self, k, r=5, momentum=0.9, normalize_over=[0], for_spin=False):
	# 	super(PolynomialEigenFunctions, self).__init__()
	# 	self.momentum = momentum
	# 	self.normalize_over = normalize_over
	# 	self.for_spin = for_spin
	# 	self.register_parameter('w', nn.Parameter(torch.randn(r + 1, k) * 1e-3))
	# 	self.register_buffer('eigennorm', torch.zeros(k))
	# 	self.register_buffer('num_calls', torch.Tensor([0]))
	#
	# def forward(self, x):
	# 	with torch.no_grad():
	# 		results = [torch.ones(x.shape[0], device=x.device)]
	# 		for _ in range(1, self.w.shape[0]):
	# 			results.append(results[-1] * x.squeeze())
	# 		results = torch.stack(results, 1)
	# 	ret_raw = results @ self.w
	# 	if self.for_spin:
	# 		return ret_raw
	#
	# 	if self.training:
	# 		norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
	# 					np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
	# 		with torch.no_grad():
	# 			if self.num_calls == 0:
	# 				self.eigennorm.copy_(norm_.data)
	# 			else:
	# 				self.eigennorm.mul_(self.momentum).add_(
	# 					norm_.data, alpha = 1-self.momentum)
	# 			self.num_calls += 1
	# 	else:
	# 		norm_ = self.eigennorm
	# 	return ret_raw / norm_

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, nonlinearity='sin_and_cos', input_size=1, hidden_size=32, num_layers=3, output_size=1,  momentum=0.9, normalize_over=[0], for_spin=False):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.for_spin = for_spin
		self.fn = ParallelMLP(input_size, output_size, k, num_layers, hidden_size, nonlinearity)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = self.fn(x).squeeze()
		if self.for_spin:
			return ret_raw

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

def our(model_class, X, k, kernel, riemannian_projection, max_grad_norm):
	lr = 1e-3
	num_iterations = 2000
	B = min(128, X.shape[0])
	K = kernel(X)

	# perform our method
	start = timer()
	nef = model_class(k)
	optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

	nef.train()
	eigenvalues = None
	for ite in range(num_iterations):
		idx = np.random.choice(X.shape[0], B, replace=False)
		X_batch = X[idx]
		psis_X = nef(X_batch)
		with torch.no_grad():
			K_psis = K[idx][:, idx] @ psis_X
			psis_K_psis = psis_X.T @ K_psis
			mask = torch.eye(k, device=psis_X.device) - \
				(psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
			grad = K_psis @ mask
			if eigenvalues is None:
				eigenvalues = psis_K_psis.diag() / (B**2)
			else:
				eigenvalues.mul_(0.9).add_(psis_K_psis.diag() / (B**2), alpha = 0.1)
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
	nef.eval()
	return eigenvalues, nef, end - start

# def spin_pt(model_class, X, k, kernel):
# 	lr = 1e-3
# 	num_iterations = 2000
# 	B = min(128, X.shape[0])
# 	K = kernel(X)
#
# 	start = timer()
# 	nef = model_class(k, for_spin=True) #1, output_size=
# 	optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
# 	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)
#
# 	nef.train()
# 	sigma = None
# 	J_sigma = None
# 	eigenvalues = None
# 	momentum = 0.01
# 	for ite in range(num_iterations):
# 		# subsample the data and the kernel matrix
# 		idx = np.random.choice(X.shape[0], B, replace=False)
# 		X_batch = X[idx]
# 		# X1, X2 = X_batch.chunk(2, dim=0)
#
# 		# network propagation
# 		U = nef(X_batch)
# 		U1, U2 = U.chunk(2, dim=0)
#
# 		# estimate the Jacobian: d_sigma/d_theta
# 		J_sigma_ = compute_jacobian(nef, X_batch)
#
# 		pi = U1.T @ ((K[idx][:, idx]).diag(diagonal=B//2)[:,None] * U2) / B
#
# 		with torch.no_grad():
#
# 			# sigma moving average
# 			if sigma is None:
# 				sigma = (U1.T @ U1 + U2.T @ U2) / B
# 			else:
# 				sigma.mul_(momentum).add_((U1.T @ U1 + U2.T @ U2) / B, alpha = 1-momentum)
#
# 			# sigma's Jacobian moving average
# 			if J_sigma is None:
# 				J_sigma = J_sigma_
# 			else:
# 				for jac, jac_ in zip(J_sigma, J_sigma_):
# 					jac.mul_(momentum).add_(jac_, alpha = 1-momentum)
#
# 			# estimate the gradients d_l/d_sigma  d_l/d_pi
# 			choli = torch.linalg.inv(psd_safe_cholesky(sigma))
# 			rq = choli @ pi @ choli.T
# 			dl = choli.diag().diag_embed()
# 			dsigma = choli.T @ (rq @ dl).triu()
# 			dpi = - choli.T @ dl
#
# 			# track eigenvals
# 			if eigenvalues is None:
# 				eigenvalues = rq.diag()
# 			else:
# 				eigenvalues.mul_(0.9).add_(rq.diag(), alpha = 0.1)
#
# 		optimizer.zero_grad()
# 		# d_l/d_pi * d_pi/d_theta
# 		pi.backward(dpi)
# 		for param, jac in zip(list(nef.parameters()), J_sigma):
# 			# plus d_l/d_sigma * d_sigma/d_theta
# 			param.grad.add_(torch.tensordot(dsigma, jac, dims=([0, 1], [0, 1])))
# 		optimizer.step()
# 		scheduler.step()
#
# 	# re-arange the eigenvalues from high to low
# 	mask = F.one_hot(torch.from_numpy(np.argsort(-eigenvalues.data.cpu().numpy())).long(), k).float().T
# 	eigenvalues = (eigenvalues[None, :] @ mask).squeeze()
#
# 	choli_T = torch.linalg.inv(psd_safe_cholesky(sigma)).T
# 	nef.eval()
# 	end = timer()
# 	return eigenvalues, lambda x: nef(x) @ choli_T @ mask, end - start

def spin_tf(model_class, X, X_val, k, kernel_type):

	lr = 1e-3
	num_iterations = 2000
	B = min(128, X.shape[0])

	if kernel_type == 'rbf':
		kernel = lambda x, y: tf.exp(-(tf.norm(x-y, axis=1, keepdims=True)**2)/2.)
	elif kernel_type == 'polynomial':
		kernel = lambda x, y: tf.math.pow(tf.math.reduce_sum(x*y, axis=1, keepdims=True) + 1.5, 4)
	linop = spin.KernelOperator(kernel)

	start = timer()
	# Create variables for simple MLP
	w1 = tf.Variable(tf.random.uniform([k, 32, X.shape[1]], -0.4/math.sqrt(X.shape[1]), 0.4/math.sqrt(X.shape[1])))
	w2 = tf.Variable(tf.random.uniform([k, 32, 32], -0.4/math.sqrt(32), 0.4/math.sqrt(32)))
	w3 = tf.Variable(tf.random.uniform([k, 1, 32], -0.4/math.sqrt(32), 0.4/math.sqrt(32)))

	b1 = tf.Variable(tf.zeros([k, 32, 1]))
	b2 = tf.Variable(tf.zeros([k, 32, 1]))
	b3 = tf.Variable(tf.zeros([k, 1, 1]))

	# Create function to construct simple MLP
	def network(x):
	  h1 = tf.tensordot(w1, x, [[2], [1]]) + b1
	  h1_1, h1_2 = tf.split(h1, 2, axis=1)
	  h1_act = tf.concat([tf.math.sin(h1_1), tf.math.cos(h1_2)], 1)

	  h2 = tf.matmul(w2, h1_act) + b2
	  h2_1, h2_2 = tf.split(h2, 2, axis=1)
	  h2_act = tf.concat([tf.math.sin(h2_1), tf.math.cos(h2_2)], 1)

	  h3 = tf.matmul(w3, h2_act) + b3
	  return tf.squeeze(tf.transpose(h3, perm=[2, 1, 0]))

	optim = tf.train.AdamOptimizer(learning_rate=lr)
	# Constructs the internal training ops for spectral inference networks.
	spectral_net = spin.SpectralNetwork(
	    linop,
	    network,
	    X,
	    [w1, w2, b1, b2],
		B, decay=0.99)

	# Trivial defaults for logging and stats hooks.
	logging_config = {
	    'config': {},
	    'log_image_every': 100000000,
	    'save_params_every': 100000000,
	    'saver_path': './tmp',
	    'saver_name': 'example',
	}

	stats_hooks = {
	    'create': spin.util.create_default_stats,
	    'update': spin.util.update_default_stats,
	}

	# Executes the training of spectral inference networks.
	stats, outputs = spectral_net.train(
	    optim,
	    num_iterations,
	    logging_config,
	    stats_hooks,
		data_for_plotting = tf.constant(X_val))
	end = timer()
	return outputs, end - start

def plot_efs(ax, k, X_val, eigenfuncs_eval_list, label_list, linestyle_list,
			 k_lines=3, xlim=[-2., 2.], ylim=[-2., 2.]):

	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	sns.color_palette()
	for iii, eigenfuncs_eval in enumerate(eigenfuncs_eval_list):
		plt.gca().set_prop_cycle(None)
		for i in range(k_lines):
			data = eigenfuncs_eval[:, i] \
				if eigenfuncs_eval[1300:1400, i].mean() > 0 else -eigenfuncs_eval[:, i]
			ax.plot(X_val.view(-1), data, linestyle=linestyle_list[iii],
					label=label_list[iii].format(i+1))

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
	model_class = NeuralEigenFunctions # PolynomialEigenFunctions
	for kernel_type in ['rbf', 'polynomial']: #, 'periodic_plus_rbf']: #
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
		projections_spin_list, cost_spin_list = [], []
		eigenvalues_our_list, nefs_our_list, cost_our_list = [], [], []
		NS = [64, 256, 1024, 4096]
		for N in NS:
			X = torch.empty(N, x_dim).uniform_(x_range[0], x_range[1])

			eigenvalues_nystrom, eigenfuncs_nystrom, c = nystrom(X, k, kernel)
			eigenvalues_nystrom_list.append(eigenvalues_nystrom)
			eigenfuncs_nystrom_list.append(eigenfuncs_nystrom)
			cost_nystrom_list.append(c)

			projections_spin, c = spin_tf(model_class, X, X_val, k, kernel_type)
			projections_spin_list.append(projections_spin)
			cost_spin_list.append(c)

			eigenvalues_our, nef, c = our(model_class, X, k, kernel,
										  riemannian_projection, max_grad_norm)
			eigenvalues_our_list.append(eigenvalues_our)
			nefs_our_list.append(nef)
			cost_our_list.append(c)

			print("-------------------" + str(N) + "-------------------")
			# print("Eigenvalues estimated by nystrom method:")
			# print(eigenvalues_nystrom_list[-1])
			# print("Eigenvalues estimated by spin:")
			# print(eigenvalues_spin_list[-1])
			# print("Eigenvalues estimated by our method:")
			# print(eigenvalues_our_list[-1])
			print("Time comparison {} vs. {} vs. {}".format(
				cost_nystrom_list[-1], cost_spin_list[-1], cost_our_list[-1]))

		label_list = ['$\hat\psi_{}$ (Nyström)', '$\hat\psi_{}$ (SpIN)', '$\hat\psi_{}$ (our)']
		linestyle_list = ['solid', 'dotted', 'dashdot']
		# plots
		fig = plt.figure(figsize=(25, 4.5))
		ax = fig.add_subplot(151)
		with torch.no_grad():
			plot_efs(ax, k, X_val,
					 [eigenfuncs_nystrom_list[0](X_val), projections_spin_list[0], nefs_our_list[0](X_val)],
					 label_list, linestyle_list,
					 k, x_range, ylim)
		if kernel_type != 'rbf':
			# ax.legend(ncol=3, columnspacing=1.2, handletextpad=0.5)
			ax.text(-1.5, -2.2, '$\\kappa(x, x\')=(x^\\top x\' + 1.5)^4$', rotation=90, fontsize=18)
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[0]), pad=20)
		else:
			ax.text(-3.1, -2., '$\\kappa(x, x\')=exp(-||x - x\'||^2/2)$', rotation=90, fontsize=18)
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(152)
		with torch.no_grad():
			plot_efs(ax, k, X_val,
					 [eigenfuncs_nystrom_list[1](X_val), projections_spin_list[1], nefs_our_list[1](X_val)],
					 label_list, linestyle_list,
					 k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[1]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(153)
		with torch.no_grad():
			plot_efs(ax, k, X_val,
					 [eigenfuncs_nystrom_list[2](X_val), projections_spin_list[2], nefs_our_list[2](X_val)],
					 label_list, linestyle_list,
					 k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[2]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		# compare eigenfunctions
		ax = fig.add_subplot(154)
		with torch.no_grad():
			plot_efs(ax, k, X_val,
					 [eigenfuncs_nystrom_list[3](X_val), projections_spin_list[3], nefs_our_list[3](X_val)],
					 label_list, linestyle_list,
					 k, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[3]), pad=20)
		else:
			ax.set_title(' ', pad=20)
		handles, labels = ax.get_legend_handles_labels()

		ax = fig.add_subplot(155)
		ax.tick_params(axis='y', which='major', labelsize=12)
		ax.tick_params(axis='y', which='minor', labelsize=12)
		ax.tick_params(axis='x', which='major', labelsize=12)
		ax.tick_params(axis='x', which='minor', labelsize=12)
		# sns.color_palette()
		ax.plot(range(1, len(NS) + 1), cost_nystrom_list, label='Nyström', color='k')
		ax.plot(range(1, len(NS) + 1), cost_spin_list, label='SpIN', linestyle='dotted', color='k')
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
		ax.legend()
		if kernel_type != 'rbf':
			ax.set_title('Training time comparison', pad=20)
		else:
			ax.set_title(' ', pad=20)

		if kernel_type == 'rbf':
			fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=9, fancybox=True, shadow=True, prop={'size':18})
		fig.tight_layout()
		fig.savefig('toy_plots/eigen_funcs_comp_{}.pdf'.format(kernel_type),
					format='pdf', dpi=1000, bbox_inches='tight')

### utils for spin ###
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		_del_nested_attr(getattr(obj, names[0]), names[1:])

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
	orig_params = tuple(mod.parameters())
	names = []
	for name, p in list(mod.named_parameters()):
		_del_nested_attr(mod, name.split("."))
		names.append(name)
	params = tuple(p.detach().requires_grad_() for p in orig_params)
	return params, names

def _set_nested_attr(obj: Module, names: List[str], value: Tensor) -> None:
	if len(names) == 1:
		setattr(obj, names[0], value)
	else:
		_set_nested_attr(getattr(obj, names[0]), names[1:], value)

def load_weights(mod: Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
	for name, p in zip(names, params):
		_set_nested_attr(mod, name.split("."), p)

def compute_jacobian(model, x):
	jac_model = copy.deepcopy(model)
	all_params, all_names = extract_weights(jac_model)

	def sigma(model, x, names, params):
		load_weights(model, names, params)
		out = model(x)
		U1, U2 = out.chunk(2, dim=0)
		return (U1.T @ U1 + U2.T @ U2) / out.shape[0]

	jacs = torch.autograd.functional.jacobian(lambda *params: sigma(jac_model, x, all_names, params),
										all_params, strict=True, vectorize=False)
	return jacs

if __name__ == '__main__':
	main()
