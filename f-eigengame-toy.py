# g20: CUDA_VISIBLE_DEVICES=2 python f-eigengame-toy.py
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
	def __init__(self, k, nonlinearity='sin_and_cos', input_size=1, hidden_size=32, num_layers=3, output_size=1,  momentum=0.9, normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.fn = ParallelMLP(input_size, output_size, k, num_layers, hidden_size, nonlinearity)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = self.fn(x).squeeze()
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

def our(model_class, X_, X_val_, k, kernel, riemannian_projection, max_grad_norm):
	X = X_.cuda()
	X_val = X_val_.cuda()
	lr = 1e-3
	num_iterations = 2000
	B = min(256, X.shape[0])
	K = kernel(X)

	# perform our method
	start = timer()
	nef = model_class(k).cuda()
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
	with torch.no_grad():
		projections_val = nef(X_val).data.cpu().numpy()
	return eigenvalues.data.cpu(), projections_val, end - start

def spin_tf(X, X_val, k, kernel_type):

	lr = 1e-3
	num_iterations = 2000
	B = min(256, X.shape[0])

	if kernel_type == 'rbf':
		kernel = lambda x, y: tf.exp(-(tf.norm(x-y, axis=1, keepdims=True)**2)/2.)
	elif kernel_type == 'polynomial':
		kernel = lambda x, y: tf.math.pow(tf.math.reduce_sum(x*y, axis=1, keepdims=True) + 1.5, 4)
	linop = spin.KernelOperator(kernel)

	start = timer()
	# Create variables for simple MLP
	w1 = tf.Variable(tf.random.normal([k, 32, X.shape[1]], 0, math.sqrt(2./X.shape[1])))
	w2 = tf.Variable(tf.random.normal([k, 32, 32], 0, math.sqrt(2./32)))
	w3 = tf.Variable(tf.random.normal([k, 1, 32], 0, math.sqrt(2./32)))

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

def plot_efs(ax, X_val, eigenfuncs_eval_list, label_list, linestyle_list,
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
	k = 10
	riemannian_projection = False
	max_grad_norm = None
	model_class = NeuralEigenFunctions # PolynomialEigenFunctions
	for kernel_type in ['rbf', 'polynomial']: # polynomial rbf #, 'periodic_plus_rbf']: #
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
		# NS = [64, 256, 1024, 4096]
		NS = [64, 512, 8192]
		XS = [torch.empty(NS[-1], x_dim).uniform_(x_range[0], x_range[1])]
		for N in NS[:-1]:
			XS.insert(-1, XS[-1][:N])

		eigenvalues_nystrom_list, projections_nystrom_list, cost_nystrom_list = [], [], []
		eigenvalues_our_list, projections_our_list, cost_our_list = [], [], []
		projections_spin_list, cost_spin_list = [], []
		for X in XS:
			eigenvalues_our, projections_our, c = our(model_class, X, X_val, k, kernel,
										  riemannian_projection, max_grad_norm)
			eigenvalues_our_list.append(eigenvalues_our)
			projections_our_list.append(projections_our)
			cost_our_list.append(c)
			print("---------{}---------".format(X.shape[0]))
			print("Eigenvalues estimated by our method:")
			print(eigenvalues_our_list[-1])

		for X in XS:
			eigenvalues_nystrom, eigenfuncs_nystrom, c = nystrom(X, k, kernel)
			eigenvalues_nystrom_list.append(eigenvalues_nystrom)
			projections_nystrom_list.append(eigenfuncs_nystrom(X_val).data.cpu().numpy())
			cost_nystrom_list.append(c)
			print("Eigenvalues estimated by nystrom method:")
			print(eigenvalues_nystrom_list[-1])

		for X in XS:
			projections_spin, c = spin_tf(X, X_val, k, kernel_type)
			if kernel_type == 'polynomial' and X.shape[0] == 64:
				projections_spin = - projections_spin
			projections_spin_list.append(projections_spin)
			cost_spin_list.append(c)

		label_list = ['$\hat\psi_{}$ (Nyström)', '$\hat\psi_{}$ (SpIN)', '$\hat\psi_{}$ (our)']
		linestyle_list = ['solid', 'dotted', 'dashdot']
		# plots
		fig = plt.figure(figsize=(5*len(NS) + 5, 4))
		ax = fig.add_subplot(141)
		plot_efs(ax, X_val,
				 [projections_nystrom_list[0], projections_spin_list[0], projections_our_list[0]],
				 label_list, linestyle_list,
				 3, x_range, ylim)
		if kernel_type != 'rbf':
			# ax.legend(ncol=3, columnspacing=1.2, handletextpad=0.5)
			ax.text(-1.5, -2.7, '$\\kappa(x, x\')=(x^\\top x\' + 1.5)^4$', rotation=90, fontsize=16)
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[0]), pad=20)
		else:
			ax.text(-3.1, -2.3, '$\\kappa(x, x\')=exp(-||x - x\'||^2/2)$', rotation=90, fontsize=16)
			ax.set_title(' ', pad=20)

		ax = fig.add_subplot(142)
		plot_efs(ax, X_val,
				 [projections_nystrom_list[1], projections_spin_list[1], projections_our_list[1]],
				 label_list, linestyle_list,
				 3, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[1]), pad=20)
		else:
			ax.set_title(' ', pad=20)

		# ax = fig.add_subplot(143)
		# plot_efs(ax, X_val,
		# 		 [projections_nystrom_list[2], projections_spin_list[2], projections_our_list[2]],
		# 		 label_list, linestyle_list,
		# 		 3, x_range, ylim)
		# if kernel_type != 'rbf':
		# 	ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[2]), pad=20)
		# else:
		# 	ax.set_title(' ', pad=20)

		# compare eigenfunctions
		ax = fig.add_subplot(143)
		plot_efs(ax, X_val,
				 [projections_nystrom_list[2], projections_spin_list[2], projections_our_list[2]],
				 label_list, linestyle_list,
				 3, x_range, ylim)
		if kernel_type != 'rbf':
			ax.set_title('Eigenfunction comparison ({} samples)'.format(NS[2]), pad=20)
		else:
			ax.set_title(' ', pad=20)
		handles, labels = ax.get_legend_handles_labels()

		ax = fig.add_subplot(144)
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
			fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=9, fancybox=True, shadow=True, prop={'size':16})
			fig.tight_layout()
			fig.savefig('toy_plots/eigen_funcs_comp_{}.pdf'.format(kernel_type),
						format='pdf', dpi=1000, bbox_inches='tight')
		else:
			fig.tight_layout()
			fig.savefig('toy_plots/eigen_funcs_comp_{}.pdf'.format(kernel_type),
						format='pdf', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
	main()
