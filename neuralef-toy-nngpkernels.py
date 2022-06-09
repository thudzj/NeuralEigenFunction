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
plt.rcParams.update({'font.size': 18})
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns

import random
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import spectral_inference_networks as spin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from utils import nystrom, build_mlp_given_config, init_NN, ParallelMLP
from nngpk import NNGPKernel

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, kernel_type, input_size, hidden_size, num_layers,
				 output_size=1, momentum=0.9, normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.functions = nn.ModuleList()
		for i in range(k):
			function = build_mlp_given_config(nonlinearity=kernel_type,
											  input_size=input_size,
											  hidden_size=hidden_size,
											  output_size=output_size,
											  num_layers=num_layers,
											  bias=True)
			self.functions.append(function)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = torch.cat([f(x) for f in self.functions], 1)
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

def our(X, k, kernel, kernel_type, w_var_list, b_var_list):
	hidden_size = 32
	num_layers = len(w_var_list)
	optimizer_type = 'Adam'
	lr = 1e-3
	momentum = 0.9
	num_iterations = 2000
	num_samples = 10000
	B = min(128, X.shape[0])

	random_model =  build_mlp_given_config(nonlinearity=kernel_type,
										   input_size=X.shape[-1],
										   hidden_size=16,
										   output_size=1,
										   bias=True,
										   num_layers=num_layers)
	samples = []
	with torch.no_grad():
		for _ in range(num_samples):
			init_NN(random_model, w_var_list, b_var_list)
			samples.append(random_model(X))
	samples = torch.cat(samples, -1).T

	start = timer()
	nef = NeuralEigenFunctions(k, kernel_type, X.shape[-1], hidden_size, num_layers)
	print(nef.functions[0])
	if optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
	elif optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)

	nef.train()
	eigenvalues_our = None
	for ite in range(num_iterations):
		idx = np.random.choice(X.shape[0], B, replace=False)
		samples_batch = samples[:, idx]
		psis_X = nef(X[idx])
		with torch.no_grad():
			samples_batch_psis = samples_batch @ psis_X
			psis_K_psis = samples_batch_psis.T @ samples_batch_psis / num_samples
			mask = torch.eye(k, device=psis_X.device) - \
				(psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
			grad = samples_batch.T @ (samples_batch_psis @ mask / num_samples)

			if eigenvalues_our is None:
				eigenvalues_our = psis_K_psis.diag() / (B**2)
			else:
				eigenvalues_our.mul_(0.9).add_(psis_K_psis.diag() / (B**2), alpha = 0.1)

		optimizer.zero_grad()
		psis_X.backward(-grad)
		optimizer.step()
	end = timer()
	return eigenvalues_our, nef, end - start

def spin_tf(X, X_val, k, kernel, kernel_type):

	lr = 1e-3
	num_iterations = 2000
	B = min(128, X.shape[0])

	linop = spin.KernelOperator(kernel.make_relu_kernel_tf()
		if kernel_type == 'relu' else kernel.make_erf_kernel_tf())

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
	  if kernel_type == 'erf':
		  h1_act = tf.math.erf(h1)
	  elif kernel_type == 'relu':
		  h1_act = tf.nn.relu(h1)
	  else:
		  raise NotImplementedError

	  h2 = tf.matmul(w2, h1_act) + b2
	  if kernel_type == 'erf':
		  h2_act = tf.math.erf(h2)
	  elif kernel_type == 'relu':
		  h2_act = tf.nn.relu(h2)
	  else:
		  raise NotImplementedError

	  h3 = tf.matmul(w3, h2_act) + b3
	  return tf.squeeze(tf.transpose(h3, perm=[2, 1, 0]))

	optim = tf.train.AdamOptimizer(learning_rate=lr)
	# Constructs the internal training ops for spectral inference networks.
	spectral_net = spin.SpectralNetwork(
	    linop,
	    network,
	    X,
	    [w1, w2, w3, b1, b2, b3],
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

def main():
	# set random seed
	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# setup the nngp kernel
	kernel_type = 'erf'
	w_var_list = [2., 2., 2.]
	b_var_list = [1, 1, 1]

	# general settings
	num_alldata = 1000
	k = 3
	# dataset settings
	dataset = 'circles'
	if dataset == 'two_moon':
		X, y = make_moons(num_alldata, noise=0.04, random_state=seed)
		kernel_type = 'relu'
	elif dataset == 'circles':
		X, y = make_circles(num_alldata, noise=0.04, factor=0.5, random_state=seed)
	else:
		raise NotImplementedError

	kernel = NNGPKernel(kernel_type=kernel_type, w_var_list=w_var_list, b_var_list=b_var_list)

	X = StandardScaler().fit_transform(X)
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + 0.6
	X = torch.from_numpy(X).float()

	# plot the dataset
	figure = plt.figure(figsize=(15, 5))
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	ax = figure.add_subplot(131)
	ax.set_title("Input data")
	# Plot the training points
	ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
			   edgecolors='k')
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.set_xticks(())
	ax.set_yticks(())
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.set_axisbelow(True)

	eigenvalues_nystrom, eigenfuncs_nystrom, c_nystrom = nystrom(X, k, kernel)
	eigenvalues_our, nef, c_our = our(X, k, kernel, kernel_type, w_var_list, b_var_list)

	print("Eigenvalues estimated by nystrom method:")
	print(eigenvalues_nystrom)
	print("Eigenvalues estimated by our method:")
	print(eigenvalues_our)
	print("Time comparison {} vs. {}".format(c_nystrom, c_our))

	nef.eval()
	with torch.no_grad():
		X_projected_by_nystrom = eigenfuncs_nystrom(X)
		X_projected_by_our = nef(X)
		print(X_projected_by_nystrom[: 5])
		print(X_projected_by_our[: 5])

	X_projected_by_spin, c_spin = spin_tf(X, X, k, kernel, kernel_type)

	ax = figure.add_subplot(132, projection='3d')
	ax.set_title("Projected by our method")
	X_projected_by_our_0 = -X_projected_by_our[:, 0] if dataset == 'two_moon' else X_projected_by_our[:, 0]
	X_projected_by_our_1 = -X_projected_by_our[:, 1] if dataset == 'two_moon' else X_projected_by_our[:, 1]
	X_projected_by_our_2 = X_projected_by_our[:, 2]
	ax.scatter(X_projected_by_our_0, X_projected_by_our_1, X_projected_by_our_2, c=y, cmap=cm_bright,
			   edgecolors='k')
	ax.grid(True)
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)
	plt.setp( ax.get_zticklabels(), visible=False)

	ax = figure.add_subplot(133, projection='3d')
	ax.set_title("Projected by SpIN")
	X_projected_by_spin_0 = X_projected_by_spin[:, 0] if dataset == 'two_moon' else X_projected_by_spin[:, 0]
	X_projected_by_spin_1 = -X_projected_by_spin[:, 1] if dataset == 'two_moon' else X_projected_by_spin[:, 1]
	X_projected_by_spin_2 = X_projected_by_spin[:, 2]
	ax.scatter(X_projected_by_spin_0, X_projected_by_spin_1, X_projected_by_spin_2, c=y, cmap=cm_bright,
			   edgecolors='k')
	ax.grid(True)
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)
	plt.setp( ax.get_zticklabels(), visible=False)

	figure.tight_layout()
	figure.savefig('nngp_plots/{}.pdf'.format(dataset), format='pdf', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
	main()
