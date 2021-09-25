# -*- coding: utf-8 -*-
# Original code: https://github.com/TeaPearce/Bayesian_NN_Ensembles/blob/master/regression/module_gp.py
import math
import numpy as np
import importlib
import torch
#
# import neural_tangents as nt
# from neural_tangents import stax

class NNGPKernel:
	def __init__(self, kernel_type, b_var_list=[1., 0.], w_var_list=[2., 1.]):
		self.kernel_type = kernel_type
		self.name_ = 'GP'

		self.b_var_list = b_var_list
		self.w_var_list = w_var_list

	def __relu_kernel(self, X, X2=None):
		def relu_inner(x, x2):
			k_x_x = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x.T))/d_in
			k_x2_x2 = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x2,x2.T))/d_in
			k_x_x2 = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x2.T))/d_in
			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			if k_s>1.0: k_s=1.0 # occasionally get some overflow errors
			theta = np.arccos(k_s)
			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)
			return self.b_var_list[1] + self.w_var_list[1]/(2*np.pi) * x_bar * x2_bar * (np.sin(theta) + (np.pi-theta)*np.cos(theta))

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])
		d_in = X.shape[-1]

		if not same_inputs:
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(X2.shape[0]):
					cov[i,j] = relu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = relu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov

	def __relu_kernel_pt(self, X, X2=None):
		# https://arxiv.org/pdf/1711.00165.pdf
		if X2 is not None:
			X = torch.cat([X, X2], 0)

		K = X @ X.T / X.shape[-1] * self.w_var_list[0] + self.b_var_list[0]
		for i in range(1, len(self.w_var_list)):
			K_diag_sqrt = K.diag().sqrt()
			normalizer = K_diag_sqrt.view(-1, 1) @ K_diag_sqrt.view(1, -1)
			Theta = ((K / normalizer).clamp_(max=1.)).arccos()
			K = (Theta.sin() + (np.pi-Theta) * Theta.cos()) * normalizer / (2*np.pi) * self.w_var_list[i] + self.b_var_list[i]
		if X2 is not None:
			return K[:-X2.shape[0], -X2.shape[0]:]
		else:
			return K

	def __Lrelu_kernel(self, X, X2=None, a=0.2):
		# leaky relu kernel from Tsuchida, 2018, eq. 6

		def Lrelu_inner(x, x2):
			# actually these should be 1/d_in going by Lee. But we leave it normal
			# to be equivalent to our NN ens implementation
			k_x_x = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x.T))/d_in
			k_x2_x2 = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x2,x2.T))/d_in
			k_x_x2 = self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x2.T))/d_in

			k_s = k_x_x2 / np.sqrt(k_x_x * k_x2_x2)
			theta = np.arccos(k_s)

			x_bar = np.sqrt(k_x_x)
			x2_bar = np.sqrt(k_x2_x2)
			return self.b_var_list[1] + self.w_var_list[1] * x_bar * x2_bar * ( np.square(1-a)/(2*np.pi) * (np.sin(theta) + (np.pi-theta)*np.cos(theta)) + a*np.cos(theta))

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])
		d_in = X.shape[-1]

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
		else: # use symmetry
			for i in range(X.shape[0]):
				if i % 10 == 0:
					print('compiling cov, row... '+str(i) + ' / ' + str(X.shape[0]),end='\r')
				for j in range(i+1):
					cov[i,j] = Lrelu_inner(X[i], X2[j])
			cov += np.tril(cov,k=-1).T
		return cov

	def __Lrelu_kernel_pt(self, X, X2=None, a=0.2):
		if X2 is not None:
			X = torch.cat([X, X2], 0)

		K = X @ X.T / X.shape[-1] * self.w_var_list[0] + self.b_var_list[0]
		for i in range(1, len(self.w_var_list)):
			K_diag_sqrt = K.diag().sqrt()
			normalizer = K_diag_sqrt.view(-1, 1) @ K_diag_sqrt.view(1, -1)
			Theta = ((K / normalizer).clamp_(max=1.)).arccos()
			K = ((Theta.sin() + (np.pi-Theta) * Theta.cos()) * np.square(1-a)/(2*np.pi) + a * Theta.cos()) * normalizer * self.w_var_list[i] + self.b_var_list[i]
		if X2 is not None:
			return K[:-X2.shape[0], -X2.shape[0]:]
		else:
			return K

	def __erf_kernel(self, X, X2=None):
		# erf kernel from Williams 1996, eq. 11

		def erf_inner(x,x2):
			# actually these should be 1/d_in
			k_x_x = 2*(self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x.T))/d_in)
			k_x2_x2 = 2*(self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x2,x2.T))/d_in)
			k_x_x2 = 2*(self.b_var_list[0] + self.w_var_list[0]*(np.matmul(x,x2.T))/d_in)
			a = k_x_x2 / np.sqrt((1+k_x_x)*(1+k_x2_x2))
			return self.b_var_list[1] + self.w_var_list[1]*2*np.arcsin(a)/np.pi

		if X2 is None:
			same_inputs=True
			X2 = X
		else:
			same_inputs=False

		cov = np.zeros([X.shape[0],X2.shape[0]])
		d_in = X.shape[-1]

		if not same_inputs:
			for i in range(X.shape[0]):
				for j in range(X2.shape[0]):
					cov[i,j] = erf_inner(X[i],X2[j])
		else:
			for i in range(X.shape[0]):
				for j in range(i+1):
					cov[i,j] = erf_inner(X[i],X2[j])
			# now just reflect - saves recomputing half the matrix
			cov += np.tril(cov,k=-1).T
		return cov

	def __erf_kernel_pt(self, X, X2=None):
		if X2 is not None:
			X = torch.cat([X, X2], 0)

		K = X @ X.T / X.shape[-1] * self.w_var_list[0] + self.b_var_list[0]
		for i in range(1, len(self.w_var_list)):
			K = K * 2
			K_diag_sqrt = K.diag().add(1.).sqrt()
			normalizer = K_diag_sqrt.view(-1, 1) @ K_diag_sqrt.view(1, -1)
			Theta = ((K / normalizer).clamp_(max=1.)).arcsin()
			K = Theta * 2 / np.pi * self.w_var_list[i] + self.b_var_list[i]
		if X2 is not None:
			return K[:-X2.shape[0], -X2.shape[0]:]
		else:
			return K

	@torch.no_grad()
	def __call__(self, X, X2=None):
		if self.kernel_type == 'relu':
			return self.__relu_kernel_pt(X, X2)
		elif 'lrelu' in self.kernel_type:
			return self.__Lrelu_kernel_pt(X, X2, a=float(self.kernel_type.replace("lrelu", "")))
		elif self.kernel_type == 'erf':
			return self.__erf_kernel_pt(X, X2)
		else:
			raise NotImplementedError

if __name__ == '__main__':
	test_mlp_nngpk = 1
	if test_mlp_nngpk:
		from utils import build_mlp_given_config, init_NN

		w_var_list = [2., 2., 2.]
		b_var_list = [1., 1., 1.]
		X = torch.randn(128, 32)
		X2 = torch.randn(64, 32)
		for kernel_type in ['relu', 'lrelu0.2', 'erf']:
			kernel = NNGPKernel(kernel_type=kernel_type, w_var_list=w_var_list, b_var_list=b_var_list)

			k1_0 = kernel(X)
			k2_0 = kernel(X, X2)

			model = build_mlp_given_config(nonlinearity=kernel_type, input_size=X.shape[-1], hidden_size=16, output_size=1, bias=True, num_layers=len(w_var_list))
			# print(model)

			samples = []
			with torch.no_grad():
				for _ in range(100000):
					# if _ % 100 == 0:
					# 	print(_)
					init_NN(model, w_var_list, b_var_list)
					samples.append(model(torch.cat([X, X2])))
			samples = torch.cat(samples, -1)
			k1_1 = samples[:X.shape[0]] @ samples[:X.shape[0]].T / samples.shape[-1]
			k2_1 = samples[:X.shape[0]] @ samples[X.shape[0]:].T / samples.shape[-1]

			# print(k1_0.shape, k1_1.shape)
			print(k1_0[:5, :5], k1_1[:5, :5])

			print(torch.dist(k1_0, k1_1))
			print(torch.dist(k2_0, k2_1))
	else:
		pass
