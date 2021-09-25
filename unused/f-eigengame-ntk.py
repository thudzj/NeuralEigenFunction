import copy
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
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from sklearn import svm
from sklearn.linear_model import SGDClassifier

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from utils import nystrom, init_NN, load_mnist, ConvNet, fuse_bn_recursively, \
	_ECELoss, data_transform, binary_classification_given_uncertainty, \
	psd_safe_cholesky


parser = argparse.ArgumentParser(description='Decompose the ConvNet kernel on MNIST')
parser.add_argument('--data-path', type=str,
					default='/data/zhijie/data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=1000, type=int,
					metavar='N', help='mini-batch size (default: 1000)')

parser.add_argument('--arch', default='convnet3', type=str)
parser.add_argument('--bhs-r', default=[8, 8, 8], type=int, nargs='+', help='hidden size')
parser.add_argument('--num-samples', default=3000, type=int)
parser.add_argument('--random-dist-type', default='rademacher', type=str)
parser.add_argument('--epsilon', default=1e-5, type=float, help='epsilon')
parser.add_argument('--delta', default=1, type=float, help='delta')

parser.add_argument('--bhs', default=[16, 32, 64], type=int, nargs='+', help='base hidden size for eigenfuncs')
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--optimizer-type', default='Adam', type=str)
parser.add_argument('--num-iterations', default=20000, type=int)
parser.add_argument('--riemannian-projection', action='store_true')
parser.add_argument('--max-grad-norm', default=10., type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--num-samples-eval', default=256, type=int)
parser.add_argument('--class-cond', action='store_true')

parser.add_argument('--job-id', default='', type=str)

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, arch, bhs, input_size, output_size=1):
		super(NeuralEigenFunctions, self).__init__()
		self.functions = nn.ModuleList()
		for i in range(k):
			function = ConvNet(arch, bhs, input_size, output_size)
			self.functions.append(function)

	def forward(self, x):
		return torch.stack([f(x) for f in self.functions], -1)

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

	# pre-train the classifier
	model = ConvNet(args.arch, args.bhs_r, input_size=[1, 28, 28], output_size=10)
	model = model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
	for epoch in tqdm(range(2), desc = 'Pre-training the classifier'):
		model.train()
		for x, y in train_loader:
			x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
			output = model(x)
			loss = F.cross_entropy(output, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		validate(test_loader, model)

	model = fuse_bn_recursively(model)
	num_params = sum(p.numel() for p in model.parameters())
	model.eval()
	validate(test_loader, model)

	# get mc estimate of the kernel
	# model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
	# original_params = copy.deepcopy(list(model.parameters()))
	# with torch.no_grad():
	# 	original_logits = model(X)
	# samples = []
	# for i in tqdm(range(args.num_samples), desc = 'Sampling from the NTK kernel'):
	# 	for op, p in zip(original_params, list(model.parameters())):
	# 		if args.random_dist_type == 'normal':
	# 			perturbation = torch.randn_like(p) * args.epsilon
	# 		elif args.random_dist_type == 'rademacher':
	# 			perturbation = torch.randn_like(p).sign() * args.epsilon
	# 		else:
	# 			raise NotImplementedError
	# 		p.data.copy_(op.data + perturbation)
	# 	with torch.no_grad():
	# 		new_logits = model(X)
	# 	samples.append((new_logits - original_logits).div(args.epsilon).data.cpu())
	# samples = torch.stack(samples, 0)
	# model = model.module

	# K1 = torch.einsum("smc,snd->mcnd", samples[:, :100, :], samples[:, :100, :])/args.num_samples
	X_batch, Y_batch = X[:100], Y[:100].cuda()
	model = extend(model)
	F_pred = []
	for class_idx in range(10):
		output = model(X_batch)
		model.zero_grad()
		with backpack(BatchGrad()):
			output[:, class_idx].sum().backward()
		F_pred.append(torch.cat([p.grad_batch.flatten(1) for p in model.parameters()], -1))
	F_pred = torch.stack(F_pred, 1)
	K2 = torch.einsum("mcs,nds->mcnd", F_pred, F_pred)#.flatten(0,1).flatten(1)

	lr = 1e-3
	logits = model(X_batch)
	with torch.no_grad():
		dl_df = (logits.softmax(-1) - F.one_hot(Y_batch, logits.shape[-1]))
	model.zero_grad()
	with backpack(BatchGrad()):
		logits.backward(dl_df)


	new_model = copy.deepcopy(model)
	K1 = []
	for i in range(len(X_batch)):
		with torch.no_grad():
			for p, p_ in zip(new_model.parameters(), model.parameters()):
				p.data.copy_(p_.data - p_.grad_batch[i] * lr)
		new_logits = new_model(X_batch)
		K1.append((logits-new_logits).unsqueeze(-1) / dl_df[i] / lr)
	K1 = torch.stack(K1, -2)

	print(torch.dist(K1, K2))
	print(K1.view(100, 10, 100, 10)[0, :, 0, :], K2.view(100, 10, 100, 10)[0, :, 0, :])
	exit()

	# train nef
	start = timer()
	nef = NeuralEigenFunctions(args.k, args.arch, args.bhs, input_size=[1, 28, 28], output_size=10).to(device)
	nef = torch.nn.DataParallel(nef, device_ids=list(range(torch.cuda.device_count())))
	if args.optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=args.lr)
	elif args.optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=args.lr, momentum=args.momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=args.lr, momentum=args.momentum)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_iterations)
	eigenvalues = None
	nef.train()
	for ite in tqdm(range(args.num_iterations), desc = 'Training NEF'):
		idx = np.random.choice(X.shape[0], args.bs, replace=False)
		samples_batch = samples[:, idx, :].to(device)
		psis_X_raw = nef(X[idx])
		B, C = psis_X_raw.shape[0], psis_X_raw.shape[1]
		psis_X = psis_X_raw * (math.sqrt(B)/psis_X_raw.norm(dim=0) if args.class_cond
						  else math.sqrt(B*C)/psis_X_raw.norm(dim=[0, 1]))
		with torch.no_grad():
			if args.class_cond:
				samples_batch_psis = torch.einsum('nbc,bck->nkc', samples_batch, psis_X)
				psis_K_psis = torch.einsum('nkc,nlc->klc', samples_batch_psis, samples_batch_psis) / args.num_samples
				psis_K_psis = psis_K_psis.permute(2, 0, 1)

				cur_eigenvalues = torch.diagonal(psis_K_psis, dim1=-2, dim2=-1)
				mask = - (psis_K_psis / cur_eigenvalues.unsqueeze(1)).tril(diagonal=-1).permute(0, 2, 1)
				mask += torch.eye(args.k, device=psis_X.device)
				mask /= args.num_samples
				grad = torch.einsum('nkc,ckl->nlc', samples_batch_psis, mask)
				grad = torch.einsum('nbc,nkc->bck', samples_batch, grad)
				cur_eigenvalues /= B**2
			else:
				samples_batch_psis = torch.einsum('nbc,bck->nk', samples_batch, psis_X)
				psis_K_psis = samples_batch_psis.T @ samples_batch_psis / args.num_samples

				cur_eigenvalues = psis_K_psis.diag()
				mask = - (psis_K_psis / cur_eigenvalues).tril(diagonal=-1).T
				mask += torch.eye(args.k, device=psis_X.device)
				mask /= args.num_samples
				grad = samples_batch_psis @ mask
				grad = torch.einsum('nbc,nk->bck', samples_batch, grad)
				cur_eigenvalues /= (B*C)**2

			if eigenvalues is None:
				eigenvalues = cur_eigenvalues
			else:
				eigenvalues.mul_(0.9).add_(cur_eigenvalues, alpha = 0.1)

			if args.riemannian_projection:
				grad.sub_((psis_X*grad).sum(0) * psis_X)

			clip_coef = args.max_grad_norm / (grad.norm(dim=0) + 1e-6)
			grad.mul_(clip_coef)

		optimizer.zero_grad()
		psis_X.backward(-grad)
		optimizer.step()
		scheduler.step()
	end = timer()
	print("Our method consumes {}s".format(end - start))
	print(eigenvalues)

	idx = np.random.choice(X.shape[0], 100, replace=False)
	nef = nef.module
	nef.eval()
	nef_output = nef(X[idx])
	if args.class_cond:
		nef_output = nef_output / nef_output.norm(dim=0) * math.sqrt(nef_output.shape[0])
		nef_output *= eigenvalues.sqrt()
		K1 = nef_output.permute(1, 0, 2) @ nef_output.permute(1, 2, 0)
		K1 = torch.block_diag(*K1).view(10, len(idx), 10, len(idx)).permute(1, 0, 3, 2).flatten(0,1).flatten(1)
	else:
		nef_output = nef_output / nef_output.norm(dim=[0, 1]) * math.sqrt(nef_output.shape[0] * nef_output.shape[1])
		nef_output *= eigenvalues.sqrt()
		K1 = nef_output.flatten(0, 1) @ nef_output.flatten(0, 1).T

	F_pred = []
	model = extend(model)
	for class_idx in range(10):
		output = model(X[idx])
		model.zero_grad()
		with backpack(BatchGrad()):
			output[:, class_idx].sum().backward()
		F_pred.append(torch.cat([p.grad_batch.flatten(1) for p in model.parameters()], -1))
	F_pred = torch.stack(F_pred, 1)
	K2 = F_pred.flatten(0, 1) @ F_pred.flatten(0, 1).T

	print(torch.dist(K1, K2))

	print(K1.view(len(idx), 10, len(idx), 10)[0, :, 0, :], K2.view(len(idx), 10, len(idx), 10)[0, :, 0, :])
	exit()

	# explicitly get the Jacobian matrix
	model = extend(model)
	FXT_LambdaX_FX = torch.zeros(num_params, num_params).cuda(non_blocking=True)
	for x, y in train_loader:
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		F_batch = []
		for class_idx in range(10):
			output = model(x)
			model.zero_grad()
			with backpack(BatchGrad()):
				output[:, class_idx].sum().backward()
			F_batch.append(torch.cat([p.grad_batch.flatten(1) for p in model.parameters()], -1))
		with torch.no_grad():
			F_batch = torch.stack(F_batch, 1)
			prob = output.softmax(-1)
			Lamdba_batch = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			FXT_LambdaX_FX += torch.einsum('bkp,bkj,bjq->pq', F_batch, Lamdba_batch, F_batch)
	with torch.no_grad():
		K_F = FXT_LambdaX_FX.add(torch.eye(num_params).cuda() * args.delta).inverse()
	print(K_F[:5, :5], FXT_LambdaX_FX[:5, :5])

	# test on in-distribution data
	test_loss, correct, test_loss_ntkunc, correct_ntkunc = 0, 0, 0, 0
	uncs, confs, ents = [], [], []
	for x, y in test_loader:
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		F_pred = []
		for class_idx in range(10):
			output = model(x)
			model.zero_grad()
			with backpack(BatchGrad()):
				output[:, class_idx].sum().backward()
			F_pred.append(torch.cat([p.grad_batch.flatten(1) for p in model.parameters()], -1))
		with torch.no_grad():
			prob = output.softmax(-1)
			test_loss += F.cross_entropy(prob.log(), y).item() * y.size(0)
			correct += prob.argmax(dim=1).eq(y).sum().item()
			ents.append(ent(prob))
			confs.append(prob.max(-1)[0])

			F_pred = torch.stack(F_pred, 1)
			# Lambda_pred = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			F_var = F_pred @ K_F @ F_pred.permute(0, 2, 1)
			print(F_var[0])
			# model_unc = Lambda_pred @ model_unc @ Lambda_pred
			# F_std = psd_safe_cholesky(F_var)
			# F_samples = (F_std @ torch.randn(F_std.shape[0], F_std.shape[1], args.num_samples_eval, device=F_std.device)).permute(2, 0, 1) + output
			F_samples = torch.distributions.multivariate_normal.MultivariateNormal(output, F_var).sample((args.num_samples_eval,))
			prob = F_samples.softmax(-1).mean(0)
			test_loss_ntkunc += F.cross_entropy(prob.log(), y).item() * y.size(0)
			correct_ntkunc += prob.argmax(dim=1).eq(y).sum().item()
			uncs.append(ent(prob))

			# unc = (model_unc + 1e-8*torch.eye(model_unc.shape[-1]).cuda()).logdet() #torch.linalg.slogdet
			# uncs.append(unc)

	uncs, confs, ents = torch.cat(uncs), torch.cat(confs), torch.cat(ents)
	uncs[torch.isnan(uncs)] = uncs[~torch.isnan(uncs)].min()
	print(uncs.max(), uncs.min(), confs.max(), confs.min(), ents.max(), ents.min())
	test_loss /= len(test_loader.dataset)
	top1 = float(correct) / len(test_loader.dataset)
	test_loss_ntkunc /= len(test_loader.dataset)
	top1_ntkunc = float(correct_ntkunc) / len(test_loader.dataset)
	print('\tTest set: Average loss: {:.4f},'
	      ' Accuracy: {:.4f}\n'
		  '\tTest set: Average loss: {:.4f},'
	  	  ' Accuracy: {:.4f}'.format(test_loss, top1, test_loss_ntkunc, top1_ntkunc))

	# test on out-of-distribution data
	ood_dataset = torchvision.datasets.FashionMNIST(args.data_path,
													train=False,
													transform=transforms.Compose([
	 													transforms.ToTensor(),
	 													transforms.Lambda(data_transform),
	 												]),
	 											    download=True)
	# ood_dataset = torchvision.datasets.USPS(args.data_path,
	# 										train=False,
	# 										transform=transforms.Compose([
	# 											transforms.Resize(28),
	# 											transforms.ToTensor(),
	# 											transforms.Lambda(data_transform),
	# 										]),
	# 									    download=True)
	ood_loader = torch.utils.data.DataLoader(ood_dataset,
											 batch_size=args.batch_size,
											 shuffle=False,
											 num_workers=args.workers,
											 pin_memory=True)
	uncs_ood = []
	confs_ood, ents_ood = [], []
	for x, y in ood_loader:
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		F_pred = []
		for class_idx in range(10):
			output = model(x)
			model.zero_grad()
			with backpack(BatchGrad()):
				output[:, class_idx].sum().backward()
			F_pred.append(torch.cat([p.grad_batch.flatten(1) for p in model.parameters()], -1))
		with torch.no_grad():
			prob = output.softmax(-1)
			ents_ood.append(ent(prob))
			confs_ood.append(prob.max(-1)[0])

			F_pred = torch.stack(F_pred, 1)
			# Lambda_pred = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			F_var = F_pred @ K_F @ F_pred.permute(0, 2, 1)
			# model_unc = Lambda_pred @ model_unc @ Lambda_pred
			# F_std = psd_safe_cholesky(F_var)
			# F_samples = (F_std @ torch.randn(F_std.shape[0], F_std.shape[1], args.num_samples_eval, device=F_std.device)).permute(2, 0, 1) + output
			F_samples = torch.distributions.multivariate_normal.MultivariateNormal(output, F_var).sample((args.num_samples_eval,))
			prob = F_samples.softmax(-1).mean(0)
			uncs_ood.append(ent(prob))

			# F_pred = torch.stack(F_pred, 1)
			# prob = output.softmax(-1)
			# Lambda_pred = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			# model_unc = F_pred @ K_F @ F_pred.permute(0, 2, 1)
			# model_unc = Lambda_pred @ model_unc @ Lambda_pred
			# unc = (model_unc + 1e-8*torch.eye(model_unc.shape[-1]).cuda()).logdet() #torch.linalg.slogdet
			# uncs_ood.append(unc)
			#
			# confs_ood.append(prob.max(-1)[0])
			# ents_ood.append(ent(prob))
	uncs_ood = torch.cat(uncs_ood)
	uncs_ood[torch.isnan(uncs_ood)] = uncs_ood[~torch.isnan(uncs_ood)].min()
	# uncs_ood[torch.isneginf(uncs_ood)] = uncs_ood[~torch.isneginf(uncs_ood)].min()
	confs_ood, ents_ood = torch.cat(confs_ood), torch.cat(ents_ood)
	print(uncs_ood.max(), uncs_ood.min(), confs_ood.max(), confs_ood.min(), ents_ood.max(), ents_ood.min())

	binary_classification_given_uncertainty(uncs,uncs_ood, 'mnist_plots/id_vs_ood_ntkunc.pdf')
	binary_classification_given_uncertainty(confs,confs_ood, 'mnist_plots/id_vs_conf.pdf', reverse=True)
	binary_classification_given_uncertainty(ents,ents_ood, 'mnist_plots/id_vs_ood_ent.pdf')

	exit()


	idx = np.random.choice(X.shape[0], 500, replace=False)

	model = model.module
	model = extend(model)
	model.zero_grad()
	logits = model(X[idx])
	loss = logits.sum()
	with backpack(BatchGrad()):
		loss.backward()
	grad_batch = []
	for name, p in model.named_parameters():
		if p.dim() > 1:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(p)
			grad_batch.append(p.grad_batch.flatten(1) * math.sqrt(args.w_var_r/fan_in))
		else:
			grad_batch.append(p.grad_batch.flatten(1) * math.sqrt(args.b_var_r))
	grad_batch = torch.cat(grad_batch, 1)
	# print(grad_batch @ grad_batch.T)

	print(args.random_dist_type, args.epsilon, torch.dist((samples[:, idx].T @ samples[:, idx]).cuda(), grad_batch @ grad_batch.T))

	exit()



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
		X_projected_by_our = F.normalize(X_projected_by_our, dim=0) * math.sqrt(X_projected_by_our.shape[0])
		X_val_projected_by_our = torch.cat(X_val_projected_by_our).float()
		X_val_projected_by_our = F.normalize(X_val_projected_by_our, dim=0) * math.sqrt(X_val_projected_by_our.shape[0])
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

def validate(val_loader, classifier, nef=None, eigenvalues=None, verbose=True):
	classifier.eval()

	test_loss, correct = 0, 0
	probs, labels, uncs = [], [], []
	with torch.no_grad():
		for data, target in val_loader:
			data = data.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)
			with torch.cuda.amp.autocast():
				output = classifier(data).float()

			if nef is not None:
				pass

			uncs.append(ent(output.softmax(-1)))
			test_loss += F.cross_entropy(output, target).item() * target.size(0)
			correct += output.argmax(dim=1).eq(target).sum().item()
			probs.append(output)
			labels.append(target)

		labels = torch.cat(labels)
		probs = torch.cat(probs).softmax(-1)
		uncs = torch.cat(uncs)
		confidences, predictions = torch.max(probs, 1)
		ece_func = _ECELoss().cuda()
		ece = ece_func(confidences, predictions, labels, title=None).item()

	test_loss /= len(val_loader.dataset)
	top1 = float(correct) / len(val_loader.dataset)
	if verbose:
		print('\tTest set: Average loss: {:.4f},'
			  ' Accuracy: {:.4f}, ECE: {:.4f}'.format(test_loss, top1, ece))
	return test_loss, top1, ece, uncs

def ent(p):
	return -(p*p.add(1e-6).log()).sum(-1)

if __name__ == '__main__':
	main()
