'''
CUDA_VISIBLE_DEVICES=0 python f-eigengame-ntk-cifar.py --classes 0 1 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --job-id resnet20-ntk-bs256-ip32  --ood-classes 8 9 --draw-eigenvalues --resume auto (--nef-resume snapshots//resnet20-ntk-bs256-ip32/nef_checkpoint_199.th)
	Clustering acc on in-dis. validation data 0.678
	Clustering acc given clf features on in-dis. validation data 0.9895
	Clustering acc given eigen projections on in-dis. validation data 0.9745
	Clustering acc on ood validation data 0.504
	Clustering acc given clf features on ood validation data 0.979
	Clustering acc given eigen projections on ood validation data 0.795


CUDA_VISIBLE_DEVICES=7 python f-eigengame-ntk-cifar.py --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --job-id resnet20-ntk-bs256-ip32-10cls --ntk-std-scale 20 (--nef-resume snapshots/resnet20-ntk-bs256-ip32-10cls/nef_checkpoint_199.th)
	Number of parameters: 269034
	        Test set: Average loss: 0.3568, Accuracy: 0.9193
	/data/zhijie/NeuralEigenFunction/utils.py:72: RuntimeWarning: A not p.d., added jitter of 1e-06 to the diagonal
	  RuntimeWarning,
	        Test set: Average loss: 0.3567, Accuracy: 0.9193  ECE: 0.0487
	        Test set: Average loss: 0.2767, Accuracy: 0.9200  ECE: 0.0162
'''
import argparse
import os
import shutil
import copy
import time
import math
import random
from contextlib import suppress
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.utils.data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.utils import AverageMeter

from backpack import backpack, extend
from backpack.extensions import BatchGrad

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

from utils import _ECELoss, time_string, convert_secs2time, dataset_with_indices, \
	fuse_bn_recursively, psd_safe_cholesky, binary_classification_given_uncertainty
from models.resnet import *
from models.wide_resnet import *

parser = argparse.ArgumentParser(description='SGD training on CIFAR')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='dataset to use (default: cifar10)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='./snapshots', type=str)
parser.add_argument('--data-dir', dest='data_dir',
					help='The directory saving the data',
					default='/data/LargeData/Regular/cifar', type=str)
parser.add_argument('--job-id', default='default-ntk', type=str)

# for specifying the classifier
parser.add_argument('--epochs', default=20, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--nesterov', action='store_true',
					help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--clf-arch', type=str, default='resnet20')
parser.add_argument('--clf-in-planes', type=int, default=16)
parser.add_argument('--classes', default=None, type=int, nargs='+')
parser.add_argument('--ood-classes', default=None, type=int, nargs='+')

# for specifying the neural eigenfunctions
parser.add_argument('--num-samples', default=4000, type=int)
parser.add_argument('--random-dist-type', default='rademacher', type=str)
parser.add_argument('--epsilon', default=1e-5, type=float, help='epsilon')
parser.add_argument('--nef-resume', default='', type=str)
parser.add_argument('--nef-batch-size', default=128, type=int)
parser.add_argument('--nef-arch', type=str, default='resnet20')
parser.add_argument('--nef-in-planes', type=int, default=16)
parser.add_argument('--nef-k', default=10, type=int)
parser.add_argument('--nef-lr', default=1e-3, type=float)
parser.add_argument('--nef-momentum', default=0.9, type=float)
parser.add_argument('--nef-optimizer-type', default='Adam', type=str)
parser.add_argument('--nef-epochs', default=100, type=int)
parser.add_argument('--nef-riemannian-projection', action='store_true')
parser.add_argument('--nef-max-grad-norm', default=None, type=float)
parser.add_argument('--nef-num-samples-eval', default=256, type=int)
parser.add_argument('--nef-no-bn', action='store_true')
parser.add_argument('--nef-share', action='store_true')
parser.add_argument('--nef-amp', action='store_true')
parser.add_argument('--draw-eigenvalues', action='store_true')
parser.add_argument('--delta', default=5, type=float, help='delta') # # of data * weight_decay = 50000 * 1e-4 = 5
parser.add_argument('--ntk-std-scale', default=1, type=float)

def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.job_id)

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	if args.classes is None:
		args.num_classes = 10 if args.dataset == 'cifar10' else 100
		args.classes = list(range(args.num_classes))
	else:
		args.num_classes = len(args.classes)
		assert args.num_classes == 2

	if args.ood_classes is None:
		args.ood_classes = list(range(10 if args.dataset == 'cifar10' else 100))

	train_loader, nef_train_loader, nef_train_val_loader, val_loader, val_loader_ood = load_cifar(args)

	classifier = eval(args.clf_arch)(args.clf_in_planes, 10 if args.dataset == 'cifar10' else 100)
	# load pre-trained ckpt
	checkpoint = torch.load('snapshots/{}-cc-swalr0.1/checkpoint_150.th'.format(args.clf_arch), map_location='cpu') # to 150
	classifier.load_state_dict(checkpoint['state_dict'])

	if args.num_classes == 2:
		finetune_binary_classifier(args, classifier, train_loader, val_loader)
	else:
		classifier.cuda()

	classifier = fuse_bn_recursively(classifier)
	num_params = sum(p.numel() for p in classifier.parameters())
	print("Number of parameters:", num_params)
	validate(args, val_loader, classifier)

	# ground_truth_NTK, ground_truth_NTK_val = get_ground_truth_ntk(args, classifier, nef_train_val_loader, val_loader)
	# NTK_samples = sample_from_ntk(args, classifier, nef_train_val_loader)
	# print("---------", 'ground truth NTK on training data', "---------")
	# print(ground_truth_NTK[:10, :10].data.numpy())
	# print("---------", 'NTK estimated by sampling on training data', "---------")
	# print((NTK_samples[:, :10].T @ NTK_samples[:, :10] / args.num_samples).data.cpu().numpy())
	#
	# scale_ = ((NTK_samples/math.sqrt(args.num_samples)).norm(dim=0)**2).mean().item()
	# NTK_samples /= math.sqrt(scale_)
	# ground_truth_NTK /= scale_
	# ground_truth_NTK_val /= scale_
	#
	# print('Distance between gd NTK and estimated NTK: noise {}, eps {}, scale {}, dist {}'.format(
	# 	args.random_dist_type, args.epsilon, scale_,
	# 	torch.dist(ground_truth_NTK[:100, :100],
	# 			   NTK_samples[:, :100].T @ NTK_samples[:, :100] / args.num_samples).item()))

	nef = NeuralEigenFunctions(args.nef_k, args.nef_arch, args.nef_in_planes, args.num_classes, args.nef_no_bn, args.nef_share).cuda()
	# eigenvalues = train_nef(
	# 	args, nef, NTK_samples, nef_train_loader,
	# 	args.nef_k, args.nef_epochs, args.nef_optimizer_type,
	# 	args.nef_lr, args.nef_momentum,
	# 	args.nef_riemannian_projection,
	# 	args.nef_max_grad_norm, args.nef_amp,
	# 	nef_train_val_loader, val_loader, ground_truth_NTK_val)
	#
	# if args.draw_eigenvalues:
	# 	draw_eigenvalues(args, eigenvalues, NTK_samples)

	nef.load_state_dict(torch.load(args.nef_resume, map_location='cpu')['state_dict'])
	eigenvalues = torch.load(args.nef_resume, map_location='cpu')['eigenvalues'].cuda()

	if args.num_classes == 2:
		clustering(args, classifier, nef, eigenvalues, val_loader, val_loader_ood)
	else:
		ntkgp_validate(args, classifier, nef, eigenvalues, nef_train_loader, val_loader)

def train_nef(args, nef, collected_samples, train_loader,
			  k, epochs, optimizer_type, lr,
			  momentum, riemannian_projection,
			  max_grad_norm, amp, nef_train_val_loader, val_loader, ground_truth_NTK_val):

	num_samples = collected_samples.shape[0]
	print(collected_samples.shape) # 4000*(50000*num_classes)

	if optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
	elif optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)

	if amp:
		amp_autocast = torch.cuda.amp.autocast
		loss_scaler = torch.cuda.amp.GradScaler()
	else:
		amp_autocast = suppress  # do nothing
		loss_scaler = None

	eigenvalues = None
	start_epoch = 0
	if args.nef_resume:
		if os.path.isfile(args.nef_resume):
			print("=> loading checkpoint '{}'".format(args.nef_resume))
			checkpoint = torch.load(args.nef_resume, map_location='cpu')
			start_epoch = checkpoint['epoch']
			nef.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			if loss_scaler is not None:
				loss_scaler.load_state_dict(checkpoint['loss_scaler'])
			eigenvalues = checkpoint['eigenvalues'].cuda()
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.nef_resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.nef_resume))
	scheduler = CosineAnnealingLR(optimizer, epochs, last_epoch=start_epoch - 1)

	for epoch in tqdm(range(start_epoch, epochs), desc="Training NEF"):
		nef.train()
		for i, (data, _, indices) in enumerate(train_loader):

			samples_batch = collected_samples.view(num_samples, -1, args.num_classes if args.num_classes != 2 else 1)[:, indices].flatten(1).cuda(non_blocking=True)
			with amp_autocast():
				psis_X = nef(data.cuda())

			with torch.no_grad():
				samples_batch_psis = samples_batch @ psis_X
				psis_K_psis = samples_batch_psis.T @ samples_batch_psis / num_samples

				cur_eigenvalues = psis_K_psis.diag()
				mask = - (psis_K_psis / cur_eigenvalues).tril(diagonal=-1).T
				mask += torch.eye(k, device=psis_X.device)
				mask /= num_samples
				grad = samples_batch.T @ (samples_batch_psis @ mask)

				cur_eigenvalues /= samples_batch.shape[1]**2
				if eigenvalues is None:
					eigenvalues = cur_eigenvalues
				else:
					eigenvalues.mul_(0.9).add_(cur_eigenvalues, alpha = 0.1)

				if riemannian_projection:
					grad.sub_((psis_X*grad).sum(0) * psis_X / data.shape[0])
				if max_grad_norm is not None:
					clip_coef = max_grad_norm / (grad.norm(dim=0) + 1e-6)
					grad.mul_(clip_coef)

			optimizer.zero_grad()
			if loss_scaler is not None:
				loss_scaler.scale(psis_X).backward(-grad)
				loss_scaler.step(optimizer)
				loss_scaler.update()
			else:
				psis_X.backward(-grad)
				optimizer.step()

		scheduler.step()
		nef.eval()
		with torch.no_grad():
			nef_output = torch.cat([nef(data.cuda()) for (data, _) in nef_train_val_loader]) * eigenvalues.sqrt()
			NTK_train_our = (nef_output[:ground_truth_NTK_val.shape[0]] @ nef_output[:ground_truth_NTK_val.shape[0]].T).cpu()
			nef_output = torch.cat([nef(data.cuda()) for (data, _) in val_loader]) * eigenvalues.sqrt()
			NTK_val_our = (nef_output[:ground_truth_NTK_val.shape[0]] @ nef_output[:ground_truth_NTK_val.shape[0]].T).cpu()
			dist_train = torch.dist(collected_samples[:, :100].T @ collected_samples[:, :100] / num_samples, NTK_train_our[:100, :100])
			dist_val = torch.dist(ground_truth_NTK_val[:100, :100], NTK_val_our[:100, :100])
			print(eigenvalues[:10].data.cpu().numpy(), dist_train, dist_val)
			print(torch.cat([collected_samples[:, :3].T @ collected_samples[:, :3] / num_samples, NTK_train_our[:3, :3],
							 ground_truth_NTK_val[:3, :3], NTK_val_our[:3, :3]], -1).data.cpu().numpy())
			draw_kernel(args, [NTK_val_our.numpy(), ground_truth_NTK_val.numpy()],
						['Our', 'Ground truth'], epoch)

		if epoch % 10 == 0 or epoch == epochs - 1:
			ckpt = {'epoch': epoch + 1}
			ckpt['state_dict'] = nef.state_dict()
			ckpt['optimizer'] = optimizer.state_dict()
			if loss_scaler is not None:
				ckpt['loss_scaler'] = loss_scaler.state_dict()
			ckpt['eigenvalues'] = eigenvalues.data.cpu()
			torch.save(ckpt, os.path.join(args.save_dir,
				'nef_checkpoint_{}.th'.format(epoch)))
	print('\tEigenvalues estimated by ours:', eigenvalues.data.cpu().numpy())
	return eigenvalues

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, arch, in_planes, num_classes, no_bn, share, momentum=0.9, normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.k = k
		self.share = share
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.functions = nn.ModuleList()
		num_models = 1 if share else k
		out_dim = (k if share else 1) * (num_classes if num_classes != 2 else 1)
		for i in range(num_models):
			function = eval(arch)(in_planes, out_dim)
			self.functions.append(function)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		if self.share:
			ret_raw = self.functions[0](x).view(x.shape[0], -1, self.k).flatten(0, 1)
		else:
			ret_raw = torch.stack([f(x) for f in self.functions], -1).flatten(0, 1)
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

def sample_from_ntk(args, model, train_loader):
	all_images = torch.cat([images.cuda(non_blocking=True) for images, _ in train_loader])
	logits = logit(all_images, model, args)
	new_model = copy.deepcopy(model)

	if os.path.exists(os.path.join(args.save_dir, 'collected_samples.npz')):
		NTK_samples = np.load(os.path.join(args.save_dir, 'collected_samples.npz'))['arr_0']
		NTK_samples = torch.from_numpy(NTK_samples).float()
	else:
		NTK_samples = []
		for i in tqdm(range(args.num_samples),
					  desc = 'Sampling from the NTK kernel'):
			for p, p_ in zip(new_model.parameters(), model.parameters()):
				if args.random_dist_type == 'normal':
					perturbation = torch.randn_like(p) * args.epsilon #/ math.sqrt(num_params)
				elif args.random_dist_type == 'rademacher':
					perturbation = torch.randn_like(p).sign() * args.epsilon #/ math.sqrt(num_params)
				else:
					raise NotImplementedError
				p.data.copy_(p_.data).add_(perturbation)
			new_logits = logit(all_images, new_model, args)
			NTK_samples.append(((new_logits - logits) / args.epsilon).cpu())
		NTK_samples = torch.stack(NTK_samples, 0)
		np.savez_compressed(os.path.join(args.save_dir, 'collected_samples'),
							NTK_samples.data.numpy())
	return NTK_samples.flatten(1)

def get_ground_truth_ntk(args, classifier, train_loader, val_loader):
	bp_model = extend(copy.deepcopy(classifier))
	Jacobian, Jacobian_val = [], []
	for (images, _) in tqdm(train_loader, desc = 'Calc Jacobian for training data'):
		images = images.cuda()
		Jacobian_batch = []
		for k in range(args.num_classes if args.num_classes != 2 else 1):
			output = bp_model(images)
			bp_model.zero_grad()
			with backpack(BatchGrad()):
				output[:,k].sum().backward()
			Jacobian_batch.append(torch.cat([p.grad_batch.flatten(1) for p in bp_model.parameters()], -1).cpu())
		Jacobian.append(torch.stack(Jacobian_batch, 1))
		if len(Jacobian) == 4:
			break

	for (images, _) in tqdm(val_loader, desc = 'Calc Jacobian for validation data'):
		images = images.cuda()
		Jacobian_batch = []
		for k in range(args.num_classes if args.num_classes != 2 else 1):
			output = bp_model(images)
			bp_model.zero_grad()
			with backpack(BatchGrad()):
				output[:,k].sum().backward()
			Jacobian_batch.append(torch.cat([p.grad_batch.flatten(1) for p in bp_model.parameters()], -1).cpu())
		Jacobian_val.append(torch.stack(Jacobian_batch, 1))
		if len(Jacobian_val) == 4:
			break
	Jacobian, Jacobian_val = torch.cat(Jacobian), torch.cat(Jacobian_val) #/math.sqrt(num_params) /math.sqrt(num_params)
	ground_truth_NTK, ground_truth_NTK_val = Jacobian[:100].flatten(0,1) @ Jacobian[:100].flatten(0,1).T, Jacobian_val[:100].flatten(0,1) @ Jacobian_val[:100].flatten(0,1).T
	return ground_truth_NTK, ground_truth_NTK_val

def clustering(args, classifier, nef, eigenvalues, val_loader, val_loader_ood):
	nef.eval()
	classifier.eval()
	with torch.no_grad():
		val_data = torch.cat([data.flatten(1) for (data, _) in val_loader])
		val_clf_features = torch.cat([classifier(data.cuda(), True) for (data, _) in val_loader]).cpu()
		val_eigen_projections = (torch.cat([nef(data.cuda()) for (data, _) in val_loader]).cpu() * eigenvalues.sqrt().cpu()).view(val_data.shape[0], -1)
		val_labels = torch.cat([label for (_, label) in val_loader])

		val_ood_data = torch.cat([data.flatten(1) for (data, _) in val_loader_ood])
		val_ood_clf_features = torch.cat([classifier(data.cuda(), True) for (data, _) in val_loader_ood]).cpu()
		val_ood_eigen_projections = (torch.cat([nef(data.cuda()) for (data, _) in val_loader_ood]).cpu() * eigenvalues.sqrt().cpu()).view(val_data.shape[0], -1)
		val_ood_labels = torch.cat([label for (_, label) in val_loader_ood])

	assignment = KMeans(len(args.classes)).fit_predict(val_data)
	preds = assignment2pred(assignment, val_labels, len(args.classes))
	print("Clustering acc on in-dis. validation data", (preds==val_labels.numpy()).astype(np.float32).mean())

	assignment = KMeans(len(args.classes)).fit_predict(val_clf_features)
	preds = assignment2pred(assignment, val_labels, len(args.classes))
	print("Clustering acc given clf features on in-dis. validation data", (preds==val_labels.numpy()).astype(np.float32).mean())

	assignment = KMeans(len(args.classes)).fit_predict(val_eigen_projections)
	preds = assignment2pred(assignment, val_labels, len(args.classes))
	print("Clustering acc given eigen projections on in-dis. validation data", (preds==val_labels.numpy()).astype(np.float32).mean())

	assignment = KMeans(len(args.ood_classes)).fit_predict(val_ood_data)
	preds = assignment2pred(assignment, val_ood_labels, len(args.ood_classes))
	print("Clustering acc on ood validation data", (preds==val_ood_labels.numpy()).astype(np.float32).mean())

	assignment = KMeans(len(args.ood_classes)).fit_predict(val_ood_clf_features)
	preds = assignment2pred(assignment, val_ood_labels, len(args.ood_classes))
	print("Clustering acc given clf features on ood validation data", (preds==val_ood_labels.numpy()).astype(np.float32).mean())

	assignment = KMeans(len(args.ood_classes)).fit_predict(val_ood_eigen_projections)
	preds = assignment2pred(assignment, val_ood_labels, len(args.ood_classes))
	print("Clustering acc given eigen projections on ood validation data", (preds==val_ood_labels.numpy()).astype(np.float32).mean())

def ntkgp_validate(args, classifier, nef, eigenvalues, nef_train_loader, val_loader):
	nef.eval()
	classifier.eval()

	EXT_LambdaX_EX = torch.zeros(args.nef_k, args.nef_k).cuda(non_blocking=True)
	with torch.no_grad():
		for i, (x, _, _) in enumerate(nef_train_loader):
			x = x.cuda(non_blocking=True)
			output = classifier(x)
			prob = output.softmax(-1)
			Lamdba = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			# print(Lamdba[0])
			E = nef(x).view(x.shape[0], -1, args.nef_k) * eigenvalues.sqrt()
			EXT_LambdaX_EX += torch.einsum('bck,bcj,bjl->kl', E, Lamdba, E)
			# if i == 10:
				# break
		EXT_LambdaX_EX.diagonal().add_(args.delta)
		K_X_inv = EXT_LambdaX_EX.inverse()
	# print(K_X_inv[:5, :5], EXT_LambdaX_EX[:5, :5])

	# test on in-distribution data
	test_loss, correct, test_loss_ntkunc, correct_ntkunc = 0, 0, 0, 0
	uncs, confs, ents = [], [], []
	probs, probs_ntkunc, labels = [], [], []
	with torch.no_grad():
		for x, y in val_loader:
			x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
			labels.append(y)

			output = classifier(x)
			prob = output.softmax(-1)
			probs.append(prob)
			test_loss += F.cross_entropy(prob.log(), y).item() * y.size(0)
			correct += prob.argmax(dim=1).eq(y).sum().item()
			ents.append(ent(prob))
			confs.append(prob.max(-1)[0])

			E = nef(x).view(x.shape[0], -1, args.nef_k) * eigenvalues.sqrt()
			# print(E[0])
			F_var = E @ K_X_inv.unsqueeze(0) @ E.permute(0, 2, 1)
			# print(F_var[0])
			F_samples = (psd_safe_cholesky(F_var) @ torch.randn(F_var.shape[0], F_var.shape[1], args.nef_num_samples_eval, device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + output
			# if y[0] == 0:
			# 	print(F_samples[0, 0, :], output[0])
			# F_samples = torch.distributions.multivariate_normal.MultivariateNormal(output, F_var).sample((args.nef_num_samples_eval,))
			prob = F_samples.softmax(-1).mean(0)
			probs_ntkunc.append(prob)
			test_loss_ntkunc += F.cross_entropy(prob.log(), y).item() * y.size(0)
			correct_ntkunc += prob.argmax(dim=1).eq(y).sum().item()
			uncs.append(ent(prob))

	uncs, confs, ents = torch.cat(uncs), torch.cat(confs), torch.cat(ents)
	uncs[torch.isnan(uncs)] = uncs[~torch.isnan(uncs)].min()
	# print(uncs.max(), uncs.min(), confs.max(), confs.min(), ents.max(), ents.min())
	test_loss /= len(val_loader.dataset)
	top1 = float(correct) / len(val_loader.dataset)
	test_loss_ntkunc /= len(val_loader.dataset)
	top1_ntkunc = float(correct_ntkunc) / len(val_loader.dataset)

	labels, probs, probs_ntkunc = torch.cat(labels), torch.cat(probs), torch.cat(probs_ntkunc)
	confidences, predictions = torch.max(probs, 1)
	confidences_ntkunc, predictions_ntkunc = torch.max(probs_ntkunc, 1)
	ece_func = _ECELoss().cuda()
	ece = ece_func(confidences, predictions, labels,
				   title='cifar_plots/ntk/ece.pdf').item()
	ece_ntkunc = ece_func(confidences_ntkunc, predictions_ntkunc, labels,
				   title='cifar_plots/ntk/ece_ntkunc.pdf').item()

	print('\tTest set: Average loss: {:.4f},'
	      ' Accuracy: {:.4f}  ECE: {:.4f}\n'
		  '\tTest set: Average loss: {:.4f},'
	  	  ' Accuracy: {:.4f}  ECE: {:.4f}'.format(test_loss, top1, ece, test_loss_ntkunc, top1_ntkunc, ece_ntkunc))

	# test on out-of-distribution data
	ood_loader = torch.utils.data.DataLoader(
		torchvision.datasets.SVHN(root='/data/LargeData/Regular/svhn', split='test',
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]), download=True),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
	uncs_ood, confs_ood, ents_ood = [], [],  []
	with torch.no_grad():
		for x, _ in ood_loader:
			x = x.cuda(non_blocking=True)
			output = classifier(x)
			prob = output.softmax(-1)
			ents_ood.append(ent(prob))
			confs_ood.append(prob.max(-1)[0])

			E = nef(x).view(x.shape[0], -1, args.nef_k) * eigenvalues.sqrt()
			F_var = E @ K_X_inv @ E.permute(0, 2, 1)
			F_samples = (psd_safe_cholesky(F_var) @ torch.randn(F_var.shape[0], F_var.shape[1], args.nef_num_samples_eval, device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + output
			# F_samples = torch.distributions.multivariate_normal.MultivariateNormal(output, F_var).sample((args.nef_num_samples_eval,))
			prob = F_samples.softmax(-1).mean(0)
			uncs_ood.append(ent(prob))

	uncs_ood, confs_ood, ents_ood = torch.cat(uncs_ood), torch.cat(confs_ood), torch.cat(ents_ood)
	uncs_ood[torch.isnan(uncs_ood)] = uncs_ood[~torch.isnan(uncs_ood)].min()
	# uncs_ood[torch.isneginf(uncs_ood)] = uncs_ood[~torch.isneginf(uncs_ood)].min()
	# print(uncs_ood.max(), uncs_ood.min(), confs_ood.max(), confs_ood.min(), ents_ood.max(), ents_ood.min())

	binary_classification_given_uncertainty(uncs,uncs_ood, 'cifar_plots/ntk/id_vs_ood_ntkunc.pdf')
	binary_classification_given_uncertainty(confs,confs_ood, 'cifar_plots/ntk/id_vs_conf.pdf', reverse=True)
	binary_classification_given_uncertainty(ents,ents_ood, 'cifar_plots/ntk/id_vs_ood_ent.pdf')


def finetune_binary_classifier(args, classifier, train_loader, val_loader):
	best_prec1 = 0

	if hasattr(classifier, 'fc'):
		fc = nn.Linear(classifier.fc.in_features, 1)
		del classifier.fc
		classifier.fc = fc
	elif hasattr(classifier, 'linear'):
		linear = nn.Linear(classifier.linear.in_features, 1)
		del classifier.linear
		classifier.linear = linear
	else:
		raise NotImplementedError

	classifier.cuda()

	# define optimizer
	pretrained, added = [], []
	for n, p in classifier.named_parameters():
		if 'fc' in n or 'linear' in n:
			added.append(p)
		else:
			pretrained.append(p)
	print(len(pretrained), len(added))
	optimizer = torch.optim.SGD([{'params': pretrained, 'lr': 1e-3},
								 {'params': added, 'lr': args.lr},],
								 momentum=args.momentum,
								 weight_decay=args.weight_decay,
								 nesterov = args.nesterov)

	# optionally resume from a checkpoint
	if args.resume:
		if args.resume == 'auto':
			args.resume = os.path.join(args.save_dir, 'checkpoint_{}.th'.format(args.epochs - 1))
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cpu')
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			classifier.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {} acc {})"
				  .format(args.resume, checkpoint['epoch'], checkpoint['prec1']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
		args.epochs, last_epoch=args.start_epoch - 1)

	start_time = time.time()
	epoch_time = AverageMeter()
	for epoch in range(args.start_epoch, args.epochs):

		need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
		need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
		print('==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(
									time_string(), epoch, args.epochs, need_time) \
					+ ' [Best : Accuracy={:.4f}]'.format(best_prec1))
		# train for one epoch
		train_classifier_one_epoch(args, train_loader,
								   classifier, optimizer, epoch)
		scheduler.step()

		# evaluate on validation set
		_, prec1 = validate(args, val_loader, classifier)
		best_prec1 = max(prec1, best_prec1)

		if epoch % 10 == 0 or epoch == args.epochs - 1:
			ckpt = {'epoch': epoch + 1, 'best_prec1': best_prec1, 'prec1': prec1}
			ckpt['state_dict'] = classifier.state_dict()
			ckpt['optimizer'] = optimizer.state_dict()
			torch.save(ckpt, os.path.join(args.save_dir, 'checkpoint_{}.th'.format(epoch)))

		epoch_time.update(time.time() - start_time)
		start_time = time.time()

def train_classifier_one_epoch(args, train_loader, classifier, optimizer, epoch):
	batch_time, data_time = AverageMeter(), AverageMeter()
	losses, top1 = AverageMeter(), AverageMeter()

	classifier.train()
	end = time.time()
	for i, (data, label) in enumerate(train_loader):
		data_time.update(time.time() - end)
		data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

		output = classifier(data)
		if args.num_classes == 2:
			loss = F.binary_cross_entropy_with_logits(output, label.unsqueeze(-1).float())
			acc = ((output > 0).float().squeeze() == label).float().mean()
		else:
			loss = F.cross_entropy(output, label)
			acc = output.argmax(dim=1).eq(label).float().mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), label.size(0))
		top1.update(acc.item(), label.size(0))

		batch_time.update(time.time() - end)
		end = time.time()

	print('\tLr: {lr:.4f}, '
		  'Time {batch_time.avg:.3f}, '
		  'Data {data_time.avg:.3f}, '
		  'Loss {loss.avg:.4f}, '
		  'Prec@1 {top1.avg:.4f}'.format(lr=optimizer.param_groups[0]['lr'],
			  batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

def validate(args, val_loader, classifier, verbose=True):
	classifier.eval()

	test_loss, correct = 0, 0
	with torch.no_grad():
		for data, target in val_loader:
			data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
			with torch.cuda.amp.autocast():
				output = classifier(data).float()

			if args.num_classes == 2:
				test_loss += F.binary_cross_entropy_with_logits(output, target.unsqueeze(-1).float()).item() * target.size(0)
				correct += ((output > 0).float().squeeze() == target).float().sum().item()
			else:
				test_loss += F.cross_entropy(output, target).item() * target.size(0)
				correct += output.argmax(dim=1).eq(target).sum().item()

	test_loss /= len(val_loader.dataset)
	top1 = float(correct) / len(val_loader.dataset)
	if verbose:
		print('\tTest set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, top1))
	return test_loss, top1

def load_cifar(args):
	if args.dataset == 'cifar10':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		dataset = torchvision.datasets.CIFAR10
	elif args.dataset == 'cifar100':
		mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
		dataset = torchvision.datasets.CIFAR100

	normalize = transforms.Normalize(mean=mean, std=std)

	train_dataset = dataset(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]), download=True)
	idx = sum((np.array(train_dataset.targets) == c).astype(np.int8) for c in args.classes) > 0
	train_dataset.data = train_dataset.data[idx]
	train_dataset.targets = [train_dataset.targets[i] for i, j in enumerate(idx) if j]
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	nef_train_dataset = dataset_with_indices(dataset)(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True)
	idx = sum((np.array(nef_train_dataset.targets) == c).astype(np.int8) for c in args.classes) > 0
	nef_train_dataset.data = nef_train_dataset.data[idx]
	nef_train_dataset.targets = [nef_train_dataset.targets[i] for i, j in enumerate(idx) if j]
	nef_train_loader = torch.utils.data.DataLoader(
		nef_train_dataset,
		batch_size=args.nef_batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	nef_train_val_dataset = dataset(root=args.data_dir, train=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True)
	idx = sum((np.array(nef_train_val_dataset.targets) == c).astype(np.int8) for c in args.classes) > 0
	nef_train_val_dataset.data = nef_train_val_dataset.data[idx]
	nef_train_val_dataset.targets = [nef_train_val_dataset.targets[i] for i, j in enumerate(idx) if j]
	nef_train_val_loader = torch.utils.data.DataLoader(
		nef_train_val_dataset,
		batch_size=args.nef_batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	val_dataset = dataset(root=args.data_dir, train=False,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True)
	idx = sum((np.array(val_dataset.targets) == c).astype(np.int8) for c in args.classes) > 0
	val_dataset.data = val_dataset.data[idx]
	val_dataset.targets = [val_dataset.targets[i] for i, j in enumerate(idx) if j]
	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	val_dataset_ood = dataset(root=args.data_dir, train=False,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]), download=True)
	idx = sum((np.array(val_dataset_ood.targets) == c).astype(np.int8) for c in args.ood_classes) > 0
	val_dataset_ood.data = val_dataset_ood.data[idx]
	val_dataset_ood.targets = [val_dataset_ood.targets[i] for i, j in enumerate(idx) if j]
	val_loader_ood = torch.utils.data.DataLoader(
		val_dataset_ood,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	return train_loader, nef_train_loader, nef_train_val_loader, val_loader, val_loader_ood

def draw_eigenvalues(args, eigenval_our, collected_samples):
	num_samples = collected_samples.shape[0]
	k = args.nef_k
	K = collected_samples.T @ collected_samples / num_samples
	# eigenval_gd, eigenvec_gd = torch.symeig(K[:1000, :1000], eigenvectors=True)
	eigenval_gd = torch.symeig(K, eigenvectors=False)[0]
	eigenval_gd = eigenval_gd[range(len(eigenval_gd)-1, -1, -1)] / collected_samples.shape[1]
	# projection_gd = eigenvec_gd[:, range(-1, -(k+1), -1)] * math.sqrt(collected_samples.shape[1]) * eigenval_gd[:k].sqrt()
	# print(torch.dist(K, projection_gd @ projection_gd.T))
	print('Ground truth top 10 eigenvalues', eigenval_gd[:10].data.cpu().numpy())

	fig = plt.figure(figsize=(5, 4.5))
	ax = fig.add_subplot(111)
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	sns.color_palette()
	ax.plot(list(range(len(eigenval_gd[:3000]))), eigenval_gd[:3000].data.cpu().numpy(), alpha=1, label='Ground truth')
	ax.plot(list(range(len(eigenval_our))), eigenval_our.data.cpu().numpy(), alpha=1, label='Our')
	ax.set_yscale('log')
	ax.set_xlabel('k', fontsize=16)
	ax.set_ylabel('Eigenvalue', fontsize=16)

	ax.spines['bottom'].set_color('gray')
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.spines['left'].set_color('gray')
	ax.set_axisbelow(True)
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(args.save_dir, 'eigenvalues_{}.pdf'.format(args.nef_k)), format='pdf', dpi=1000, bbox_inches='tight')

def draw_kernel(args, kernels, labels, epoch):
	ma = 2
	mi = -1

	fig = plt.figure(figsize=(5*len(kernels), 5))
	for i, (k, l) in enumerate(zip(kernels, labels)):
		ax = fig.add_subplot(100 + 10 * len(kernels) + i + 1)
		im = ax.imshow(k, cmap='inferno', vmin=mi, vmax=ma)
		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_title(l)

		ax.set_xticks([])
		ax.set_xticks([], minor=True)
		ax.set_yticks([])
		ax.set_yticks([], minor=True)

		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.1)

		fig.colorbar(im, cax=cax, format='%0.1f')

	fig.tight_layout()
	fig.savefig(os.path.join(args.save_dir, 'kernels_{}_{}.pdf'.format(args.nef_k, epoch)), format='pdf', dpi=1000, bbox_inches='tight')

def assignment2pred(assignment, labels, num_classes):
	m = {}
	for i in range(num_classes):
		values, counts = np.unique(labels[assignment == i], return_counts=True)
		m[i] = values[np.argmax(counts)]
		# print(i, values, counts, values[np.argmax(counts)])
	pred = np.array([m[i] for i in assignment])
	return pred


def logit(all_images, model, args):
	res = []
	for i in range(0, len(all_images), 256):
		# with torch.cuda.amp.autocast():
		with torch.no_grad():
			res.append(model(all_images[i: min(i+256, len(all_images))]))
	return torch.cat(res)

def ent(p):
	return -(p*p.add(1e-6).log()).sum(-1)

if __name__ == '__main__':
	main()
