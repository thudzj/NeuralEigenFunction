'''
CUDA_VISIBLE_DEVICES=1 python f-eigengame-ntk-imagenet.py --resume auto -b 32  --nef-epochs 200 --nef-batch-size 256 --nef-arch resnet18 --nef-share (--nef-riemannian-projection --nef-max-grad-norm 10)
'''

import argparse
import os
import random
import shutil
import time
import copy
import warnings
from tqdm import tqdm
import math
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_in as resnet_in
import torchvision.models as models
import torch.nn.functional as F

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from utils import load_imagenet, nystrom, fuse_bn_recursively, \
	_ECELoss, binary_classification_given_uncertainty, \
	psd_safe_cholesky, rbf_kernel


model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/data/LargeData/Large/ImageNet',
					help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 5e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0', type=int,
					help='GPU id to use.')
parser.add_argument('--num-classes', default=2, type=int)

# for specifying the neural eigenfunctions
# parser.add_argument('--lr-for-estimate-NTK', default=1e-3, type=float)
parser.add_argument('--num-samples', default=4000, type=int)
parser.add_argument('--random-dist-type', default='rademacher', type=str)
parser.add_argument('--epsilon', default=1e-5, type=float, help='epsilon')
parser.add_argument('--nef-resume', default='', type=str)
parser.add_argument('--nef-batch-size', default=128, type=int)
parser.add_argument('--nef-arch', type=str, default='resnet18')
parser.add_argument('--nef-k', default=10, type=int)
parser.add_argument('--nef-lr', default=1e-3, type=float)
parser.add_argument('--nef-momentum', default=0.9, type=float)
parser.add_argument('--nef-optimizer-type', default='Adam', type=str)
parser.add_argument('--nef-epochs', default=20, type=int)
parser.add_argument('--nef-riemannian-projection', action='store_true')
parser.add_argument('--nef-max-grad-norm', default=None, type=float)
parser.add_argument('--nef-num-samples-eval', default=256, type=int)
parser.add_argument('--nef-no-bn', action='store_true')
parser.add_argument('--nef-share', action='store_true')

best_acc1 = 0


def main():
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	main_worker(args.gpu, args)


def main_worker(gpu, args):
	global best_acc1
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	args.save_dir = os.path.join('snapshots', 'in_{}'.format(args.arch))
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# create model
	# if args.pretrained:
	print("=> using pre-trained model '{}'".format(args.arch))
	model = models.__dict__[args.arch](pretrained=True)
	# else:
	# 	print("=> creating model '{}'".format(args.arch))
	# 	model = models.__dict__[args.arch]()

	fc = nn.Linear(model.fc.in_features, args.num_classes if args.num_classes > 2 else 1)
	del model.fc
	model.fc = fc

	if args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(args.gpu) if args.num_classes > 2 \
				else nn.BCEWithLogitsLoss().cuda(args.gpu)

	pretrained, added = [], []
	for n, p in model.named_parameters():
		if 'fc' in n:
			added.append(p)
		else:
			pretrained.append(p)
	print(len(pretrained), len(added))

	optimizer = torch.optim.SGD([{'params': pretrained, 'lr': 1e-3},
								 {'params': added, 'lr': args.lr},],
								 momentum=args.momentum,
								 weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if args.resume == 'auto':
			args.resume = os.path.join(args.save_dir, 'checkpoint.pth.tar')
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
		args.epochs, last_epoch=args.start_epoch if args.start_epoch > 0 else -1)

	cudnn.benchmark = True

	train_loader, train_loader_no_aug, train_loader_no_aug_with_indices, val_loader = load_imagenet(args)

	if args.evaluate:
		validate(val_loader, model, criterion, args)
		return

	for epoch in range(args.start_epoch, args.epochs):

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch, args)
		scheduler.step()

		# evaluate on validation set
		acc1 = validate(val_loader, model, criterion, args)

		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)

		torch.save({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_acc1': best_acc1,
			'optimizer' : optimizer.state_dict(),
		}, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

	assert(args.num_classes == 2)

	model = fuse_bn_recursively(model)
	num_params = sum(p.numel() for p in model.parameters())
	print("Number of parameters:", num_params)
	validate(val_loader, model, criterion, args)

	# calculate ground truth
	bp_model = extend(copy.deepcopy(model))
	Jacobian, Jacobian_val = [], []
	for (images, _) in tqdm(train_loader_no_aug, desc = 'Calc Jacobian for training data'):
		output = bp_model(images.cuda(args.gpu))
		bp_model.zero_grad()
		with backpack(BatchGrad()):
			output.sum().backward()
		Jacobian.append(torch.cat([p.grad_batch.flatten(1) for p in bp_model.parameters()], -1).cpu())
		if len(Jacobian) == 4:
			break

	for (images, _) in tqdm(val_loader, desc = 'Calc Jacobian for validation data'):
		output = bp_model(images.cuda(args.gpu))
		bp_model.zero_grad()
		with backpack(BatchGrad()):
			output.sum().backward()
		Jacobian_val.append(torch.cat([p.grad_batch.flatten(1) for p in bp_model.parameters()], -1).cpu())

	Jacobian, Jacobian_val = torch.cat(Jacobian), torch.cat(Jacobian_val) #/math.sqrt(num_params) /math.sqrt(num_params)
	ground_truth_NTK, ground_truth_NTK_val = Jacobian[:100] @ Jacobian[:100].T, Jacobian_val @ Jacobian_val.T
	print("---------", 'ground truth NTK on training data', "---------")
	print(ground_truth_NTK[:10, :10].data.numpy())
	print("---------", 'ground truth NTK on validation data', "---------")
	print(ground_truth_NTK_val[:10, :10].data.numpy())


	# prepare
	all_images, all_targets = [], []
	for images, target in train_loader_no_aug:
		all_images.append(images.cuda(args.gpu, non_blocking=True))
		all_targets.append(target.cuda(args.gpu, non_blocking=True))
	all_images = torch.cat(all_images)
	all_targets = torch.cat(all_targets).float()
	logits = logit(all_images, model, args)
	new_model = copy.deepcopy(model)


	# estimate NTK via SGD
	# NTK_by_SGD = []
	# for i in tqdm(range(len(all_images)),
	# 			  desc = 'Calc Jacobian for training data by SGD'):
	# 	output = model(all_images[i:i+1]).squeeze()
	# 	with torch.no_grad():
	# 		dl_df = output.sigmoid() - all_targets[i]
	# 	model.zero_grad()
	# 	output.backward(dl_df)
	# 	for p, p_ in zip(new_model.parameters(), model.parameters()):
	# 		p.data.copy_(p_.data).add_(p_.grad, alpha= -args.lr_for_estimate_NTK)
	# 	new_logits = logit(all_images, new_model, args)
	# 	print(logits.max(), logits.min(), new_logits.max(), new_logits.min(), dl_df, all_targets[i], output.sigmoid())
	# 	NTK_by_SGD.append((new_logits - logits) / dl_df / args.lr_for_estimate_NTK)
	# NTK_by_SGD = torch.stack(NTK_by_SGD)
	# print("---------", 'NTK estimated by SGD on training data', "---------")
	# print(NTK_by_SGD[:10, :10].data.cpu().numpy())


	# get mc estimate of the kernel
	if os.path.exists(os.path.join(args.save_dir, 'collected_samples.npz')):
		NTK_samples = np.load(os.path.join(args.save_dir, 'collected_samples.npz'))['arr_0']
		NTK_samples = torch.from_numpy(NTK_samples).float().cuda()
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
			# print(new_logits.max(), new_logits.min(), logits.max(), logits.min())
			NTK_samples.append((new_logits - logits) / args.epsilon)

			# tmp = torch.stack(NTK_samples, 0) / math.sqrt(i+1)
			# print((tmp.T @ tmp)[:10, :10].data.cpu().numpy())
		NTK_samples = torch.stack(NTK_samples, 0)
		print("---------", 'NTK estimated by sampling on training data', "---------")
		print((NTK_samples.T @ NTK_samples / args.num_samples)[:10, :10].data.cpu().numpy())
		np.savez_compressed(os.path.join(args.save_dir, 'collected_samples'),
							NTK_samples.data.cpu().numpy())

	scale_ = ((NTK_samples/math.sqrt(args.num_samples)).norm(dim=0)**2).mean()
	print(args.random_dist_type, args.epsilon, scale_, torch.dist(ground_truth_NTK[:100, :100].cuda()/scale_, (NTK_samples.T @ NTK_samples / args.num_samples)[:100, :100]/scale_))


	nef = NeuralEigenFunctions(args.nef_k, args.nef_arch, args.nef_no_bn, args.nef_share).cuda()
	# nef = NeuralEigenFunctionsDebug(args.nef_k, len(all_images)).cuda();
	# NTK_samples = torch.distributions.MultivariateNormal(torch.zeros(NTK_samples.shape[1], device=NTK_samples.device), scale_tril=psd_safe_cholesky(rbf_kernel(1, 1, torch.empty(NTK_samples.shape[1], 1).cuda().uniform_(-2, 2)))).sample((NTK_samples.shape[0],))

	eigenvalues = train_nef(
		args, nef, NTK_samples/math.sqrt(scale_), train_loader_no_aug_with_indices,
		args.nef_k, args.nef_epochs, args.nef_optimizer_type,
		args.nef_lr, args.nef_momentum,
		args.nef_riemannian_projection,
		args.nef_max_grad_norm,
		train_loader_no_aug, val_loader, ground_truth_NTK_val.cuda()/scale_)

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, arch, no_bn, share, momentum=0.9, normalize_over=[0,]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.functions = nn.ModuleList()
		num_models = 1 if share else k
		out_dim = k if share else 1
		for i in range(num_models):
			if arch in models.__dict__:
				model = models.__dict__[arch](pretrained=True)
			else:
				model = resnet_in.__dict__[arch]()
				print(model)
			if hasattr(model, 'fc'):
				fc = nn.Linear(model.fc.in_features, out_dim)
				del model.fc
				model.fc = fc
			elif hasattr(model, 'classifier'):
				classifier = nn.Linear(model.classifier[0].in_features, out_dim)
				del model.classifier
				model.classifier = classifier
			else:
				raise NotImplementedError
			self.functions.append(fuse_bn_recursively(model) if no_bn else model)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = torch.stack([f(x).squeeze() for f in self.functions], -1).squeeze()
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

class NeuralEigenFunctionsDebug(nn.Module):
	def __init__(self, k, l, momentum=0.9, normalize_over=[0,]):
		super(NeuralEigenFunctionsDebug, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.eigenfuncs = nn.Parameter(torch.randn(l, k))
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = self.eigenfuncs[x]
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

def train_nef(args, nef, collected_samples, train_loader,
			  k, epochs, optimizer_type,
			  lr, momentum,
			  riemannian_projection,
			  max_grad_norm,
			  train_loader_no_aug, val_loader, ground_truth_NTK_val):

	num_samples = collected_samples.shape[0]
	print(collected_samples.shape) # 1000*5000

	if optimizer_type == 'Adam':
		optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
	elif optimizer_type == 'RMSprop':
		optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
	else:
		optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)

	eigenvalues = None
	start_epoch = 0
	if args.nef_resume:
		if os.path.isfile(args.nef_resume):
			print("=> loading checkpoint '{}'".format(args.nef_resume))
			checkpoint = torch.load(args.nef_resume, map_location='cpu')
			start_epoch = checkpoint['epoch']
			nef.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			eigenvalues = checkpoint['eigenvalues'].cuda()
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.nef_resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.nef_resume))
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, last_epoch=start_epoch - 1)

	for epoch in tqdm(range(start_epoch, epochs), desc="Training NEF"):
		nef.train()
		for i, (data, _, indices) in enumerate(train_loader):
			psis_X = nef(data.cuda())
			with torch.no_grad():
				samples_batch = collected_samples[:, indices]
				samples_batch_psis = samples_batch @ psis_X
				psis_K_psis = samples_batch_psis.T @ samples_batch_psis / num_samples

				cur_eigenvalues = psis_K_psis.diag()
				mask = - (psis_K_psis / cur_eigenvalues).tril(diagonal=-1).T
				mask += torch.eye(k, device=psis_X.device)
				mask /= num_samples
				grad = samples_batch.T @ (samples_batch_psis @ mask)

				cur_eigenvalues /= data.shape[0]**2
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
			psis_X.backward(-grad)
			optimizer.step()

		scheduler.step()
		nef.eval()
		with torch.no_grad():
			nef_output = torch.cat([nef(data.cuda()) for (data, _) in train_loader_no_aug]) * eigenvalues.sqrt()
			NTK_train_our = nef_output @ nef_output.T
			nef_output = torch.cat([nef(data.cuda()) for (data, _) in val_loader]) * eigenvalues.sqrt()
			NTK_val_our = nef_output @ nef_output.T
			dist_train = torch.dist(collected_samples.T @ collected_samples / num_samples, NTK_train_our)
			dist_val = torch.dist(ground_truth_NTK_val, NTK_val_our)
			print(eigenvalues.data.cpu().numpy(), dist_train, dist_val)
			print(torch.cat([collected_samples[:, :3].T @ collected_samples[:, :3] / num_samples, NTK_train_our[:3, :3],
							 ground_truth_NTK_val[:3, :3], NTK_val_our[:3, :3]], -1).data.cpu().numpy())

		if epoch % 10 == 0 or epoch == epochs - 1:
			ckpt = {'epoch': epoch + 1}
			ckpt['state_dict'] = nef.state_dict()
			ckpt['optimizer'] = optimizer.state_dict()
			ckpt['eigenvalues'] = eigenvalues.data.cpu()
			torch.save(ckpt, os.path.join(args.save_dir,
				'nef_checkpoint_{}.th'.format(epoch)))
	print('\tEigenvalues:', eigenvalues.data.cpu().numpy())
	return eigenvalues

def train(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		images, target = images.cuda(args.gpu, non_blocking=True), \
			target.cuda(args.gpu, non_blocking=True)

		# compute output
		output = model(images)
		loss = criterion(output, target if args.num_classes > 2 else target.unsqueeze(-1).float())

		# measure accuracy and record loss
		acc1 = accuracy(output, target)
		losses.update(loss.item(), images.size(0))
		top1.update(acc1[0], images.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i)


def validate(val_loader, model, criterion, args):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(val_loader):
			images, target = images.cuda(args.gpu, non_blocking=True), \
				target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(images)
			loss = criterion(output, target if args.num_classes > 2 else target.unsqueeze(-1).float())

			# measure accuracy and record loss
			acc1 = accuracy(output, target)
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.display(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

	return top1.avg


def logit(all_images, model, args):
	res = []
	for i in range(0, len(all_images), 256):
		# with torch.cuda.amp.autocast():
		with torch.no_grad():
			res.append(model(all_images[i: min(i+256, len(all_images))]).view(-1))
	return torch.cat(res)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		if output.shape[1] == 1:
			batch_size = target.size(0)
			pred = output.squeeze() > 0
			correct = pred.eq(target).float().sum()
			res = [correct.mul_(100.0 / batch_size)]
		else:
			maxk = max(topk)
			batch_size = target.size(0)

			_, pred = output.topk(maxk, 1, True, True)
			pred = pred.t()
			correct = pred.eq(target.view(1, -1).expand_as(pred))

			res = []
			for k in topk:
				correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	main()
