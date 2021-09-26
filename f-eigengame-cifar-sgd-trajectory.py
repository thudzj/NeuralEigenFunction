'''
[3441, 50000, 10]


CUDA_VISIBLE_DEVICES=0 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet20-cc-swalr0.1 --clf-arch resnet20 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --nef-class-cond --swa-lr 0.1 --pre-trained-dir snapshots/resnet20-cc-swalr0.1
	Performance of classifier
        On in-distribution test data
        Test set: Average loss: 0.3567, Accuracy: 0.9194, ECE: 0.0486
        On out-of-distribution test data
        Test set: Average loss: 10.4488, Accuracy: 0.1091, ECE: 0.7482
        AUROC is 0.8525
	Performance of swa classifier
        On in-distribution test data
        Test set: Average loss: 0.2390, Accuracy: 0.9264, ECE: 0.0254
        On out-of-distribution test data
        Test set: Average loss: 5.7104, Accuracy: 0.0937, ECE: 0.5566
        AUROC is 0.9457
	Performance of swag classifier (CUDA_VISIBLE_DEVICES=6 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet20-swag-swalr0.1 --clf-arch resnet20  --swa-lr 0.1 --resume snapshots/resnet20-cc-swalr0.1/checkpoint_150.th --swag --nef-collect-freq 1000000)
        On in-distribution test data
        Test set: Average loss: 0.2247, Accuracy: 0.9256, ECE: 0.0071
        On out-of-distribution test data
        Test set: Average loss: 4.7835, Accuracy: 0.0998, ECE: 0.4787
        AUROC is 0.9528
	Performance of classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.2956, Accuracy: 0.9196, ECE: 0.0291
        On out-of-distribution test data
        Test set: Average loss: 9.9792, Accuracy: 0.1093, ECE: 0.7136
        AUROC is 0.8143
	Performance of swa classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.2229, Accuracy: 0.9256, ECE: 0.0079
        On out-of-distribution test data
        Test set: Average loss: 5.4191, Accuracy: 0.0937, ECE: 0.5065
        AUROC is 0.9332

CUDA_VISIBLE_DEVICES=2 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet32-cc-swalr0.1 --clf-arch resnet32 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --nef-class-cond --swa-lr 0.1 --pre-trained-dir snapshots/resnet32-cc-swalr0.1
	Performance of classifier
        On in-distribution test data
        Test set: Average loss: 0.3696, Accuracy: 0.9203, ECE: 0.0517
        On out-of-distribution test data
        Test set: Average loss: 10.4330, Accuracy: 0.0908, ECE: 0.7646
        AUROC is 0.8795
	Performance of swa classifier
        On in-distribution test data
        Test set: Average loss: 0.2349, Accuracy: 0.9271, ECE: 0.0265
        On out-of-distribution test data
        Test set: Average loss: 5.8156, Accuracy: 0.0873, ECE: 0.5814
        AUROC is 0.9402
	Performance of swag classifier(CUDA_VISIBLE_DEVICES=7 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet32-swag-swalr0.1 --clf-arch resnet32  --swa-lr 0.1 --resume snapshots/resnet32-cc-swalr0.1/checkpoint_150.th --swag --nef-collect-freq 1000000)
		On in-distribution test data
        Test set: Average loss: 0.2126, Accuracy: 0.9264, ECE: 0.0058
        On out-of-distribution test data
        Test set: Average loss: 5.2045, Accuracy: 0.0897, ECE: 0.5040
        AUROC is 0.9377
	Performance of classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.3081, Accuracy: 0.9214, ECE: 0.0358
        On out-of-distribution test data
        Test set: Average loss: 10.0267, Accuracy: 0.0909, ECE: 0.7360
        AUROC is 0.8492
	Performance of swa classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.2160, Accuracy: 0.9276, ECE: 0.0067
        On out-of-distribution test data
        Test set: Average loss: 5.5620, Accuracy: 0.0874, ECE: 0.5394
        AUROC is 0.9257

CUDA_VISIBLE_DEVICES=3 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet56-cc-swalr0.1 --clf-arch resnet56 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --nef-class-cond --swa-lr 0.1 --pre-trained-dir snapshots/resnet56-cc-swalr0.1
	Performance of classifier
        On in-distribution test data
        Test set: Average loss: 0.3358, Accuracy: 0.9243, ECE: 0.0497
        On out-of-distribution test data
        Test set: Average loss: 9.9513, Accuracy: 0.0904, ECE: 0.7398
        AUROC is 0.9049
	Performance of swa classifier
        On in-distribution test data
        Test set: Average loss: 0.2104, Accuracy: 0.9359, ECE: 0.0263
        On out-of-distribution test data
        Test set: Average loss: 5.1433, Accuracy: 0.0939, ECE: 0.5476
        AUROC is 0.9598
	Performance of swag classifier (CUDA_VISIBLE_DEVICES=6 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet56-swag-swalr0.1 --clf-arch resnet56  --swa-lr 0.1 --resume snapshots/resnet56-cc-swalr0.1/checkpoint_150.th --swag --nef-collect-freq 1000000)
        On in-distribution test data
        Test set: Average loss: 0.1887, Accuracy: 0.9343, ECE: 0.0055
        On out-of-distribution test data
        Test set: Average loss: 4.7064, Accuracy: 0.0945, ECE: 0.4798
        AUROC is 0.9561
	Performance of classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.2711, Accuracy: 0.9241, ECE: 0.0333
        On out-of-distribution test data
        Test set: Average loss: 9.4138, Accuracy: 0.0904, ECE: 0.7002
        AUROC is 0.8806
	Performance of swa classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.1905, Accuracy: 0.9364, ECE: 0.0051
        On out-of-distribution test data
        Test set: Average loss: 4.8565, Accuracy: 0.0935, ECE: 0.4941
        AUROC is 0.9511

CUDA_VISIBLE_DEVICES=4 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet110-cc-swalr0.1 --clf-arch resnet110 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --nef-class-cond --swa-lr 0.1 --pre-trained-dir snapshots/resnet110-cc-swalr0.1
	Performance of classifier
        On in-distribution test data
        Test set: Average loss: 0.3455, Accuracy: 0.9285, ECE: 0.0463
        On out-of-distribution test data
        Test set: Average loss: 13.6674, Accuracy: 0.0908, ECE: 0.8106
        AUROC is 0.8207
	Performance of swa classifier
        On in-distribution test data
        Test set: Average loss: 0.2031, Accuracy: 0.9385, ECE: 0.0245
        On out-of-distribution test data
        Test set: Average loss: 5.6235, Accuracy: 0.0913, ECE: 0.5844
        AUROC is 0.9439
	Performance of swag classifier (CUDA_VISIBLE_DEVICES=7 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id resnet110-swag-swalr0.1 --clf-arch resnet110  --swa-lr 0.1 --resume snapshots/resnet110-cc-swalr0.1/checkpoint_150.th --swag --nef-collect-freq 1000000)
        On in-distribution test data
        Test set: Average loss: 0.1882, Accuracy: 0.9371, ECE: 0.0059
        On out-of-distribution test data
        Test set: Average loss: 4.8265, Accuracy: 0.0894, ECE: 0.5066
        AUROC is 0.9411
	Performance of classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.2873, Accuracy: 0.9288, ECE: 0.0320
        On out-of-distribution test data
        Test set: Average loss: 12.9351, Accuracy: 0.0910, ECE: 0.7796
        AUROC is 0.7860
	Performance of swa classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.1851, Accuracy: 0.9387, ECE: 0.0048
        On out-of-distribution test data
        Test set: Average loss: 5.2531, Accuracy: 0.0914, ECE: 0.5247
        AUROC is 0.9368


--------------- our best swa-lr wide resnet-------------
CUDA_VISIBLE_DEVICES=5 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id wrn1610-cc-swalr0.01 --clf-arch wrn16_10 --milestones 50 90 120 --gamma 0.2 --nesterov --weight-decay 5e-4 --nef-in-planes 32 --nef-batch-size 256 --nef-epochs 200 --nef-amp --nef-class-cond --swa-lr 0.01 --pre-trained-dir snapshots/wrn1610-cc-swalr0.01
	Performance of classifier
        On in-distribution test data
        Test set: Average loss: 0.1559, Accuracy: 0.9562, ECE: 0.0189
        On out-of-distribution test data
        Test set: Average loss: 4.7546, Accuracy: 0.1096, ECE: 0.5952
        AUROC is 0.9457
	Performance of swa classifier
        On in-distribution test data
        Test set: Average loss: 0.1369, Accuracy: 0.9625, ECE: 0.0160
        On out-of-distribution test data
        Test set: Average loss: 4.1903, Accuracy: 0.1258, ECE: 0.5089
        AUROC is 0.9771
	Performance of swag classifier (CUDA_VISIBLE_DEVICES=1 python f-eigengame-cifar-sgd-trajectory.py --data-dir /data/zhijie/data --job-id wrn1610-swag-swalr0.01 --clf-arch wrn16_10 --milestones 50 90 120 --gamma 0.2 --nesterov --weight-decay 5e-4 --swa-lr 0.01 --resume snapshots/wrn1610-cc-swalr0.1/checkpoint_150.th --swag --nef-collect-freq 1000000)
        On in-distribution test data
        Test set: Average loss: 0.1221, Accuracy: 0.9616, ECE: 0.0046
        On out-of-distribution test data
        Test set: Average loss: 3.7301, Accuracy: 0.1230, ECE: 0.4097
        AUROC is 0.9851
	Performance of classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.1455, Accuracy: 0.9560, ECE: 0.0106
        On out-of-distribution test data
        Test set: Average loss: 4.6149, Accuracy: 0.1096, ECE: 0.5701
        AUROC is 0.9395
	Performance of swa classifier w/ nef
        On in-distribution test data
        Test set: Average loss: 0.1266, Accuracy: 0.9625, ECE: 0.0099
        On out-of-distribution test data
        Test set: Average loss: 4.0840, Accuracy: 0.1260, ECE: 0.4817
        AUROC is 0.9766
'''
import argparse
import os
import shutil
import copy
import time
import math
import random
from contextlib import suppress
import numpy as np
np.set_printoptions(precision=4)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from timm.utils import AverageMeter

from utils import load_cifar, _ECELoss, time_string, convert_secs2time,\
	binary_classification_given_uncertainty
from models.resnet import *
from models.wide_resnet import *
from swag import SWAG

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
parser.add_argument('--job-id', default='default', type=str)
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
# 					help='evaluate model on validation set')

# for specifying the classifier
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--milestones', default=[80, 120], type=int, nargs='+',
					help='milestones for decaying learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
					help='learning rate decay rate')
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
parser.add_argument('--swa-start', default=150, type=int)
parser.add_argument('--swa-lr', default=0.05, type=float)
parser.add_argument('--swa-anneal-epochs', default=5, type=int)
parser.add_argument('--clf-arch', type=str, default='resnet20')
parser.add_argument('--clf-in-planes', type=int, default=16)

# for specifying the neural eigenfunctions
parser.add_argument('--pre-trained-dir', default='', type=str)
# parser.add_argument('--nef-collect-start', default=155, type=int)
parser.add_argument('--nef-resume', default='', type=str)
parser.add_argument('--nef-collect-freq', default=5, type=int)
parser.add_argument('--nef-batch-size-collect', default=1000, type=int)
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
parser.add_argument('--nef-class-cond', action='store_true')
parser.add_argument('--nef-amp', action='store_true')

# for baseline (SWAG)
parser.add_argument('--swag', action='store_true')

best_prec1 = 0
collected_samples = []

class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, arch, in_planes, num_classes, momentum=0.9, normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.functions = nn.ModuleList()
		for i in range(k):
			function = eval(arch)(in_planes, num_classes)
			self.functions.append(function)
		self.register_buffer('eigennorm', torch.zeros(num_classes, k) if len(normalize_over) == 1 else torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = torch.stack([f(x) for f in self.functions], -1)
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

def main():
	global collected_samples
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

	train_loader, nef_collect_train_loader, nef_train_loader, val_loader, \
		ood_loader, num_classes = load_cifar(args)
	args.num_classes = num_classes

	classifier = eval(args.clf_arch)(args.clf_in_planes, args.num_classes)
	classifier.cuda()
	swa_classifier = AveragedModel(classifier)
	swag_classifier = SWAG(copy.deepcopy(classifier)) if args.swag else None

	if args.pre_trained_dir:
		checkpoint = torch.load(os.path.join(args.pre_trained_dir,
			'checkpoint_{}.th'.format(args.swa_start)), map_location='cpu')
		classifier.load_state_dict(checkpoint['state_dict'])
		checkpoint = torch.load(os.path.join(args.pre_trained_dir,
			'checkpoint_{}.th'.format(args.epochs-1)), map_location='cpu')
		swa_classifier.load_state_dict(checkpoint['swa_state_dict'])
		collected_samples = np.load(os.path.join(args.pre_trained_dir, 'collected_samples.npz'))['arr_0']
		collected_samples = torch.from_numpy(collected_samples).float()
		# collected_samples=None
	else:
		train_classifier(args, classifier, swa_classifier, swag_classifier,
						 train_loader, nef_collect_train_loader, val_loader)
		collected_samples = torch.stack(collected_samples)
		collected_samples.sub_(collected_samples.mean(0))
		np.savez_compressed(os.path.join(args.save_dir, 'collected_samples'),
							collected_samples.data.cpu().numpy())

		checkpoint = torch.load(os.path.join('/'.join(args.resume.split("/")[:-1])\
												if args.resume else args.save_dir,
											 'checkpoint_{}.th'.format(args.swa_start)),
								map_location='cpu')
		classifier.load_state_dict(checkpoint['state_dict'])

	if args.swag:
		print("Performance of swag classifier")
		print("\tOn in-distribution test data")
		uncs_swagclf = swag_validate(args, train_loader, val_loader, swag_classifier)[-1]
		print("\tOn out-of-distribution test data")
		uncs_ood_swagclf = swag_validate(args, train_loader, ood_loader, swag_classifier)[-1]
		binary_classification_given_uncertainty(uncs_swagclf, uncs_ood_swagclf,
			'cifar_plots/binary_clf_given_unc/swag_clf_{}.pdf'.format(args.job_id))
		eval_corrupted_data(args, 'swag_clf_{}'.format(args.job_id), train_loader, None, swag_classifier)

		exit()

	print("Performance of classifier")
	print("\tOn in-distribution test data")
	uncs_clf = validate(args, val_loader, classifier)[-1]
	print("\tOn out-of-distribution test data")
	uncs_ood_clf = validate(args, ood_loader, classifier)[-1]
	binary_classification_given_uncertainty(uncs_clf, uncs_ood_clf,
		'cifar_plots/binary_clf_given_unc/clf_{}.pdf'.format(args.job_id))
	eval_corrupted_data(args,'clf_{}'.format(args.job_id), None, classifier)

	print("Performance of swa classifier")
	print("\tOn in-distribution test data")
	uncs_swaclf = validate(args, val_loader, swa_classifier)[-1]
	print("\tOn out-of-distribution test data")
	uncs_ood_swaclf = validate(args, ood_loader, swa_classifier)[-1]
	binary_classification_given_uncertainty(uncs_swaclf, uncs_ood_swaclf,
		'cifar_plots/binary_clf_given_unc/swa_clf_{}.pdf'.format(args.job_id))
	eval_corrupted_data(args, 'swa_clf_{}'.format(args.job_id), None, swa_classifier)

	nef = NeuralEigenFunctions(args.nef_k, args.nef_arch,
							   args.nef_in_planes, args.num_classes, normalize_over=[0,] if args.nef_class_cond else [0, 1]).cuda()
	eigenvalues = train_nef(args, nef, collected_samples, classifier, swa_classifier,
			  nef_train_loader, val_loader, ood_loader,
			  args.nef_k, args.nef_epochs, args.nef_optimizer_type, args.nef_lr,
			  args.nef_momentum, args.nef_riemannian_projection,
			  args.nef_max_grad_norm, args.nef_amp, args.num_classes, args.nef_class_cond)

	print("Performance of classifier w/ nef")
	print("\tOn in-distribution test data")
	uncs_clf_nef = validate(args, val_loader, classifier, nef, eigenvalues)[-1]
	print("\tOn out-of-distribution test data")
	uncs_ood_clf_nef = validate(args, ood_loader, classifier, nef, eigenvalues)[-1]
	binary_classification_given_uncertainty(uncs_clf_nef, uncs_ood_clf_nef,
		'cifar_plots/binary_clf_given_unc/nef_clf_{}.pdf'.format(args.job_id))
	eval_corrupted_data(args, 'nef_clf_{}'.format(args.job_id), None,
						classifier, nef=nef, eigenvalues=eigenvalues)

	print("Performance of swa classifier w/ nef")
	print("\tOn in-distribution test data")
	uncs_swaclf_nef = validate(args, val_loader, swa_classifier, nef, eigenvalues)[-1]
	print("\tOn out-of-distribution test data")
	uncs_ood_swaclf_nef = validate(args, ood_loader, swa_classifier, nef, eigenvalues)[-1]
	binary_classification_given_uncertainty(uncs_swaclf_nef, uncs_ood_swaclf_nef,
		'cifar_plots/binary_clf_given_unc/nef_swa_clf_{}.pdf'.format(args.job_id))
	eval_corrupted_data(args, 'nef_swa_clf_{}'.format(args.job_id), None,
						swa_classifier, nef=nef, eigenvalues=eigenvalues)


def train_nef(args, nef, collected_samples, classifier, swa_classifier,
			  train_loader, val_loader, ood_loader,
			  k, epochs, optimizer_type, lr,
			  momentum, riemannian_projection,
			  max_grad_norm, amp, num_classes, class_cond):

	num_samples = collected_samples.shape[0]
	print(collected_samples.shape) # 1000*50000*10

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

	for epoch in range(start_epoch, epochs):
		nef.train()
		for i, (data, _, indices) in enumerate(train_loader):
			data = data.cuda(non_blocking=True)
			samples_batch = collected_samples[:, indices, :].cuda(non_blocking=True)
			with amp_autocast():
				psis_X = nef(data)
			with torch.no_grad():
				if class_cond:
					samples_batch_psis = torch.einsum('nbc,bck->nkc', samples_batch, psis_X)
					psis_K_psis = torch.einsum('nkc,nlc->klc', samples_batch_psis, samples_batch_psis) / num_samples
					psis_K_psis = psis_K_psis.permute(2, 0, 1)

					cur_eigenvalues = torch.diagonal(psis_K_psis, dim1=-2, dim2=-1)
					mask = - (psis_K_psis / cur_eigenvalues.unsqueeze(1)).tril(diagonal=-1).permute(0, 2, 1)
					mask += torch.eye(k, device=psis_X.device)
					mask /= num_samples
					grad = torch.einsum('nkc,ckl->nlc', samples_batch_psis, mask)
					grad = torch.einsum('nbc,nkc->bck', samples_batch, grad)
					cur_eigenvalues /= psis_X.shape[0]**2
				else:
					samples_batch_psis = torch.einsum('nbc,bck->nk', samples_batch, psis_X)
					psis_K_psis = samples_batch_psis.T @ samples_batch_psis / num_samples

					cur_eigenvalues = psis_K_psis.diag()
					mask = - (psis_K_psis / cur_eigenvalues).tril(diagonal=-1).T
					mask += torch.eye(k, device=psis_X.device)
					mask /= num_samples
					grad = samples_batch_psis @ mask
					grad = torch.einsum('nbc,nk->bck', samples_batch, grad)
					cur_eigenvalues /= (psis_X.shape[0]*psis_X.shape[1])**2

				if eigenvalues is None:
					eigenvalues = cur_eigenvalues
				else:
					eigenvalues.mul_(0.9).add_(cur_eigenvalues, alpha = 0.1)

				if riemannian_projection:
					grad.sub_((psis_X*grad).sum(0) * psis_X / (data.shape[0] if class_cond else psis_X.shape[0]*psis_X.shape[1]))
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



		print('Training neural eigenfunctions epoch {}'.format(epoch))
		print('\tEigenvalues:', eigenvalues.data.cpu().numpy())
		scheduler.step()
		nef.eval()
		validate(args, val_loader, swa_classifier, nef, eigenvalues)[-1]

		if epoch % 10 == 0 or epoch == epochs - 1:
			ckpt = {'epoch': epoch + 1}
			ckpt['state_dict'] = nef.state_dict()
			ckpt['optimizer'] = optimizer.state_dict()
			if loss_scaler is not None:
				ckpt['loss_scaler'] = loss_scaler.state_dict()
			ckpt['eigenvalues'] = eigenvalues.data.cpu()
			torch.save(ckpt, os.path.join(args.save_dir,
				'nef_checkpoint_{}.th'.format(epoch)))
	return eigenvalues


def train_classifier(args, classifier, swa_classifier, swag_classifier,
					 train_loader, nef_collect_train_loader, val_loader):
	global best_prec1

	# define optimizer
	optimizer = torch.optim.SGD(classifier.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay,
								nesterov = args.nesterov,)

	# optionally resume from a checkpoint
	if args.resume:
		if args.resume == 'auto':
			args.resume = os.path.join(args.save_dir, 'checkpoint.th')
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cpu')
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			classifier.load_state_dict(checkpoint['state_dict'])
			swa_classifier.load_state_dict(checkpoint['swa_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {} acc {})"
				  .format(args.resume, checkpoint['epoch'], checkpoint['prec1']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	scheduler = MultiStepLR(optimizer, milestones=args.milestones,
							gamma=args.gamma,
							last_epoch=args.start_epoch - 1)
	swa_scheduler = SWALR(optimizer, anneal_strategy="linear",
						  anneal_epochs=args.swa_anneal_epochs,
						  swa_lr=args.swa_lr, last_epoch=args.start_epoch - 1)

	start_time = time.time()
	epoch_time = AverageMeter()
	for epoch in range(args.start_epoch, args.epochs):

		need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
		need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
		print('==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(
									time_string(), epoch, args.epochs, need_time) \
					+ ' [Best : Accuracy={:.4f}]'.format(best_prec1))
		# train for one epoch
		train_classifier_one_epoch(args, train_loader, nef_collect_train_loader,
								   classifier, optimizer, epoch)
		if epoch > args.swa_start:
			swa_classifier.update_parameters(classifier)
			if args.swag:
				swag_classifier.collect_model(classifier)
			swa_scheduler.step()
		else:
			scheduler.step()

		# evaluate on validation set
		_, prec1, _, _ = validate(args, val_loader, classifier)
		best_prec1 = max(prec1, best_prec1)

		if epoch % 10 == 0 or epoch == args.epochs - 1:
			if epoch > args.swa_start:
				update_bn(train_loader, swa_classifier, device=torch.device('cuda'))
				validate(args, val_loader, swa_classifier)
				if args.swag:
					swag_validate(args, train_loader, val_loader, swag_classifier)

			ckpt = {'epoch': epoch + 1, 'best_prec1': best_prec1, 'prec1': prec1}
			ckpt['state_dict'] = classifier.state_dict()
			ckpt['swa_state_dict'] = swa_classifier.state_dict()
			ckpt['optimizer'] = optimizer.state_dict()
			torch.save(ckpt, os.path.join(args.save_dir, 'checkpoint_{}.th'.format(epoch)))

		epoch_time.update(time.time() - start_time)
		start_time = time.time()

def train_classifier_one_epoch(args, train_loader, nef_collect_train_loader,
							   classifier, optimizer, epoch):
	global collected_samples
	batch_time, data_time = AverageMeter(), AverageMeter()
	losses, top1 = AverageMeter(), AverageMeter()

	classifier.train()
	end = time.time()
	for i, (data, label) in enumerate(train_loader):
		data_time.update(time.time() - end)
		data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)

		output = classifier(data)
		loss = F.cross_entropy(output, label)
		acc = output.argmax(dim=1).eq(label).float().mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), label.size(0))
		top1.update(acc.item(), label.size(0))

		if epoch > args.swa_start + args.swa_anneal_epochs:
			if ((epoch - args.swa_start - args.swa_anneal_epochs - 1) *
				len(train_loader) + i) % args.nef_collect_freq == 0:
				one_sample = []
				classifier.eval()
				for data_nef, _ in nef_collect_train_loader:
					data_nef = data_nef.cuda(non_blocking=True)
					with torch.no_grad():
						one_sample.append(classifier(data_nef))
				collected_samples.append(torch.cat(one_sample).cpu())
				classifier.train()

		batch_time.update(time.time() - end)
		end = time.time()

	print('\tLr: {lr:.4f}, '
		  'Time {batch_time.avg:.3f}, '
		  'Data {data_time.avg:.3f}, '
		  'Loss {loss.avg:.4f}, '
		  'Prec@1 {top1.avg:.4f}'.format(lr=optimizer.param_groups[0]['lr'],
			  batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

def validate(args, val_loader, classifier, nef=None, eigenvalues=None, verbose=True):
	classifier.eval()
	if nef is not None:
		nef.eval()

	test_loss, correct = 0, 0
	probs, labels, uncs = [], [], []
	with torch.no_grad():
		for data, target in val_loader:
			data = data.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)
			with torch.cuda.amp.autocast():
				output = classifier(data).float()

			if nef is not None:

				with torch.cuda.amp.autocast():
					nef_output = nef(data)
					noise = torch.randn(args.nef_num_samples_eval, *eigenvalues.shape).cuda() * eigenvalues.sqrt()
					if args.nef_class_cond:
						output = torch.einsum("sck,bck->sbc", noise, nef_output) + output
					else:
						output = torch.einsum("sk,bck->sbc", noise, nef_output) + output

				# uncs.append(ent(output.softmax(-1).mean(0)) - ent(output.softmax(-1)).mean(0))
				output = output.softmax(-1).mean(0).log()

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
		ece = ece_func(confidences, predictions, labels,
					   title='cifar_plots/{}_nef{}.pdf'.format(
							args.job_id, nef is not None)).item()

	test_loss /= len(val_loader.dataset)
	top1 = float(correct) / len(val_loader.dataset)
	if verbose:
		print('\tTest set: Average loss: {:.4f},'
			  ' Accuracy: {:.4f}, ECE: {:.4f}'.format(test_loss, top1, ece))
	return test_loss, top1, ece, uncs

def swag_validate(args, train_loader, val_loader, swag_classifier,
				  num_mc_samples=32, verbose=True):
	outputs, labels = [], []
	for _ in range(num_mc_samples):
		swag_classifier.sample(.5)
		swag_classifier.train()
		update_bn(train_loader, swag_classifier, device=torch.device('cuda'))

		swag_classifier.eval()
		with torch.no_grad():
			one_run_output = []
			for data, target in val_loader:
				data = data.cuda(non_blocking=True)
				if _ == 0:
					labels.append(target.cuda(non_blocking=True))
				with torch.cuda.amp.autocast():
					output = swag_classifier(data).float()
				one_run_output.append(output)
			outputs.append(torch.cat(one_run_output))

	outputs = torch.stack(outputs)
	labels = torch.cat(labels)
	# uncs = ent(outputs.softmax(-1).mean(0)) - ent(outputs.softmax(-1)).mean(0)
	outputs = outputs.softmax(-1).mean(0)
	uncs = ent(outputs)
	confidences, predictions = torch.max(outputs, 1)
	test_loss = F.cross_entropy(outputs.log(), labels).item()
	top1 = outputs.argmax(dim=1).eq(labels).float().mean().item()
	ece_func = _ECELoss().cuda()
	ece = ece_func(confidences, predictions, labels,
				   title='cifar_plots/{}_swag.pdf'.format(args.job_id)).item()
	if verbose:
		print('\tTest set: Average loss: {:.4f},'
			  ' Accuracy: {:.4f}, ECE: {:.4f}'.format(test_loss, top1, ece))
	return test_loss, top1, ece, uncs

def eval_corrupted_data(args, token='default', train_loader=None, classifier=None,
						swag_classifier=None, nef=None, eigenvalues=None):
	# for only cifar10
	data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
	data_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()

	corrupted_data_path = './CIFAR-10-C/CIFAR-10-C'
	corrupted_data_files = os.listdir(corrupted_data_path)
	corrupted_data_files.remove('labels.npy')
	results = np.zeros((5, len(corrupted_data_files), 3))
	labels = torch.from_numpy(np.load(os.path.join(corrupted_data_path, 'labels.npy'))).long()
	for ii, corrupted_data_file in enumerate(corrupted_data_files):
		corrupted_data = np.load(os.path.join(corrupted_data_path, corrupted_data_file))
		for i in range(5):
			# print(corrupted_data_file, i)
			images = torch.from_numpy(corrupted_data[i*10000:(i+1)*10000]).float().permute(0, 3, 1, 2)/255.
			images = (images - data_mean.cpu())/data_std.cpu()
			corrupted_dataset = torch.utils.data.TensorDataset(images, labels[i*10000:(i+1)*10000])
			corrupted_loader = torch.utils.data.DataLoader(corrupted_dataset,
				batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
				pin_memory=False, sampler=None, drop_last=False)
			if swag_classifier is None:
				r1, r2, r3, _ = validate(args, corrupted_loader, classifier,
										 nef, eigenvalues, verbose=False)
			else:
				r1, r2, r3, _ = swag_validate(args, train_loader, corrupted_loader,
											  swag_classifier, verbose=False)
			results[i, ii] = np.array([r1, r2, r3])
	# print(results.mean(1)[:, 2])
	np.save('corrupted_results/npys/{}.npy'.format(token), results)

def ent(p):
	return -(p*p.add(1e-6).log()).sum(-1)

if __name__ == '__main__':
	main()
