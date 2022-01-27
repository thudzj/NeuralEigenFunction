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

import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns


nef_k = 10

npzfile = np.load('eigenvals.npz')
npzfile2 = np.load('reconks.npz')

eigenval_gd = npzfile['gd']
eigenval_nystrom = npzfile['ny']
eigenval_our = npzfile['our']

kernels = list(npzfile2['k'])

# kernels[1] = kernels[0] - kernels[1]
# kernels[2] = kernels[0] - kernels[2]
# kernels[3] = kernels[0] - kernels[3]

def draw_eigenvalues(eigenval_gd, eigenval_nystrom, eigenval_our):
	from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
	from mpl_toolkits.axes_grid1.inset_locator import mark_inset

	fig = plt.figure(figsize=(5, 4.5))
	ax = fig.add_subplot(111)
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	sns.color_palette()

	'''
	ax.plot(list(range(len(eigenval_gd))), eigenval_gd, alpha=1, marker='o', markersize=2, label='Ground truth')
	# ax.plot(list(range(len(eigenval_nystrom))), eigenval_nystrom, '--*', markersize=5, label='The Nyström method')
	ax.plot(list(range(len(eigenval_our))), eigenval_our, ':v', markersize=3, label='Our')
	ax.set_yscale('log')
	ax.set_xlabel('$i$-th eigenvalue', fontsize=16)
	# ax.set_ylabel('Value', fontsize=16)

	ax.spines['bottom'].set_color('gray')
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.spines['left'].set_color('gray')
	ax.set_axisbelow(True)


	axins = ax.inset_axes([0.28, 0.28, 0.7, 0.7]) # zoom = 6
	'''
	axins = ax
	axins.plot(np.array(list(range(10))) + 1, eigenval_gd[:10], marker='o', markersize=5, label='Ground truth')
	# axins.plot(np.array(list(range(10))) + 0.05, eigenval_nystrom[:10], '--*', markersize=6)
	axins.plot(np.array(list(range(10))) + 1, eigenval_our[:10], ':v', markersize=6, label='Our method')
	# sub region of the original image
	x1, x2, y1, y2 = 0, 10.5, 0.008, 0.33
	# axins.set_yscale('log')
	axins.set_xlim(x1, x2)
	axins.set_ylim(y1, y2)
	# axins.tick_params(axis='x', labelsize= 8)
	# axins.tick_params(axis='y', labelsize= 8)
	axins.set_xticks(range(10))
	axins.set_xticks(range(10), minor=True)
	axins.set_yticks([0.0, 0.1, 0.2, 0.3])
	axins.set_yticks([0.0, 0.1, 0.2, 0.3], minor=True)

	# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
    #       ncol=3, fancybox=True, shadow=True, fontsize=13)
	ax.set_xlabel('$i$-th eigenvalue', fontsize=16)
	ax.legend(loc='best')

	fig.tight_layout()
	fig.savefig(os.path.join('eigenvalues_{}.pdf'.format(nef_k)), format='pdf', dpi=1000, bbox_inches='tight')

def draw_kernel(kernels):
	labels = ['Ground truth', 'The Nyström method ($k=10$)', 'Our method ($k=10$)', 'Random feature approach ($S=10$)', ]
	ma = 1
	mi = -1

	labels.pop(1)
	kernels.pop(1)

	fig = plt.figure(figsize=(5*len(kernels), 5))
	for i, (k, l) in enumerate(zip(kernels, labels)):
		ax = fig.add_subplot(100 + 10 * len(kernels) + i + 1)
		# diag = np.diag(1. / np.sqrt(np.diag(k)))
		im = ax.imshow(k[:128,:128], cmap='seismic', vmin=mi, vmax=ma)
		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_title(l)

		ax.set_xticks([])
		ax.set_xticks([], minor=True)
		ax.set_yticks([])
		ax.set_yticks([], minor=True)

		# if i == len(kernels) - 1:
		# 	# divider = make_axes_locatable(ax)
		# 	# cax = divider.append_axes("right", size="5%", pad=0.1)
		# 	# fig.colorbar(im, cax=cax, format='%0.1f')
		#
		# 	plt.colorbar(im,fraction=0.046, pad=0.04)

	fig.subplots_adjust(right=0.95)
	cbar_ax = fig.add_axes([0.98, 0.15, 0.01, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	# fig.tight_layout()
	fig.savefig(os.path.join('kernels_{}.pdf'.format(nef_k)), format='pdf', dpi=1000, bbox_inches='tight')


draw_eigenvalues(eigenval_gd, eigenval_nystrom, eigenval_our)
draw_kernel(kernels)
