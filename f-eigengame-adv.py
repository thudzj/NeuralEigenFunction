import math
from functools import partial
import itertools
from timeit import default_timer as timer
import warnings

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
import pandas as pd
import seaborn as sns

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e

def polynomial_kernel(degree, eta, nu, x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    return ((x1.unsqueeze(1) * x2.unsqueeze(0)).sum(-1) * eta + nu) ** degree

def sigmoid_kernel(eta, nu, x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    return ((x1.unsqueeze(1) * x2.unsqueeze(0)).sum(-1) * eta + nu).tanh()

def cosine_kernel(period, output_scale, length_scale, x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    return (((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1).sqrt() * math.pi / period / length_scale).cos() * output_scale

def rbf_kernel(output_scale, length_scale, x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    return (- ((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1) / 2. / length_scale).exp() * output_scale

def periodic_plus_rbf_kernel(period, output_scale1, length_scale1, output_scale2, length_scale2, x1, x2=None):
    if x2 is None:
        x2 = x1
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    out1 = (- (((x1.unsqueeze(1) - x2.unsqueeze(0)).abs().sum(-1) * math.pi / period).sin() ** 2) * 2. / length_scale1).exp() * output_scale1
    out2 = (- ((x1.unsqueeze(1) - x2.unsqueeze(0))**2).sum(-1) / 2. / length_scale2).exp() * output_scale2
    return out1 + out2


class NeuralEigenFunctions(nn.Module):
    def __init__(self, k, input_size, hidden_size, num_layers, output_size=1, bias=True, nonlinearity=nn.ReLU):
        super(NeuralEigenFunctions, self).__init__()
        self.functions = nn.ModuleList()

        for i in range(k):
            if num_layers == 1:
                function = nn.Sequential(
                    nn.Linear(input_size, output_size, bias=bias))
            else:
                layers = [nn.Linear(input_size, hidden_size, bias=bias),
                          nonlinearity(),
                          nn.Linear(hidden_size, output_size, bias=bias)]
                for _ in range(num_layers - 2):
                    layers.insert(2, nonlinearity())
                    layers.insert(2, nn.Linear(hidden_size, hidden_size, bias=bias))
                function = nn.Sequential(*layers)
            self.functions.append(function)

    def forward(self, x):
        # return torch.cat([f(x) for f in self.functions], 1)
        # out = torch.cat([f(x) for f in self.functions], 1); return out / out.norm(dim=0, keepdim=True).detach() * math.sqrt(x.shape[0])
        return F.normalize(torch.cat([f(x) for f in self.functions], 1), dim=0)*math.sqrt(x.shape[0])

def nystrom(X_for_nystrom, x_dim, x_range, k, kernel):

    # perform nystrom method
    start = timer()
    K_for_nystrom = kernel(X_for_nystrom)
    p, q = torch.symeig(K_for_nystrom, eigenvectors=True)
    eigenvalues_nystrom = p[range(-1, -(k+1), -1)] / X_for_nystrom.shape[0]
    eigenfuncs_nystrom = lambda x: kernel(x, X_for_nystrom) @ q[:, range(-1, -(k+1), -1)] \
                                     / p[range(-1, -(k+1), -1)] * math.sqrt(X_for_nystrom.shape[0])
    end = timer()
    # print("Nystrom method consumes {}s".format(end - start))

    return eigenvalues_nystrom, eigenfuncs_nystrom, end - start

def our(X, x_dim, x_range, k, kernel):
    # hyper-parameters for our
    input_size = x_dim
    hidden_size = 32
    num_layers = 3
    optimizer_type = 'SGD'
    lr = 1e-3
    momentum = 0.9
    nonlinearity = nn.GELU #nn.GELU #nn.Tanh # nn.Sigmoid, nn.ReLU
    riemannian_projection = False
    max_grad_norm = 10.

    num_iterations = 2000
    num_samples = 100
    B = min(128, X.shape[0])

    K = kernel(X)
    dist = MultivariateNormal(torch.zeros(X.shape[0], device=X.device), scale_tril=psd_safe_cholesky(K))
    samples = dist.sample((num_samples,))

    # perform our method
    start = timer()
    nef = NeuralEigenFunctions(k, input_size, hidden_size, num_layers, nonlinearity=nonlinearity)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

    eigenvalues_our = None
    for ite in range(num_iterations):

        idx = np.random.choice(X.shape[0], B, replace=False)
        samples_batch = samples[:, idx]
        X_batch = X[idx]

        psis_X = nef(X_batch)
        with torch.no_grad():
            K_batch_est = samples_batch.T @ samples_batch / num_samples
            psis_K_psis = psis_X.T @ K_batch_est @ psis_X
            grad = K_batch_est @ psis_X @ (torch.eye(k, device=psis_X.device) - (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T)

            if eigenvalues_our is None:
                eigenvalues_our = psis_K_psis.diag() / (B**2)
            else:
                eigenvalues_our.mul_(0.9).add_(psis_K_psis.diag() / (B**2), alpha = 0.1)

            if riemannian_projection:
                grad.sub_((psis_X*grad).sum(0) * psis_X)
            grad.mul_(2. / (B**2))
            clip_coef = max_grad_norm / (grad.norm(dim=0) + 1e-6)
            # tmp = grad.norm(dim=0)
            grad.mul_(clip_coef)
            # if ite % 50 == 0:
            #     print(ite, tmp, grad.norm(dim=0))

        optimizer.zero_grad()
        psis_X.backward(-grad)
        optimizer.step()
        scheduler.step()
    end = timer()
    # print("Our method consumes {}s".format(end - start))
    return eigenvalues_our, nef, end - start

def plot_efs(ax, k, X_val, eigenfuncs_eval_nystrom, eigenfuncs_eval_our=None, k_lines=3, xlim=[-2., 2.], ylim=[-2., 2.]):

    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='y', which='minor', labelsize=12)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='x', which='minor', labelsize=12)

    sns.color_palette()
    for i in range(k_lines):
        data = eigenfuncs_eval_nystrom[:, i] if eigenfuncs_eval_nystrom[:100, i].mean() > 0 else -eigenfuncs_eval_nystrom[:, i]
        ax.plot(X_val.view(-1), data, alpha=1, label='Nyström $\hat\psi_{}$'.format(i+1))

    if eigenfuncs_eval_our is not None:
        plt.gca().set_prop_cycle(None)
        for i in range(k_lines):
            data = eigenfuncs_eval_our[:, i] if eigenfuncs_eval_our[:100, i].mean() > 0 else -eigenfuncs_eval_our[:, i]
            ax.plot(X_val.view(-1), data, linestyle='dashdot', label='Our $\hat\psi_{}$'.format(i+1))


    # ax.set_xlim(0., 0.999)
    # ax.set_title('CIFAR10+SVHN Error vs Confidence')
    # ax.set_xlabel(None)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    # ax.set_ylabel('CIFAR-10 Test Accuracy (%)', fontsize=16)

    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='lightgray', linestyle='--')
    ax.grid(axis='x', color='lightgray', linestyle='--')


def main():
    # set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # general settings
    x_dim = 1
    x_range = [-2, 2]
    k = 3
    for kernel_type in ['rbf', 'polynomial', 'periodic_plus_rbf']:
        if kernel_type == 'rbf':
            kernel = partial(rbf_kernel, 1, 1)
            ylim = [-2., 2.]
        elif kernel_type == 'periodic_plus_rbf':
            kernel = partial(periodic_plus_rbf_kernel, 3, 1, 1, 2, 1)
            ylim = [-2., 2.]
        elif kernel_type == 'cosine':
            kernel = partial(cosine_kernel, 4, 1, 1)
            ylim = [-2., 2.]
        elif kernel_type == 'polynomial':
            kernel = partial(polynomial_kernel, 4, 0.5, 1)
            ylim = [-2., 2.]
        elif kernel_type == 'sigmoid':
            kernel = partial(sigmoid_kernel, 1, 2)
            x_range = [-1, 1]
            ylim = [-2., 2.]

        X_val = torch.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 2000.).view(-1, 1)

        eigenvalues_nystrom_list, eigenfuncs_nystrom_list, cost_nystrom_list = [], [], []
        eigenvalues_our_list, nefs_our_list, cost_our_list = [], [], []
        for N in [64, 256, 1024, 4096]:
            X = torch.empty(N, x_dim).uniform_(x_range[0], x_range[1])

            eigenvalues_nystrom, eigenfuncs_nystrom, c = nystrom(X, x_dim, x_range, k, kernel)
            eigenvalues_nystrom_list.append(eigenvalues_nystrom)
            eigenfuncs_nystrom_list.append(eigenfuncs_nystrom)
            cost_nystrom_list.append(c)

            eigenvalues_our, nef, c = our(X, x_dim, x_range, k, kernel)
            eigenvalues_our_list.append(eigenvalues_our)
            nefs_our_list.append(nef)
            cost_our_list.append(c)

            print("-------------------" + str(N) + "-------------------")
            print("Eigenvalues estimated by nystrom method:")
            print(eigenvalues_nystrom_list[-1])
            print("Eigenvalues estimated by our method:")
            print(eigenvalues_our_list[-1])
            print("Time comparison {} vs. {}".format(cost_nystrom_list[-1], cost_our_list[-1]))



        # K_recon_by_nystrom = eigenfuncs_eval_nystrom @ torch.diag(eigenvalues_nystrom) @ eigenfuncs_eval_nystrom.T
        # K_recon_by_our = eigenfuncs_eval_our @ torch.diag(eigenvalues_our) @ eigenfuncs_eval_our.T
        # K_gd = kernel(X_val)
        # print("F norm between K and K_recon_by_nystrom:")
        # print(torch.linalg.norm(K_recon_by_nystrom - K_gd))
        # print("F norm between K and K_recon_by_our:")
        # print(torch.linalg.norm(K_recon_by_our - K_gd))


        # plots
        fig = plt.figure(figsize=(20, 4.5))
        ax = fig.add_subplot(141)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[0](X_val), nefs_our_list[0](X_val), k, x_range, ylim)

        ax = fig.add_subplot(142)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[1](X_val), nefs_our_list[1](X_val), k, x_range, ylim)

        ax = fig.add_subplot(143)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[2](X_val), nefs_our_list[2](X_val), k, x_range, ylim)

        # compare eigenfunctions
        ax = fig.add_subplot(144)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[3](X_val), nefs_our_list[3](X_val), k, x_range, ylim)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   ncol=k * 2, fancybox=True, shadow=True, prop={'size':16})
        fig.tight_layout()
        fig.savefig('eigen_funcs_comp_{}.pdf'.format(kernel_type), format='pdf', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
    main()
