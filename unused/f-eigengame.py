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

def nystrom(x_dim, x_range, k, kernel, N_for_nystrom):

    # perform nystrom method
    start = timer()
    X_for_nystrom = torch.empty(N_for_nystrom, x_dim).uniform_(x_range[0], x_range[1])
    K_for_nystrom = kernel(X_for_nystrom)
    p, q = torch.symeig(K_for_nystrom, eigenvectors=True)
    eigenvalues_nystrom = p[range(-1, -(k+1), -1)] / N_for_nystrom
    eigenfuncs_nystrom = lambda x: kernel(x, X_for_nystrom) @ q[:, range(-1, -(k+1), -1)] \
                                     / p[range(-1, -(k+1), -1)] * math.sqrt(N_for_nystrom)
    end = timer()
    print("Nystrom method consumes {}s".format(end - start))

    return eigenvalues_nystrom, eigenfuncs_nystrom, end - start

def our(x_dim, x_range, k, kernel):
    # hyper-parameters for our
    '''
    input_size = x_dim
    hidden_size = 32
    num_layers = 3
    optimizer_type = 'RMSprop'
    lr = 1e-3
    momentum = 0.9
    num_iterations = 1000
    B = 128
    eigenvalue_momentum = 0.99
    nonlinearity = nn.GELU #nn.Tanh # nn.Sigmoid, nn.ReLU
    riemannian_projection = False
    '''
    input_size = x_dim
    hidden_size = 32
    num_layers = 3
    optimizer_type = 'SGD'
    lr = 1e-3
    momentum = 0.9
    num_iterations = 2000 #1000 for sigmoid
    B = 128 #512 for sigmoid
    nonlinearity = nn.GELU #nn.GELU #nn.Tanh # nn.Sigmoid, nn.ReLU
    riemannian_projection = False
    penalty_w = 1 #2. for sigmoid
    # penalty_n = 10.
    sample_based = False
    new_obj = False
    momentum1 = 0.99
    max_grad_norm = 10.
    change_x_ite = 1
    prepare_K_ites = B
    mv_K = False

    # perform our method
    start = timer()

    nef = NeuralEigenFunctions(k, input_size, hidden_size, num_layers, nonlinearity=nonlinearity)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(nef.parameters(), lr=lr)
    if optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(nef.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.SGD(nef.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)

    # X = torch.empty(B, x_dim).uniform_(x_range[0], x_range[1])
    # X.requires_grad_()
    # X.register_hook(lambda g: -g)
    # optimizer_X = torch.optim.SGD([X], lr=1e-3)
    # with torch.no_grad():
    #     K_XX = kernel(X)
    #     dist = MultivariateNormal(torch.zeros(B, device=K_XX.device), scale_tril=psd_safe_cholesky(K_XX))
    #
    # X_lr = 0

    # eigenvalues_our = None
    for ite in range(num_iterations):
        # X = torch.empty(B, x_dim).uniform_(x_range[0], x_range[1])
        # X.requires_grad_()

        if ite % change_x_ite == 0:
            X = torch.empty(B, x_dim).uniform_(x_range[0], x_range[1])
            # X.requires_grad_()
            # X.register_hook(lambda g: -g)
            with torch.no_grad():
                K_XX = kernel(X)
                # dist = MultivariateNormal(torch.zeros(B, device=K_XX.device), scale_tril=psd_safe_cholesky(K_XX))
                # F_X = dist.sample((prepare_K_ites,)).view(prepare_K_ites, -1) / math.sqrt(prepare_K_ites)
                # acc_K = F_X.T @ F_X

            acc_K_psis = None
            acc_psis_K_psis = None
            # acc_K = None

        psis_X = nef(X)
        with torch.no_grad():
            # if mv_K:
            #     F_X = dist.sample((1,)).view(1, -1) / math.sqrt(1)
            #     K_cur = F_X.T @ F_X
            #     if acc_K is None:
            #         acc_K = K_cur
            #     else:
            #         acc_K.mul_(momentum1).add_(K_cur, alpha = 1 - momentum1)
            #
            #     K_psis = acc_K @ psis_X; psis_K_psis = psis_X.T @ K_psis
            #     if acc_psis_K_psis is None:
            #         acc_psis_K_psis = psis_K_psis
            #     else:
            #         acc_psis_K_psis.mul_(momentum1).add_(psis_K_psis, alpha = 1 - momentum1)
            #     if new_obj:
            #         grad = K_psis - psis_X @ psis_K_psis.tril(diagonal=-1).T
            #     else:
            #         grad = K_psis - K_psis @ (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T * penalty_w

            if sample_based:
                F_X = dist.sample().view(1, -1)

                K_cur = F_X.T @ F_X
                # if acc_K is None:
                #     acc_K = K_cur
                # else:
                # acc_K.mul_(momentum1).add_(K_cur, alpha = 1 - momentum1)
                acc_K.mul_(ite % change_x_ite + prepare_K_ites).add_(K_cur).div_(ite % change_x_ite + prepare_K_ites + 1)

                # F_psis = F_X @ psis_X
                # # K_psis = F_X.T @ F_psis
                # psis_K_psis = F_psis.T @ F_psis
                #
                # # if acc_K_psis is None:
                # #     acc_K_psis = K_psis
                # # else:
                # #     acc_K_psis.mul_(momentum1).add_(K_psis, alpha = 1 - momentum1)
                #
                # if acc_psis_K_psis is None:
                #     acc_psis_K_psis = psis_K_psis
                # else:
                #     acc_psis_K_psis.mul_(momentum1).add_(psis_K_psis, alpha = 1 - momentum1)


                if new_obj:
                    acc_psis_K_psis = psis_X.T @ acc_K @ psis_X
                    grad = acc_K @ psis_X - psis_X @ acc_psis_K_psis.tril(diagonal=-1).T
                else:
                    # print(acc_K_psis.norm(dim=0), (acc_K_psis @ (acc_psis_K_psis / acc_psis_K_psis.diag()).tril(diagonal=-1).T).norm(dim=0) )
                    tmp_psis_X = psis_X
                    for _ in range(1):
                        acc_psis_K_psis = tmp_psis_X.T @ acc_K @ tmp_psis_X
                        tmp_psis_X = tmp_psis_X @ (torch.eye(k, device=X.device) - (acc_psis_K_psis / acc_psis_K_psis.diag()).tril(diagonal=-1).T)
                    grad = acc_K @ tmp_psis_X #psis_X @ (torch.eye(k, device=X.device) - (acc_psis_K_psis / acc_psis_K_psis.diag()).tril(diagonal=-1).T) # - ((psis_X ** 2).sum(0) - psis_X.shape[0]) * psis_X * penalty_n
            else:
                # K_XX = kernel(X);
                # K_psis = K_XX @ psis_X; psis_K_psis = psis_X.T @ K_psis
                # psis_K_psis_normalized = psis_K_psis / psis_K_psis.diag() * penalty_w
                # psis_K_psis_normalized = torch.eye(k, device=X.device) - psis_K_psis_normalized.tril(diagonal=-1)
                # grad = K_psis @ psis_K_psis_normalized.T

                # with torch.enable_grad():
                #     X_grad = torch.autograd.grad(psis_X, X, -grad)[0]
                # X.add_(X_grad.sign() * X_lr).clamp_(x_range[0], x_range[1])
                #
                # with torch.enable_grad():
                #     psis_X = nef(X)
                K_psis = K_XX @ psis_X; psis_K_psis = psis_X.T @ K_psis

                if acc_psis_K_psis is None:
                    acc_psis_K_psis = psis_K_psis
                else:
                    acc_psis_K_psis.mul_(momentum1).add_(psis_K_psis, alpha = 1 - momentum1)

                grad = K_psis - K_psis @ (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T * penalty_w
                # grad = K_psis - psis_X @ psis_K_psis.tril(diagonal=-1).T
            # if eigenvalues_our is None:
            #     eigenvalues_our = psis_K_psis.diag() / (B**2)
            # else:
            #     eigenvalues_our.mul_(momentum1).add_(psis_K_psis.diag() / (B**2), alpha = 1 - momentum1)
            if riemannian_projection:
                grad.sub_((psis_X*grad).sum(0) * psis_X)

            grad.mul_(2. / (B**2))
            clip_coef = max_grad_norm / (grad.norm(dim=0) + 1e-6) #torch.minimum(max_grad_norm / (grad.norm(dim=0) + 1e-6), torch.Tensor([1.],device=grad.device))
            tmp = grad.norm(dim=0)
            grad.mul_(clip_coef)
            # if ite % 50 == 0:
            #     print(ite, tmp, grad.norm(dim=0), X[:5].view(-1))

        optimizer.zero_grad()
        # optimizer_X.zero_grad()
        psis_X.backward(-grad)
        optimizer.step()
        # optimizer_X.step()
        # X.data.add_(X.grad.sign(), alpha=1e-4)
        # X.data.clamp_(x_range[0], x_range[1])
        # X.grad.zero_()
        scheduler.step()
    end = timer()
    print("Our method consumes {}s".format(end - start))
    return acc_psis_K_psis.diag()/(B**2), nef, end - start

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
        for N_for_nystrom in [8, 64, 512, 1024]: #4096
            eigenvalues_nystrom, eigenfuncs_nystrom, c = nystrom(x_dim, x_range, k, kernel, N_for_nystrom=N_for_nystrom)
            eigenvalues_nystrom_list.append(eigenvalues_nystrom)
            eigenfuncs_nystrom_list.append(eigenfuncs_nystrom)
            cost_nystrom_list.append(c)
        eigenvalues_our, nef, cost_our = our(x_dim, x_range, k, kernel)

        # compare eigenvalues
        print("Eigenvalues estimated by nystrom method:")
        print(eigenvalues_nystrom_list[-1])
        print("Eigenvalues estimated by our method:")
        print(eigenvalues_our)


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
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[0](X_val), None, k, x_range, ylim)

        ax = fig.add_subplot(142)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[1](X_val), None, k, x_range, ylim)

        ax = fig.add_subplot(143)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[2](X_val), None, k, x_range, ylim)

        # compare eigenfunctions
        ax = fig.add_subplot(144)
        with torch.no_grad():
            plot_efs(ax, k, X_val, eigenfuncs_nystrom_list[-1](X_val), nef(X_val), k, x_range, ylim)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   ncol=k * 2, fancybox=True, shadow=True, prop={'size':16})
        fig.tight_layout()
        fig.savefig('eigen_funcs_comp_{}.pdf'.format(kernel_type), format='pdf', dpi=1000, bbox_inches='tight')

if __name__ == '__main__':
    main()
