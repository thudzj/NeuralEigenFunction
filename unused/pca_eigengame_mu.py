import math
import numpy as np
import torch
import torch.nn.functional as F

class ParallelEigengame(object):
    def __init__(self, n, K, device, lr=0.1, momentum=0, num_ites=None, lr_schedule=None, 
                 riemannian_projection=False):
        super(ParallelEigengame, self).__init__()
        self.n = n
        self.K = K
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.num_ites = num_ites
        self.lr_schedule = lr_schedule
        self.riemannian_projection = riemannian_projection

        self.V = torch.ones(K, n, device=device)
        self.V.div_(self.V.norm(dim=0))

        self.ite = 0
        self.acc_a = None
        self.acc_XtXV = None

    def step(self, X):
        lr = self.lr
        if self.num_ites is not None and self.lr_schedule is not None:
            if self.lr_schedule == 'cos':
                lr = self.lr * (1 + math.cos(math.pi * self.ite / self.num_ites)) / 2.

        XV = X @ self.V
        a = XV.T @ XV
        if self.acc_a is None:
            self.acc_a = a
        else:
            self.acc_a.mul_(self.momentum).add_(a, alpha = 1 - self.momentum)

        if self.acc_XtXV is None:
            self.acc_XtXV = X.T @ XV
        else:
            self.acc_XtXV.mul_(self.momentum).add_(X.T @ XV, alpha = 1 - self.momentum)
        
        g = self.acc_XtXV - self.V @ self.acc_a.tril(diagonal=-1).T
        if self.riemannian_projection:
            g.sub_((self.V*g).sum(0) * self.V)

        self.V.add_(g, alpha=lr)
        self.V.div_(self.V.norm(dim=0))
        self.ite += 1

    @property
    def eigvals(self):
        return self.acc_a.diag()


# Matrix X for which we want to find the PCA
# X = torch.from_numpy(
#         np.array([[7.,4.,5.,2.],
#             [2.,19.,6.,13.],
#             [34.,23.,67.,23.],
#             [1.,7.,8.,4.]])).float()

# X = torch.from_numpy(
#         np.array([[9.,0.,0.,0.],
#             [0.,8.,0.,0.],
#             [0.,0.,7.,0.],
#             [0.,0.,0.,1.]])).float()

# Centre the data
# X = X - X.mean(0)
# print(X)



X = torch.randn(128, 1024)
p,q = torch.symeig(X.T @ X / 128, eigenvectors=True)

epochs = 10000
n = 4
bs = 1
riemannian_projection = True

for lr in [0.1]:
    for lr_schedule in ['cos']:
        for momentum in [0.]:
            for riemannian_projection in [True]:
                algo = ParallelEigengame(n, X.shape[1], X.device, 
                                         lr=lr, momentum=momentum,
                                         num_ites=epochs, 
                                         lr_schedule=lr_schedule,
                                         riemannian_projection=riemannian_projection)

                for _ in range(epochs):
                    idx = np.random.choice(X.shape[0], bs, replace=False)
                    for i_ in range(n):
                        algo.step(X[idx]/math.sqrt(bs))
                
                print("\n Eigenvalues calculated using numpy are :\n",p[range(-1, -(n+1), -1)])
                print("\n Eigenvectors calculated using numpy are :\n",q[:, range(-1, -(n+1), -1)])
                print("\n Eigenvalues calculate using the Eigengame are :\n", algo.eigvals)
                print("\n Eigenvectors calculated using the Eigengame are :\n", algo.V)
                print("\n Squared error in estimation of eigenvectors as compared to numpy :\n")
                
                print(lr, lr_schedule, momentum, riemannian_projection,
                    torch.stack([q[:, -(i+1)] @ algo.V[:, i] for i in range(n)]).data.numpy())