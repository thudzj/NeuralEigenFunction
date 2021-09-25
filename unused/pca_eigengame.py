import math
import numpy as np
import torch
import torch.nn.functional as F

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X):
    p,q = np.linalg.eig(X.T @ X)
    return (p,q)

# Define utlity function, i indicates ith eigenvector
# X is the design matrix and V holds the computed eigenvectors
def eigengame_step(i, X, V, lr=0.1, riemannian_projection=False):
    # calc the utlity
    Xv = (X @ V[i])
    rewards = Xv.norm()**2
    penalties = 0
    for j in range(i):
        Xvj = X @ V[j].detach()
        penalties = penalties + (Xv @ Xvj)**2 / (Xvj.norm()**2)
    utlity = rewards-penalties
    utlity.backward()

    # perform backprop
    with torch.no_grad():
        if riemannian_projection:
            V[i].grad.add_(V[i].data, alpha=-(V[i].grad @ V[i].data))

        V[i].data.add_(V[i].grad, alpha=lr)
        V[i].data = F.normalize(V[i].data, dim=0)
        V[i].grad.zero_()


def eigengame_sequencial(X, V, iterations=100, lr=0.1, riemannian_projection=False):
    for k in range(len(V)):
        print("Finding the eigenvector ",k)
        for _ in range(iterations):
            eigengame_step(k, X, V, lr=lr, riemannian_projection=riemannian_projection and k > 0)

# Calculate eigenvalues once the eigenvectors have been computed
@torch.no_grad()
def calc_eigengame_eigenvalues(X, V):
    eigvals = []
    for i, v in enumerate(V):
        eigvals.append((X @ v).norm()**2)
    return eigvals

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

X = torch.randn(128, 1024) + torch.rand(1024)
n = 4
V = []
for _ in range(n):
    v = F.normalize(torch.ones(X.shape[1], device=X.device), dim=0)
    v.requires_grad_()
    V.append(v)


p,q = calc_numpy_eig(X/math.sqrt(128))

X = X - X.mean(0)
# print(X)

p1,q1 = calc_numpy_eig(X/math.sqrt(128))
print("\n Eigenvalues calculated using numpy are :\n",p[:n])
print("\n Eigenvectors calculated using numpy are :\n",q[:, :n])
print("\n Eigenvalues calculated using numpy are :\n",p1[:n])
print("\n Eigenvectors calculated using numpy are :\n",q1[:, :n])
exit()

eigengame_sequencial(X, V, iterations=1000, lr=0.1, riemannian_projection=False)
eigvals = calc_eigengame_eigenvalues(X, V)
print("\n Eigenvalues calculated using numpy are :\n",p[:n])
print("\n Eigenvectors calculated using numpy are :\n",q[:, :n])
print("\n Eigenvalues calculate using the Eigengame are :\n", torch.stack(eigvals, -1))
print("\n Eigenvectors calculated using the Eigengame are :\n", torch.stack(V, -1))
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n")
for i in range(n):
    print(torch.dist(torch.from_numpy(q[:, i]), V[i]), torch.dist(torch.from_numpy(q[:, i]), -V[i]))