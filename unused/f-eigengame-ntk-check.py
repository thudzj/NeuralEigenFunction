import copy
import math
import numpy as np

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from utils import ConvNetNT, ConvNet, init_NN

import torch
import torch.nn.functional as F

class Identity(torch.nn.Module):
	def __init__(self, *args):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

class LambdaLayer(torch.nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

def remove_bn(input):
    if isinstance(input, (torch.nn.BatchNorm2d)):
        output = Identity()
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, remove_bn(module))
    del input
    return output

arch = 'convnet2'
hs = [32, 64, 128,]
w, h, c = 28, 28, 1
bs1, bs2 = 5, 5
w_var, b_var = 2., 0.01
num_samples = 5000
random_dist_type = 'normal' # 'rademacher'
epsilon = 1e-6
x_train = np.random.randn(bs1, c, w, h)*0.1
x_test = np.random.randn(bs2, c, w, h)*0.1
data_cuda = torch.from_numpy(np.concatenate([x_train, x_test])).float().cuda()

# empirical ntk calculated in PyTorch
model = ConvNet(arch, hs, [c, w, h], 1)
model = remove_bn(model)
init_NN(model, w_var, b_var)
params = []
for name, m in model.named_modules():
    # print(name)
    one_layer_params = []
    if hasattr(m, 'weight'):
        one_layer_params.append(m.weight)
    if hasattr(m, 'bias'):
        one_layer_params.append(m.bias)
    if len(one_layer_params) > 0:
        params.append(one_layer_params)

model = extend(model).cuda()
model.zero_grad()
logits = model(data_cuda)
loss = logits.sum()
with backpack(BatchGrad()):
    loss.backward()
grad_batch = []
for name, p in model.named_parameters():
    grad_batch.append(p.grad_batch.flatten(1))
grad_batch = torch.cat(grad_batch, 1)
emp_ntk_test_train = grad_batch[:x_train.shape[0]] @ grad_batch[x_train.shape[0]:].T
print(emp_ntk_test_train)


# ntk calculated by neural_tangents library (check if our empirical ntk is exact)
model_nt = ConvNetNT(arch, hs, 1)
model_nt.random_init([-1, c, w, h])
params_idx = 0
for idx, p in enumerate(model_nt.params):
    if len(p) > 0:
        p = list(p)
        assert(len(p) == len(params[params_idx]))
        for i in range(len(p)):
            if params[params_idx][i].dim() == 4:
                # print(p[i].shape, params[params_idx][i].data.cpu().numpy().shape)
                p[i] = params[params_idx][i].data.cpu().numpy()
            elif params[params_idx][i].dim() == 1:
                # print(p[i].shape, params[params_idx][i].data.view(*p[i].shape).cpu().numpy().shape)
                p[i] = params[params_idx][i].data.view(*p[i].shape).cpu().numpy()
            elif params[params_idx][i].dim() == 2:
                # print(p[i].shape, params[params_idx][i].data.T.cpu().numpy().shape)
                p[i] = params[params_idx][i].data.T.cpu().numpy()
        model_nt.params[idx] = tuple(p)
        params_idx += 1
logits_nt = model_nt.f(model_nt.params, x_train)
emp_ntk_test_train_nt = model_nt.emp_ntk(x_train, x_test)
print(emp_ntk_test_train_nt)


# MC estimate of NTK
model.train()
original_params = copy.deepcopy(list(model.parameters()))
with torch.no_grad():
    original_logits = model(data_cuda).view(-1)
samples = []

print(random_dist_type, epsilon, original_params[0].view(-1)[:5])

for i in range(num_samples):
    for op, p in zip(original_params, list(model.parameters())):
        if random_dist_type == 'normal':
            perturbation = torch.randn_like(p) * epsilon
        elif random_dist_type == 'rademacher':
            perturbation = torch.randn_like(p).sign() * epsilon
        else:
            raise NotImplementedError
        p.data.copy_(op.data + perturbation)
    with torch.no_grad():
        new_logits = model(data_cuda).view(-1)
    samples.append((new_logits - original_logits).div(epsilon))
samples = torch.stack(samples, -1) / math.sqrt(num_samples)
mc_emp_ntk_test_train = samples[:x_train.shape[0]] @ samples[x_train.shape[0]:].T
print(torch.dist(emp_ntk_test_train, mc_emp_ntk_test_train))
print(mc_emp_ntk_test_train)
