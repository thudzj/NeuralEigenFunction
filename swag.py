import torch

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()

class CovarianceSpace(torch.nn.Module):

    def __init__(self, num_parameters, max_rank=100):
        super(CovarianceSpace, self).__init__()

        self.num_parameters = num_parameters

        self.register_buffer('rank', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cov_mat_sqrt',
                             torch.empty(0, self.num_parameters, dtype=torch.float32))

        self.max_rank = max_rank

    def collect_vector(self, vector):
        if self.rank.item() + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)

    def get_space(self):
        return self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        rank = state_dict[prefix + 'rank'].item()
        self.cov_mat_sqrt = self.cov_mat_sqrt.new_empty((rank, self.cov_mat_sqrt.size()[1]))
        super(CovarianceSpace, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                           strict, missing_keys, unexpected_keys,
                                                           error_msgs)

class SWAG(torch.nn.Module):

    def __init__(self, base_model,
                 subspace_kwargs=None, var_clamp=1e-6):
        super(SWAG, self).__init__()

        self.base_model = base_model
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = CovarianceSpace(num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'

    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        self.model_device = device.type
        self.subspace.to(device=torch.device('cpu'), dtype=dtype, non_blocking=non_blocking)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([param.detach().cpu() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        set_weights(self.base_model, self.mean, self.model_device)

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()

        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()
