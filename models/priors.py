import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .tensors import stable_inversion

class FCWeightwisePrior(nn.Module):
    """Fully-factorised Gaussian prior for a fully-connected layer's weights."""
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 scaled: bool = False
                 ):
        super().__init__()
        self.mus = nn.Parameter(torch.zeros((d_out, d_in+1)), requires_grad=False)
        self.log_sigmas = nn.Parameter(torch.zeros((d_out, d_in+1)), requires_grad=False)

        self.d_in = d_in
        self.d_out = d_out
        self.scaled = scaled
        self._hooks = {}

    @property
    def sigmas(self):
        sigmas = self.log_sigmas.exp()
        if self.scaled:
            sigmas /= torch.tensor(self.d_in+1).sqrt()
        return sigmas

    def trainable(self, flag: bool, just_mean: bool = False):
        self.mus.requires_grad = flag
        if just_mean:
            cov_flag = not flag
        else:
            cov_flag = flag
        self.log_sigmas.requires_grad = cov_flag

    def partially_trainable(self, n: int):
        self.trainable(True)
        num_weights = self.mus.numel()
        assert n < num_weights
        flat_idx = torch.randperm(num_weights)[:n]
        mask_m = torch.zeros(num_weights) # mask for gradients of mean matrix param
        mask_m[flat_idx] = 1
        mask_m = mask_m.view_as(self.mus)
        mask_s = mask_m.clone() # mask for gradients of log sigmas param

        for key, hook in self._hooks.items():
            hook.remove()
        self._hooks.clear()

        self._hooks['m'] = self.mus.register_hook(lambda g: g * mask_m)
        self._hooks['s'] = self.log_sigmas.register_hook(lambda g: g * mask_s)

    def forward(self, num_repeats=None):
        if num_repeats is not None:
            m = self.mus.unsqueeze(0).repeat(num_repeats, 1, 1)
            S = self.sigmas.unsqueeze(0).repeat(num_repeats, 1, 1)
            return torch.distributions.Normal(m, S)
        return torch.distributions.Normal(self.mus, self.sigmas)
    
    
class FCUnitwisePrior(nn.Module):
    """Unitwise full-rank Gaussian priors for a fully-connected layer's weights."""
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 scaled: bool = False,
                 ):
        super().__init__()
        self.mus = nn.Parameter(torch.zeros((d_out, d_in+1)), requires_grad=False)
        self.log_L_diags = nn.Parameter(torch.zeros((d_out, d_in+1)), requires_grad=False)
        self.L_off_diags = nn.Parameter(torch.zeros((d_out, d_in+1, d_in+1)), requires_grad=False)

        self.d_in = d_in
        self.d_out = d_out
        self.scaled = scaled
        self._hooks = {}

    @property
    def Sigmas(self):
        L_diags = self.log_L_diags.exp().diag_embed()
        L_off_diags = self.L_off_diags.tril(diagonal=-1)
        Ls = L_diags + L_off_diags
        Sigmas = Ls @ Ls.transpose(-2, -1) 
        if self.scaled:
            Sigmas /= (self.d_in + 1)
        return Sigmas

    def trainable(self, flag: bool, just_mean: bool = False):
        self.mus.requires_grad = flag
        if just_mean:
            cov_flag = not flag
        else:
            cov_flag = flag
        self.log_L_diags.requires_grad = cov_flag
        self.L_off_diags.requires_grad = cov_flag

    def partially_trainable(self, n: int):
        self.trainable(True)
        num_weights = self.mus.numel()
        assert n < num_weights
        flat_idx = torch.randperm(num_weights)[:n]
        mask_m = torch.zeros(num_weights) # mask for gradients of mean matrix param
        mask_m[flat_idx] = 1
        mask_m = mask_m.view_as(self.mus)
        mask_d = mask_m.clone() # mask for gradients of log diagonal of covariance matrix param
        mask_off_d = torch.zeros_like(self.L_off_diags) # mask for gradients of off-diagonals of cholesky matrix param
        for i in range(self.d_out):
            row_mask = mask_m[i]
            block_mask = torch.outer(row_mask, row_mask)
            mask_off_d[i] = block_mask

        for key, hook in self._hooks.items():
            hook.remove()
        self._hooks.clear()

        self._hooks['m'] = self.mus.register_hook(lambda g: g * mask_m)
        self._hooks['d'] = self.log_L_diags.register_hook(lambda g: g * mask_d)
        self._hooks['off_d'] = self.L_off_diags.register_hook(lambda g: g * mask_off_d)


    def forward(self, num_repeats=None):
        if num_repeats is not None:
            m = self.mus.unsqueeze(0).repeat(num_repeats, 1, 1)
            S = self.Sigmas.unsqueeze(0).repeat(num_repeats, 1, 1, 1)
            return torch.distributions.MultivariateNormal(m, S)
        return torch.distributions.MultivariateNormal(self.mus, self.Sigmas)
    

class FCLayerwisePrior(nn.Module):
    """Layerwise full-rank Gaussian prior for a fully-connected layer's weights."""
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 ):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros((1, d_out * (d_in + 1))), requires_grad=False)
        self.log_L_diag = nn.Parameter(torch.zeros((d_out * (d_in + 1),)), requires_grad=False)
        self.L_off_diag = nn.Parameter(torch.zeros((d_out * (d_in + 1), d_out * (d_in + 1))), requires_grad=False)

        self.d_in = d_in
        self.d_out = d_out

    @property
    def Sigma(self):
        L_diag = self.log_L_diag.exp().diag_embed()
        L_off_diag = self.L_off_diag.tril(diagonal=-1)
        L = L_diag + L_off_diag
        return L @ L.T / (self.d_in + 1)
    
    def trainable(self, flag: bool, just_mean: bool = False):
        self.mu.requires_grad = flag
        if just_mean:
            cov_flag = not flag
        else:
            cov_flag = flag
        self.log_L_diag.requires_grad = cov_flag
        self.L_off_diag.requires_grad = cov_flag
    
    def forward(self, num_repeats=1):
        return torch.distributions.MultivariateNormal(self.mu.repeat(num_repeats, 1), self.Sigma.unsqueeze(0).repeat(num_repeats, 1, 1))


class FCNetworkwisePrior(nn.Module):
    """Full-rank Gaussian prior over all the weights of a fully-connected network."""
    def __init__(self,
                 dims: List[int],
                 ):
        super().__init__()
        L = len(dims) - 1
        wpl = [(dims[i]+1) * dims[i+1] for i in range(L)] # list of number of weights per layer
        cum_wpl = [sum(wpl[:i+1]) for i in range(L)] # cumluative number of weights per layer
        num_w = cum_wpl[-1] # total number of weights

        self.mu = nn.Parameter(torch.zeros((num_w,)), requires_grad=False)
        self.log_L_diag = nn.Parameter(torch.zeros((num_w,)), requires_grad=False)
        self.L_off_diag = nn.Parameter(torch.zeros((num_w, num_w)), requires_grad=False)

        self.dims = dims
        self.L = L
        self.wpl = wpl
        self.cum_wpl = cum_wpl
        self.num_w = num_w

    @property
    def Sigma(self):
        L_diag = self.log_L_diag.exp().diag_embed()
        L_off_diag = self.L_off_diag.tril(diagonal=-1)
        L = L_diag + L_off_diag
        return L @ L.T
    
    def trainable(self, flag: bool, just_mean: bool = False):
        self.mu.requires_grad = flag
        if just_mean:
            cov_flag = not flag
        else:
            cov_flag = flag
        self.log_L_diag.requires_grad = cov_flag
        self.L_off_diag.requires_grad = cov_flag

    def forward(self, previous_layer_weights=None, first_layer=False):
        if previous_layer_weights is None and not first_layer:
            return torch.distributions.MultivariateNormal(self.mu, self.Sigma)
        elif first_layer:
            mu_1 = self.mu[:self.wpl[0]]
            Sigma_1 = self.Sigma[:self.wpl[0], :self.wpl[0]]
            return torch.distributions.MultivariateNormal(mu_1, Sigma_1)
        else:
            # each tensor in previous_layer_weights is shape (num_samples, d_in+1, d_out)
            num_samples = previous_layer_weights[0].shape[0]
            w_prev = torch.cat([w.reshape((num_samples, -1)) for w in previous_layer_weights], dim=-1) # shape (samples, n_prev)
            n_prev = w_prev.shape[1]
            if n_prev not in self.cum_wpl:
                raise ValueError("Unexpected number of weights passed to condition networkwise prior.")
            layer_i = self.cum_wpl.index(n_prev) + 1
            total_marginal_weights = self.cum_wpl[layer_i] # total number of weights not including the later layers
            n_curr = total_marginal_weights - n_prev

            mu, Sigma = self.mu[:total_marginal_weights], self.Sigma[:total_marginal_weights, :total_marginal_weights]

            mu_c = mu[n_prev:] # shape (n_curr,)
            mu_p = mu[:n_prev] # shape (n_prev,)

            Sigma_cc = Sigma[n_prev:, n_prev:]
            Sigma_cp = Sigma[n_prev:, :n_prev]
            Sigma_pp = Sigma[:n_prev, :n_prev]

            Sigma_pp_inv = stable_inversion(Sigma_pp)

            mu_conditional = mu_c.unsqueeze(0) + ((Sigma_cp @ Sigma_pp_inv).unsqueeze(0) @ (w_prev - mu_p.unsqueeze(0)).unsqueeze(-1)).squeeze(-1) # shape (samples, n_curr)
            Sigma_conditional = (Sigma_cc - Sigma_cp @ Sigma_pp_inv @ Sigma_cp.T).unsqueeze(0) + torch.eye(n_curr).unsqueeze(0)*0.001 # shape (1, n_curr, n_curr)

            return torch.distributions.MultivariateNormal(mu_conditional, Sigma_conditional) # object shape (samples, n_curr)