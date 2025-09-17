import torch
from torch import nn
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import List, Optional
from .variational_linear_layer import MFVILinearLayer, UCVILinearLayer, LCVILinearLayer, GIVILinearLayer
from ... import likelihoods

class BaseGaussianVIBNN(nn.Module, ABC):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dims: List[int],
                 likelihood: nn.Module,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 residual: bool = False):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.likelihood = likelihood
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.residual = residual

    def build_bnn(self, layer, **kwargs):
        dims = [self.x_dim] + self.hidden_dims + [self.y_dim]
        layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            if i == 0:
                nl = nn.Identity()
            else:
                nl = self.nonlinearity
            layers.append(layer(dims[i], dims[i+1], scale_prior=self.scale_prior, nonlinearity=nl, residual=self.residual, **kwargs))
        
        self.layers = layers

    def adopt_prior(self, layerwise_list):
        # layerwise_list is a list of (m, S) tuples for each layer.
        # each m is of shape (d_out, d_in+1)
        # each S is of shape (d_out, d_in+1, d_in+1)
        for i, prior_params in enumerate(layerwise_list):
            self.layers[i].adopt_prior(*prior_params)

    def forward(self, X, num_samples=1, return_kl=False):
        # X is shape (batch, x_dim)
        X = X.unsqueeze(0).repeat((num_samples, 1, 1))

        cum_kl = torch.tensor(0.0)
        for layer in self.layers:
            out = layer(X, return_kl=return_kl, num_samples=num_samples)
            if return_kl:
                cum_kl += out[1]
                X = out[0]
            else:
                X = out

        if return_kl:
            return X, cum_kl
        return X
    
    def loss(self, X, Y, num_samples=1):
        
        preds, kl = self(X, return_kl=True, num_samples=num_samples)
        e_ll = self.likelihood.log_prob(preds, Y).mean(0).sum() # average over samples, sum over batch and y_dim
        elbo = e_ll - kl

        metrics = {
            "elbo": elbo.detach().item(),
            "e_ll": (e_ll).detach().item(),
            "kl": kl.detach().item()
        }

        if self.y_dim == 1 and isinstance(self.likelihood, likelihoods.GaussianLikelihood):
            if self.likelihood.raw_sigmas.requires_grad:
                metrics['sigma_y'] = self.likelihood.sigmas.detach().item()

        return - elbo, metrics


class MFVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_bnn(MFVILinearLayer)



class UCVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_bnn(UCVILinearLayer)

    

class LCVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_bnn(LCVILinearLayer)


class GIVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, num_inducing=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_bnn(GIVILinearLayer, num_inducing=num_inducing)

        self.num_inducing = num_inducing
        self.Z = nn.Parameter(torch.randn((num_inducing, self.x_dim)), requires_grad=True) * 4

    def init_inducing_points(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.x_dim
        if X.shape[0] >= self.num_inducing:
            idx = torch.randperm(X.shape[0])[:self.num_inducing]
            self.Z.data = X[idx,:]
        else:
            self.Z.data[:X.shape[0],:] = X

    def forward(self, X, num_samples=1, return_kl=False):
        # X is shape (batch, x_dim)
        X = X.unsqueeze(0).repeat((num_samples, 1, 1))
        U = self.Z.unsqueeze(0).repeat((num_samples, 1, 1))

        cum_kl = torch.tensor(0.0)
        for layer in self.layers:
            out = layer(X, U, return_kl=return_kl)
            if return_kl:
                cum_kl += out[-1]
                X, U = out[:-1]
            else:
                X, U = out

        if return_kl:
            return X, cum_kl
        return X
    




class FCVIBNN(nn.Module): # doesn't use above base class because too different. So guff man. Wash hands after using this class.
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dims: List[int],
                 likelihood: nn.Module,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 residual: bool = False):
        super().__init__()

        dims = [x_dim] + hidden_dims + [y_dim]
        weights_per_layer = [(dims[i]+1) * dims[i+1] for i in range(len(dims) - 1)]
        cumulative_weights_per_layer = [0] + list(accumulate(weights_per_layer))
        num_weights = cumulative_weights_per_layer[-1]
        assert num_weights == sum(weights_per_layer)

        self.p_mu = torch.zeros((num_weights,))
        self.p_Sigma = torch.eye(num_weights)
        if scale_prior:
            sf = torch.ones((num_weights,))
            cwpl = cumulative_weights_per_layer # long-ass name
            for i in range(len(cwpl)-1):
                sf[cwpl[i]:cwpl[i+1]] /= (dims[i]+1)
            self.p_Sigma *= sf.diag_embed()

        self.q_mu = nn.Parameter(torch.zeros((num_weights,)), requires_grad=True)
        self.q_log_L_diag = nn.Parameter(torch.zeros((num_weights,)), requires_grad=True)
        self.q_L_off_diags = nn.Parameter(torch.zeros((num_weights, num_weights)), requires_grad=True)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.likelihood = likelihood
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.identity = nn.Identity()
        self.residual = residual
        self.dims = dims
        self.wpl = weights_per_layer
        self.cum_wpl = cumulative_weights_per_layer
        self.num_weights = num_weights

    @property
    def q_Sigma(self):
        L_diag = self.q_log_L_diag.exp().diag_embed()
        L_off_diags = self.q_L_off_diags.tril(diagonal=-1)
        L = L_diag + L_off_diags

        jitter = torch.eye(self.num_weights) * 1e-5

        return (L @ L.transpose(-2, -1) + jitter) / 10
    
    def q_w(self, num_samples):
        q_mu = self.q_mu.unsqueeze(0).repeat((num_samples, 1))
        q_Sigma = self.q_Sigma.unsqueeze(0).repeat((num_samples, 1, 1))
        return torch.distributions.MultivariateNormal(q_mu, q_Sigma)
    
    def forward(self, X, num_samples=1):
        # X is shape (batch, x_dim)
        X = X.unsqueeze(0).repeat((num_samples, 1, 1))
        W = self.q_w(num_samples).rsample() # shape (num_samples, num_weights)
        for i in range(len(self.hidden_dims)+1): # == range(len(dims) - 1)
            old_X = X
            W_l = W[:,self.cum_wpl[i]:self.cum_wpl[i+1]]
            W_l = W_l.reshape((num_samples, self.dims[i+1], self.dims[i]+1))

            if i == 0:
                phi_X = self.identity(X)
            else:
                phi_X = self.nonlinearity(X)

            aug_X = torch.cat((phi_X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (num_samples, batch, d_in+1)
            
            X = aug_X @ W_l.transpose(-2, -1)

            if self.residual and W_l.shape[-2] == W_l.shape[-1] + 1:
                X += old_X

        return X
    
    def compute_kl(self):
        p = torch.distributions.MultivariateNormal(self.p_mu, self.p_Sigma)
        q = torch.distributions.MultivariateNormal(self.q_mu, self.q_Sigma)
        return torch.distributions.kl_divergence(q, p)
    
    def loss(self, X, Y, num_samples=1):

        preds = self(X, num_samples=num_samples)
        e_ll = self.likelihood.log_prob(preds, Y).mean(0).sum() # average over samples, sum over batch and y_dim
        kl = self.compute_kl()
        elbo = e_ll - kl

        metrics = {
            "elbo": elbo.detach().item(),
            "e_ll": (e_ll).detach().item(),
            "kl": kl.detach().item()
        }

        if self.y_dim == 1 and isinstance(self.likelihood, likelihoods.GaussianLikelihood):
            if self.likelihood.raw_sigmas.requires_grad:
                metrics['sigma_y'] = self.likelihood.sigmas.detach().item()

        return - elbo, metrics
            
