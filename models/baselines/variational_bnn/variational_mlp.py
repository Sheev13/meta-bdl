import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import List, Optional
from .variational_linear_layer import MFVILinearLayer, LCVILinearLayer
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

        self.layers = self.build_bnn()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.likelihood = likelihood
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.residual = residual

    @abstractmethod
    def build_bnn(self):
        pass

    def forward(self, X, num_samples=1, return_kl=False):
        # X is shape (batch, x_dim)
        X = X.unsqueeze(0).repeat(num_samples, 1, 1)

        cum_kl = torch.tensor(0.0)
        for layer in self.layers:
            out = layer(X, return_kl=return_kl, num_samples=num_samples)
            if return_kl:
                cum_kl += out[1]
            X = out[0]

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

        if self.x_dim == 1 and isinstance(self.likelihood, likelihoods.GaussianLikelihood):
            if self.likelihood.raw_sigmas.requires_grad:
                metrics['sigma_y'] = self.likelihood.sigmas.detach().item()

        return - elbo, metrics

class MFVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_bnn(self):
        dims = [self.x_dim] + self.hidden_dims + [self.y_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            if i == 0:
                nl = nn.Identity()
            else:
                nl = self.nonlinearity
            self.layers.append(MFVILinearLayer(dims[i], dims[i+1], scale_prior=self.scale_prior, nonlinearity=nl, residual=self.residual))
    

class LCVIBNN(BaseGaussianVIBNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_bnn(self):
        dims = [self.x_dim] + self.hidden_dims + [self.y_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            if i == 0:
                nl = nn.Identity()
            else:
                nl = self.nonlinearity
            self.layers.append(LCVILinearLayer(dims[i], dims[i+1], scale_prior=self.scale_prior, nonlinearity=nl, residual=self.residual))
