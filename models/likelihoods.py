"""
This file contains the following classes:
    GaussianLikelihood
    BernoulliLikelihood
"""
import torch
import torch.nn as nn
from typing import Optional
import warnings

class GaussianLikelihood(nn.Module):
    """
    A class representing the Gaussian likelihood function. This should be used
    for regression tasks. It is the probabilistic analogue of a squared error 
    term in a loss function (i.e. log Gaussian likelihood == sum of square residuals)
    
    Args:
        sigma_y:
            a float representing the standard deviation of the observation noise/
            Gaussian likelihood.
            Default: 1e-2.
        
        train_sigma_y:
            a boolean flag denoting whether or not sigma_y should be optimised 
            along with the other hyper and variational parameters.
            Default: False
    """
    def __init__(self, y_dim: int, sigma_y: float = 1e-2, train: bool = False, sigma_y_upper_bound: Optional[float] = None):
        super().__init__()
        if sigma_y_upper_bound is not None:
            assert sigma_y <= sigma_y_upper_bound
            self.raw_sigmas = nn.Parameter((sigma_y*torch.ones((y_dim,))/sigma_y_upper_bound).logit(), requires_grad=train)
            self.upper_bound = sigma_y_upper_bound
            self.upper_bounded = True
        else:
            self.raw_sigmas = nn.Parameter((sigma_y*torch.ones((y_dim,))).log(), requires_grad=train)
            self.upper_bounded = False

    @property
    def sigmas(self):
        if self.upper_bounded:
            return self.raw_sigmas.sigmoid() * self.upper_bound
        else:
            return self.raw_sigmas.exp()
    
    def forward(self, f: torch.Tensor):
        # represents the transformation applied to f to get to y space.
        # For regression, f and y live in the same space.
        if f is None:
            return None
        return f
    
    def log_prob(self, predictions: torch.Tensor = None, targets: torch.Tensor = None):
        # computes the log-likelihood.
        dist = torch.distributions.Normal(predictions, self.sigmas.unsqueeze(0).unsqueeze(0))
        return dist.log_prob(targets.unsqueeze(0).repeat((predictions.shape[0], 1, 1)))
    

class BernoulliLikelihood(nn.Module):
    """
    A class representing the Bernoulli likelihood function. This should be used
    for classification tasks. This is the probabilistic analogue of the binary
    cross entropy loss (i.e. log Bernoulli likelihood == binary cross entropy).
    """
    def __init__(self):
        super().__init__()

    def forward(self, f: torch.Tensor):
        # represents the transformation applied to f to get to target space
        # for classification, the outputs must be probabilities between 0 and 1,
        # so the function must be squashed accordingly.
        if f is None:
            return None
        if f.shape[-1] > 1:
            warnings.warn("Using Bernoulli likelihood for multi-output logit function. Are you sure you mean to do this?")
        return torch.sigmoid(f).clamp(min=1e-5, max=1.0-1e-5)
    
    def log_prob(self, predictions: torch.Tensor = None, targets: torch.Tensor = None):
        # computes the log-likelihood. This is needed e.g. in ELBO for Monte Carlo
        # estimate of expected log-likelihood w.r.t. posterior distribution.
        targets_stack = targets.unsqueeze(0).repeat((predictions.shape[0], 1, 1))
        return (targets_stack * torch.log(predictions) + (1 - targets_stack) * torch.log(1 - predictions))