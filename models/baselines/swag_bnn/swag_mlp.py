import torch
from torch import nn
from typing import List, Tuple
from abc import ABC, abstractmethod
from itertools import accumulate

class SWAG_BNN_Layer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 residual=False,
                ):
        super().__init__()
        self.p_mus = torch.zeros((d_out, d_in+1))
        self.p_Sigmas = torch.eye((d_in+1)).unsqueeze(0).repeat((d_out, 1, 1))
        if scale_prior:
            self.p_Sigmas /= torch.tensor(d_in+1)

        self.d_in = d_in
        self.d_out = d_out
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.residual = residual

    @property
    def p(self):
        return torch.distributions.MultivariateNormal(self.p_mus, self.p_Sigmas)

    def adopt_prior(self, m: torch.Tensor, S: torch.Tensor):
        assert len(m.shape) == 2
        assert len(S.shape) == 3
        self.p_mus = m
        self.p_Sigmas = S

    def log_prior(self, W: torch.Tensor):
        assert W.numel() == self.d_out * (self.d_in+1)
        return self.p.log_prob(W.view((self.d_out, self.d_in+1))).sum() # sum over output units (they are independent)

    def forward(self, X: torch.Tensor, w: torch.Tensor):
        # X is shape (num_samples, batch, d_in)
        # W has d_out * (d_in+1) elements
        assert w.numel() % (self.d_out * (self.d_in+1)) == 0
        w = w.reshape((-1, self.d_out, self.d_in+1)) # first dim is number of samples (1 during training)
        phi_X = self.nonlinearity(X)

        aug_X = torch.cat((phi_X, torch.ones((*X.shape[:-1], 1))), dim=-1) # shape (num_samples, batch, d_in+1)

        out = aug_X @ w.transpose(-2, -1)

        if self.d_in == self.d_out and self.residual:
            out += X

        return out


class SWAG_BNN(nn.Module, ABC):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dims: List[int],
                 likelihood: nn.Module,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 residual: bool = False,
                 K: int = 64):
        super().__init__()

        dims = [x_dim] + hidden_dims + [y_dim]
        weights_per_layer = [(dims[i]+1) * dims[i+1] for i in range(len(dims) - 1)]
        cumulative_weights_per_layer = [0] + list(accumulate(weights_per_layer))
        num_weights = cumulative_weights_per_layer[-1]
        assert num_weights == sum(weights_per_layer)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.likelihood = likelihood
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.identity = nn.Identity()
        self.residual = residual
        self.wpl = weights_per_layer
        self.cum_wpl = cumulative_weights_per_layer
        self.num_weights = num_weights

        # SWAG specific stuff
        self.W_pretrained = torch.zeros((self.num_weights,))
        self.K = K # rank of low-rank correction to posterior covariance
        self.W_swa = torch.zeros_like(self.W_pretrained)
        self.Sigma_diag = torch.ones_like(self.W_swa)
        self.D = torch.zeros((num_weights, K))

        layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            if i == 0:
                nl = nn.Identity()
            else:
                nl = self.nonlinearity
            layers.append(SWAG_BNN_Layer(dims[i], dims[i+1], scale_prior=self.scale_prior, nonlinearity=nl, residual=self.residual))

        self.layers = layers

    def set_pretrained(self, W: torch.Tensor):
        self.W_pretrained.data = W.detach().clone()

    @property
    def Sigma(self):
        diag_term = 0.5 * self.Sigma_diag.diag_embed()
        lr_sf = torch.ones((self.num_weights, self.num_weights)) - (0.5 * torch.eye(self.num_weights))
        low_rank_term = lr_sf * self.D @ self.D.T / (self.K - 1)
        return diag_term + low_rank_term


    def adopt_prior(self, layerwise_list):
        # layerwise_list is a list of (m, S) tuples for each layer.
        # each m is of shape (d_out, d_in+1)
        # each S is of shape (d_out, d_in+1, d_in+1)
        for i, prior_params in enumerate(layerwise_list):
            self.layers[i].adopt_prior(*prior_params)
    
    def weights_to_layerwise_vectors(self, W: torch.Tensor):
        if len(W.shape) == 1:
            W = W.unsqueeze(0) # first dim is number of samples
        assert len(W.shape) == 2
        assert W.shape[-1] == self.num_weights
        return [W[:,self.cum_wpl[i]:self.cum_wpl[i+1]] for i in range(len(self.cum_wpl)-1)]

    def forward(self, X: torch.Tensor, W: torch.Tensor):
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        layerwise_weights = self.weights_to_layerwise_vectors(W)
        for i, w in enumerate(layerwise_weights):
            X = self.layers[i](X, w)
        return X
    
    def bma_forward(self, X: torch.Tensor, num_samples: int):
        # X is shape (batch, x_dim)

        # # the following is simple but scales poorly with network size.
        # q = torch.distributions.MultivariateNormal(self.W_swa, self.Sigma)
        # W = q.sample((num_samples,)) # shape (num_samples, num_weights)

        # the following is more scalable. eq. (1) from SWAG paper.
        z1 = torch.randn((num_samples, self.num_weights))
        z2 = torch.randn((num_samples, self.K))
        diag_term = self.Sigma_diag.sqrt().unsqueeze(0) * z1 / torch.tensor(2.0).sqrt()
        lr_term = (self.D.unsqueeze(0) @ z2.unsqueeze(-1)).squeeze(-1) / torch.tensor(2 * (self.K - 1)).sqrt()
        assert len(diag_term.shape) == 2
        assert len(lr_term.shape) == 2
        W = self.W_swa.unsqueeze(0) + diag_term + lr_term

        X = X.unsqueeze(0).repeat((num_samples, 1, 1))
        return self(X, W)

    
    def log_likelihood(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
        pred_Y = self(X, W)
        return self.likelihood.log_prob(pred_Y, Y).mean(0).sum() # average over samples, but in training num_samples = 1
    
    def log_prior(self, W: torch.Tensor):
        layerwise_weights = self.weights_to_layerwise_vectors(W)
        return sum([self.layers[i].log_prior(w) for i, w in enumerate(layerwise_weights)])
    
    def log_posterior(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
        ll = self.log_likelihood(X, Y, W)
        lp = self.log_prior(W)
        metrics = {'log_pos': (ll + lp).item(), 'log_lik': ll.item(), 'log_pri': lp.item()}
        return ll + lp, metrics
    
    def sample_from_prior(self):
        W = torch.zeros((self.num_weights,))
        for i, layer in enumerate(self.layers):
            w = layer.p.sample()
            W[self.cum_wpl[i]:self.cum_wpl[i+1]] = w.flatten()
        return W
