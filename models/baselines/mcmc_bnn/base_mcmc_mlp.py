import torch
from torch import nn
from typing import List, Tuple
from abc import ABC, abstractmethod
from collections import accumulate

class MCMC_BNN_Layer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 residual=False,
                ):
        super().__init__()
        self.mus = torch.zeros((d_out, d_in+1))
        self.Sigmas = torch.eye((d_in+1)).unsqueeze(0).repeat((d_out, 1, 1))
        if scale_prior:
            self.Sigmas /= torch.tensor(d_in+1)

        self.d_in = d_in
        self.d_out = d_out
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity
        self.residual = residual

    @property
    def p(self):
        return torch.distributions.MultivariateNormal(self.mus, self.Sigmas)

    def adopt_prior(self, m: torch.Tensor, S: torch.Tensor):
        assert len(m.shape) == 2
        assert len(S.shape) == 3
        self.mus = m
        self.Sigmas = S

    def log_prior(self, W: torch.Tensor):
        assert W.numel() == self.d_out * (self.d_in+1)
        return self.p.log_prob(W.view((self.d_out, self.d_in+1))).sum() # sum over output units (they are independent)

    def forward(self, X: torch.Tensor, w: torch.Tensor):
        # X is shape (batch, d_in)
        # W has d_out * (d_in+1) elements
        assert w.numel() == self.d_out * (self.d_in+1)
        w = w.reshape((self.d_out, self.d_in+1))
        phi_X = self.nonlinearity(X)

        aug_X = torch.cat((phi_X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (batch, d_in+1)

        out = aug_X @ w.transpose(-2, -1)

        if self.d_in == self.d_out and self.residual:
            out += X

        return out


class MCMC_BNN(nn.Module, ABC):
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
        weights_per_layer = [dims[i] * (dims[i+1]+1) for i in range(len(dims) - 1)]
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

        layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            if i == 0:
                nl = nn.Identity()
            else:
                nl = self.nonlinearity
            layers.append(MCMC_BNN_Layer(dims[i], dims[i+1], scale_prior=self.scale_prior, nonlinearity=nl, residual=self.residual))

        self.layers = layers
    
    def weights_to_layerwise_vectors(self, W: torch.Tensor):
        assert len(W.shape) == 1
        assert W.shape[0] == self.num_weights
        return [W[self.cum_wpl[i]:self.cum_wpl[i+1]] for i in range(len(self.cum_wpl)-1)]

    def forward(self, X: torch.Tensor, W: torch.Tensor):
        layerwise_weights = self.weights_to_layerwise_vectors(W)
        for i, w in enumerate(layerwise_weights):
            X = self.layers[i](X, w)
        return X
    
    def log_likelihood(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
        pred_Y = self(X, W)
        return self.likelihood.log_prob(pred_Y.unsqueeze(0), Y).sum()
    
    def log_prior(self, W: torch.Tensor):
        layerwise_weights = self.weights_to_layerwise_vectors(W)
        return sum([self.layers[i].log_prior(w) for i, w in enumerate(layerwise_weights)])
    
    def U(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
        ll = self.log_likelihood(X, Y, W)
        lp = self.log_prior(W)
        return - (ll + lp)
    
    def grad_U(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor):
        W.requires_grad = True
        return torch.autograd.grad(self.U(X, Y, W), W)[0]
    
    def sample_from_prior(self):
        out = torch.zeros((self.num_weights,))
        for i, layer in enumerate(self.layers):
            w = layer.p.sample()
            out[self.cum_wpl[i]:self.cum_wpl[i+1]] = w.flatten()
        return out
    
    @abstractmethod
    def get_proposal(self):
        pass

    @abstractmethod
    def compute_log_acceptance(self):
        pass


class LMC_BNN(MCMC_BNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_proposal(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, step_size: float = 1e-4):
        return (W - (step_size / 2) * self.grad_U(X, Y) + torch.sqrt(torch.tensor(step_size)) * torch.randn_like(W),)

    def compute_log_proposal_prob(self, X: torch.Tensor, Y: torch.Tensor, W_curr: torch.Tensor, W_prop: torch.Tensor, step_size: float = 1e-4):
        # - 1/(2*stepsize)||q* - q - stepsize/2 * grad U(q)||^2
        grad_u = self.grad_U(X, Y, W_curr)
        norm = torch.linalg.vector_norm(
            W_prop - W_curr - (step_size / 2) * grad_u
        )
        log_prob = -(1 / (2 * step_size)) * norm.square()
        return log_prob
    
    def compute_log_acceptance(self, X: torch.Tensor, Y: torch.Tensor, W_curr: torch.Tensor, W_prop: torch.Tensor, step_size: float = 1e-4):
        U_curr = self.U(X, Y, W_curr)
        U_prop = self.U(X, Y, W_prop)
        curr_to_prop = self.compute_log_proposal_prob(
            X, Y, W_curr, W_prop, step_size=step_size
        )
        prop_to_curr = self.compute_log_proposal_prob(
            X, Y, W_prop, W_curr, step_size=step_size
        )
        return min(torch.tensor(0.0), U_prop - U_curr + prop_to_curr - curr_to_prop)
    


class HMC_BNN(MCMC_BNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute_leapfrog_step(self,
                              X: torch.Tensor,
                              Y: torch.Tensor,
                              W: torch.Tensor,
                              P: torch.Tensor,
                              step_size: float = 1e-4,
                             ):
        P_prime = P - (step_size / 2) * self.grad_U(X, Y, W)
        W_new = W + step_size * P_prime
        P_new = P_prime - (step_size / 2) * self.grad_U(X, Y, W_new)
        return W_new, P_new

    def get_proposal(self,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     W: torch.Tensor,
                     P: torch.Tensor,
                     step_size: float = 1e-4,
                     leapfrog_steps: int = 50,
                    ):
        W_new = W
        P_new = P

        for _ in range(leapfrog_steps):
            W_new, P_new = self.execute_leapfrog_step(
                W_new, P_new, X, Y, step_size=step_size
            )

        return W_new, P_new

    def compute_hamiltonian(self,
                            X: torch.Tensor,
                            Y: torch.Tensor,
                            W: torch.Tensor,
                            P: torch.Tensor,
                           ):
        potential = self.U(X, Y, W)
        kinetic = 0.5 * (torch.linalg.vector_norm(P) ** 2)
        return potential + kinetic

    def compute_log_acceptance(self,
                               X: torch.Tensor,
                               Y: torch.Tensor,
                               W_curr: torch.Tensor,
                               P_curr: torch.Tensor,
                               W_prop: torch.Tensor,
                               P_prop: torch.Tensor,
                              ):
        prop_ham = self.compute_hamiltonian(X, Y, W_prop, P_prop)
        curr_ham = self.compute_hamiltonian(X, Y, W_curr, P_curr)
        # if the leapfrog simulation is accurate enough,
        # the log acceptance probability should be zero due to energy conservation
        return min(torch.tensor(0.0), curr_ham - prop_ham)
