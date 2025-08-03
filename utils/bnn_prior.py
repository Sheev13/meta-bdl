import torch
from torch import nn
from typing import List, Optional

class GaussianLinearLayerPrior(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 scale_prior: bool = False,
                 ):
        super().__init__()
        self.mu = torch.zeros((d_out, d_in+1))
        self.sig = torch.ones((d_out, d_in+1))
        if scale_prior:
             self.sig /= torch.tensor(d_in+1).sqrt()

        self.p = torch.distributions.Normal(self.mu, self.sig)
        
    def forward(self, X: torch.Tensor):
        # X is shape (num_samples, batch, d_in)
        X = torch.cat((X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (num_samples, batch, d_in+1)
        W = self.p.sample((X.shape[0],)) # shape (num_samples, d_out, d_in+1)

        return X @ W.transpose(-2, -1)


class GaussianBNNPrior(nn.Module):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dims: List[int],
                 likelihood: Optional[nn.Module] = None,
                 scale_prior: bool = False,
                 nonlinearity: nn.Module = nn.ReLU(),
                ):
        super().__init__()

        layers = nn.ModuleList()
        dims = [x_dim] + hidden_dims + [y_dim]
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            layers.append(GaussianLinearLayerPrior(dims[i], dims[i+1], scale_prior))
            if i < len(dims) - 2:
                layers.append(nonlinearity)

        self.net = nn.Sequential(*layers)

        self.likelihood = likelihood

    def forward(self, X: torch.Tensor, num_samples: int):
        X = X.unsqueeze(0).repeat((num_samples, 1, 1))
        return self.net(X)
    
    def log_marginal_likelihood(self, X: torch.Tensor, Y: torch.Tensor, num_samples: int):
        # estimates log marginal likelihood via naive Monte Carlo integration

        if self.likelihood is None:
            raise ValueError("User Failed to specify likelihood function at MeanFieldBNNPrior initialisation.")

        pred_samps = self.likelihood(self(X, num_samples))
        log_lik = self.likelihood.log_prob(pred_samps) # shape (num_samps, batch, y_dim)
        log_marg_lik = log_lik.sum(-1).sum(-1).logsumexp(dim=0) - torch.tensor(num_samples).log()

        return log_marg_lik