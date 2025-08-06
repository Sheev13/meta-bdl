import torch 
from torch import nn

from abc import ABC, abstractmethod

class VariationalLinearLayer(nn.Module, ABC):
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

    def adopt_prior(self, m: torch.Tensor, S: torch.Tensor):
        assert len(m.shape) == 2
        assert len(S.shape) == 3
        self.mus = m
        self.Sigmas = S

    @abstractmethod
    def q_w(self, *args, **kwargs):
        raise NotImplementedError("Base class.")
    
    def forward(self, X: torch.Tensor, return_kl=False, **q_kwargs):
        # X is shape (num_samples, batch, d_in)
        X = torch.cat((X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (num_samples, batch, d_in+1)

        phi_X = self.nonlinearity(X)

        q_w = self.q_w(**q_kwargs) # event shape must include num_samples as dimension 0
        W = q_w.sample() # W is shape (num_samples, d_out, d_in+1)
        if len(W.shape) == 2: # happens if using layerwise correlated q: event shape is (num_samples, d_out * (d_in+1))
            W = W.reshape((W.shape[0], self.d_out, self.d_in+1))

        out = phi_X @ W.transpose(-2, -1)

        if self.d_in == self.d_out and self.residual:
            out += X[:,:,:-1]

        if return_kl:
            kl = self.compute_kl(q_w)
            return out, kl

        return out

    def compute_kl(self, q: torch.distributions.Distribution):
        if len(q.mean.shape) == 2: # (d_out, d_in+1)
            p = torch.distributions.MultivariateNormal(self.mus, self.Sigmas)
            return torch.distributions.kl_divergence(q, p).sum() 
        else: # q has event shape (num_samples, d_out, d_in+1)
            num_samples = q.mean.shape[0]
            mus = self.mus.unsqueeze(0).repeat((num_samples, 1, 1))
            Sigmas = self.Sigmas.unsqueeze(0).repeat((num_samples, 1, 1, 1))
            p = torch.distributions.MultivariateNormal(mus, Sigmas)
            return torch.distributions.kl_divergence(q, p).mean(0).sum() 
        

class MFVILinearLayer(VariationalLinearLayer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q_mu = nn.Parameter(torch.zeros((self.d_out, self.d_in+1)), requires_grad=True)
        self.log_q_stds = nn.Parameter(torch.zeros((self.d_out, self.d_in+1)), requires_grad=True)

    @property
    def q_Sigmas(self):
        q_vars = self.log_q_stds.exp().pow(2)
        return q_vars.diag_embed()
    
    def q_w(self, num_samples):
        q_mu = self.q_mu.unsqueeze(0).repeat((num_samples, 1, 1))
        q_Sigmas = self.q_Sigmas.unsqueeze(0).repeat((num_samples, 1, 1, 1))
        return torch.distributions.MultivariateNormal(q_mu, q_Sigmas)
    

class LCVILinearLayer(VariationalLinearLayer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        num_weights = self.d_out * (self.d_in + 1)
        self.q_mu = nn.Parameter(torch.zeros((num_weights,)), requires_grad=True)
        self.q_log_L_diag = nn.Parameter(torch.zeros((num_weights,)), requires_grad=True)
        self.q_L_off_diag = nn.Parameter(torch.zeros((num_weights, num_weights)), requires_grad=True)

        self.num_weights = num_weights

    @property
    def q_Sigma(self):
        L_diag = self.q_log_L_diag.exp().diag_embed()
        L = L_diag + self.q_L_off_diag.tril(diagonal=-1)

        jitter = torch.eye(self.num_weights) * 1e-5

        return (L @ L.T + jitter) / 10
    
    def q_w(self, num_samples):
        q_mu = self.q_mu.unsqueeze(0).repeat((num_samples, 1, 1))
        q_Sigma = self.q_Sigma
        return torch.distributions.MultivariateNormal(q_mu, q_Sigma)




    
    