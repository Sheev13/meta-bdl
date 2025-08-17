import torch 
from torch import nn
from abc import ABC, abstractmethod

from ...inference import compute_unitwise_posteriors

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
    def q_w(self):
        raise NotImplementedError("Base class.")
    
    def forward(self, X: torch.Tensor, return_kl=False, **q_kwargs):
        # X is shape (num_samples, batch, d_in)
        phi_X = self.nonlinearity(X)

        aug_X = torch.cat((phi_X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (num_samples, batch, d_in+1)

        q_w = self.q_w(**q_kwargs) # event shape must include num_samples as dimension 0
        W = q_w.rsample() # W is shape (num_samples, d_out, d_in+1)
        if len(W.shape) == 2: # happens if using layerwise correlated q: event shape is (num_samples, d_out * (d_in+1))
            W = W.reshape((W.shape[0], self.d_out, self.d_in+1))

        out = aug_X @ W.transpose(-2, -1)

        if self.d_in == self.d_out and self.residual:
            out += X

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
        super().__init__(*args, **kwargs)
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
    

class UCVILinearLayer(VariationalLinearLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_mu = nn.Parameter(torch.zeros((self.d_out, self.d_in+1)), requires_grad=True)
        self.q_log_L_diags = nn.Parameter(torch.zeros((self.d_out, self.d_in+1)), requires_grad=True)
        self.q_L_off_diags = nn.Parameter(torch.zeros((self.d_out, self.d_in+1, self.d_in+1)), requires_grad=True)

    @property
    def q_Sigmas(self):
        L_diags = self.q_log_L_diags.exp().diag_embed()
        L_off_diags = self.q_L_off_diags.tril(diagonal=-1)
        Ls = L_diags + L_off_diags

        jitter = torch.eye(self.d_in + 1).unsqueeze(0) * 1e-5

        return (Ls @ Ls.transpose(-2, -1) + jitter) / 10
    
    def q_w(self, num_samples):
        q_mu = self.q_mu.unsqueeze(0).repeat((num_samples, 1, 1))
        q_Sigmas = self.q_Sigmas.unsqueeze(0).repeat((num_samples, 1, 1, 1))
        return torch.distributions.MultivariateNormal(q_mu, q_Sigmas)
    

class LCVILinearLayer(VariationalLinearLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        return (L @ L.transpose(-2, -1) + jitter) / 10
    
    def q_w(self, num_samples):
        q_mu = self.q_mu.unsqueeze(0).repeat((num_samples, 1))
        q_Sigma = self.q_Sigma.unsqueeze(0).repeat((num_samples, 1, 1))
        return torch.distributions.MultivariateNormal(q_mu, q_Sigma)
    
    def compute_kl(self, q: torch.distributions.Distribution):
        p_mu = self.mus.reshape((self.num_weights,)) 
        p_Sigma = torch.block_diag(*self.Sigmas) # shape (num_weights, num_weights)

        if len(q.mean.shape) == 1: # (num_weights,)
            p = torch.distributions.MultivariateNormal(p_mu, p_Sigma)
            return torch.distributions.kl_divergence(q, p)
        else: # q has event shape (num_samples, num_weights)
            num_samples = q.mean.shape[0]
            p_mu = p_mu.unsqueeze(0).repeat((num_samples, 1))
            p_Sigma = p_Sigma.unsqueeze(0).repeat((num_samples, 1, 1))
            p = torch.distributions.MultivariateNormal(p_mu, p_Sigma)
            return torch.distributions.kl_divergence(q, p).mean(0)
    

class GIVILinearLayer(VariationalLinearLayer):
    def __init__(self, *args, num_inducing=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_mu = nn.Parameter(torch.zeros((num_inducing, self.d_out)), requires_grad=True) # means of pseudo likelihoods t(w)
        self.t_log_L_diag = nn.Parameter(torch.zeros((num_inducing, )), requires_grad=True) 
        self.t_L_off_diag = nn.Parameter(torch.zeros((num_inducing, num_inducing)), requires_grad=True)

        self.num_inducing = num_inducing
    
    @property
    def t_Sigma(self):
        L_diag = self.t_log_L_diag.exp().diag_embed()
        L = L_diag + self.t_L_off_diag.tril(diagonal=-1)

        jitter = torch.eye(self.num_inducing) * 1e-5

        t_Sigma = (L @ L.transpose(-2, -1) + jitter) / 10

        return t_Sigma.unsqueeze(0).repeat((self.d_out, 1, 1))
    
    @property
    def prior(self):
        return torch.distributions.MultivariateNormal(self.mus, self.Sigmas)
    
    def q_w(self, aug_U):
        # aug_U is shape (num_samples, num_inducing, d_in+1). Inducing inputs propagated thus far in network.
        # It is expected to have been passed through nonlinearity already, and then augmented with ones (for bias).
        return compute_unitwise_posteriors(aug_U, self.t_mu, self.t_Sigma, self.prior, givi=True) # event shape (samples, d_out, d_in+1)
    
    def forward(self, X, U, return_kl=False):
        phi_U = self.nonlinearity(U)
        aug_U = torch.cat((phi_U, torch.ones((U.shape[0], U.shape[1], 1))), dim=-1) # shape (num_samples, num_inducing, d_in+1)
        phi_X = self.nonlinearity(X)
        aug_X = torch.cat((phi_X, torch.ones((X.shape[0], X.shape[1], 1))), dim=-1) # shape (num_samples, N, d_in+1)

        q_w = self.q_w(aug_U)
        W = q_w.rsample() # shape (num_samples, d_out, d_in+1)

        out_U = aug_U @ W.transpose(-2, -1)
        out_X = aug_X @ W.transpose(-2, -1)

        if self.residual and self.d_in == self.d_out:
            out_U += U
            out_X += X

        if return_kl:
            p_mu = self.mus.unsqueeze(0).repeat((X.shape[0], 1, 1))
            p_Sigmas = self.Sigmas.unsqueeze(0).repeat((X.shape[0], 1, 1, 1))
            p = torch.distributions.MultivariateNormal(p_mu, p_Sigmas)
            kl = torch.distributions.kl_divergence(q_w, p).mean(0).sum() 

            return out_X, out_U, kl

        return out_X, out_U

    

    
    


    
    