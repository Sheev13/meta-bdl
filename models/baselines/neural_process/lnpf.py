import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import warnings
from .cnpf import ConvCNP
from base_networks.set_architectures import DeepSet, ConvDeepSet, Transformer
from base_networks.base_architectures import MLP
from ...likelihoods import GaussianLikelihood

from typing import List, Optional
    

class ConvNP(ConvCNP):
    # this should be fairly straightforward to extend from ConvCNP.
    # it's basically just a wrapper---read the ConvNP paper and you will see, King.
    pass



class BaseLNP(nn.Module, ABC):
    def __init__(self,
                 x_dim: int = None,
                 y_dim: int = None,
                 lik: nn.Module = None, 
                 encoder_dims: List[int]=[32, 32],
                 decoder_dims: List[int]=[32, 32],
                 nonlinearity: nn.Module = nn.ReLU(),
                ):
        super().__init__()
        encoder_dims = encoder_dims.copy()
        decoder_dims = decoder_dims.copy()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = encoder_dims[-1]
        if encoder_dims[-1] != decoder_dims[0]:
            raise ValueError("Final DeepSet layer dimension and decoding MLP first layer dimension must match.")
        self.encoder_dims = encoder_dims.copy()
        self.encoder = None # define in child classes
        
        dec_dims = [decoder_dims[0]+x_dim] + decoder_dims.copy()[1:] + [y_dim]
        self.decoder = MLP(dims=dec_dims, nonlinearity=nonlinearity)

        self.likelihood = lik
        self.nonlinearity = nonlinearity

    @abstractmethod
    def compute_posterior(self):
        pass

    def forward(self, X_t: torch.Tensor, X_c: Optional[torch.Tensor]=None, y_c: Optional[torch.Tensor]=None, num_samples: int = 1):
        if y_c is not None:
            if len(y_c.shape) < 2: # if y's are 1-dimensional, ensure they are of shape (n, 1) rather than (n,).
                y_c = y_c.unsqueeze(-1) 
        q_z = self.compute_posterior(X_c, y_c)
        z_samps = q_z.rsample((num_samples,)) # shape (num_samples, r_dim)
        # z_samps = q_z.mean.unsqueeze(0).repeat((num_samples, 1)) # for debugging - only use mean.

        n_t = X_t.shape[0]
        repeated_z_samps = z_samps.unsqueeze(1).repeat((1, n_t, 1))
        repeated_X_t = X_t.unsqueeze(0).repeat((num_samples, 1, 1))
        decoder_input = torch.cat((repeated_z_samps, repeated_X_t), dim=-1) # shape (num_samples, n_t, r_dim + x_dim)
        dec_out = self.decoder(decoder_input)
        return self.likelihood(dec_out) # shape (num_samples, n_t, y_dim)

    def loss(self, X_c, y_c, X_t, y_t, num_samples=1, **redundant_kwargs):
        """Predictive log likelihood of targets given contexts"""
        pred_t = self(X_t, X_c, y_c, num_samples=num_samples)
        lel = self.likelihood.log_prob(pred_t, y_t).sum(-1).sum(-1).logsumexp(0) - torch.tensor(num_samples).log()
        # if you're just joining the stream, lel stands for log-expected-likelihood guys. Chat he's deffo using a calc.
        metrics = {
            "lel": lel.detach().item(),
        }
        if self.y_dim == 1 and isinstance(self.likelihood, GaussianLikelihood):
            if self.likelihood.raw_sigmas.requires_grad:
                metrics['sigma_y'] = self.likelihood.sigmas.detach().item()
        return - lel, metrics


class NP(BaseLNP):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        encoder_dims = [self.x_dim + self.y_dim] + self.encoder_dims + [2*self.r_dim]
        self.encoder = DeepSet(encoder_dims, nonlinearity=self.nonlinearity)

    def compute_posterior(self, X_c: torch.Tensor, y_c: torch.Tensor):
        if X_c is None or y_c is None:
            raise ValueError("NP requires a non-empty context set.")
        r_mean, r_log_std = self.encoder(X_c, y_c, flat_representation=True).chunk(2) # each of shape (r_dim,)
        r_log_std = r_log_std - 2.0 # for stability, ensure we begin training with small variances in latent variable's posterior.
        return torch.distributions.Normal(r_mean, 0.0001 + 0.9999*r_log_std.sigmoid())


class BNP(BaseLNP):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        encoder_dims = [self.x_dim + self.y_dim] + self.encoder_dims + [2*self.r_dim]
        self.encoder = MLP(encoder_dims, nonlinearity=self.nonlinearity)

        self.p_mean = torch.zeros((self.r_dim,))
        self.p_std = torch.ones((self.r_dim,))

    def compute_posterior(self, X_c: torch.Tensor, y_c: torch.Tensor):
        if X_c is None or y_c is None: # if empty context set, use prior predictive
            return torch.distributions.Normal(self.p_mean, self.p_std)

        # if non-empty context set, proceed as normal in an NP:
        m, raw_s = self.encoder(torch.cat((X_c, y_c), dim=-1)).chunk(2, dim=-1) # each of shape (n_t, r_dim)
        s = 0.0001 + 0.9999*(raw_s - 1.0).exp() # the most stable operation in the history of stable operations, maybe ever.

        # below implements equation (8) of Volpp paper on Bayesian context aggregation for NPs.
        q_std = (self.p_std.pow(-2) + s.pow(-2).sum(0)).pow(-0.5)
        d = m - self.p_mean.unsqueeze(0).repeat((m.shape[0], 1))
        q_mean = self.p_mean + q_std * (d / s.pow(2)).sum(0)
        return torch.distributions.Normal(q_mean, q_std)


class LANP(BaseLNP):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        num_layers = len(self.encoder_dims)
        width = max(self.encoder_dims)
        enc_dims = num_layers * [width]
        for i in range(num_layers):
            if self.encoder_dims[i] != enc_dims[i]:
                warnings.warn(f"User gave weird guff as ANP encoder architecture. Simplifying to {enc_dims}.")
                break
            self.encoder_dims = enc_dims

        self.encoder = Transformer(x_dim=self.x_dim,
                                   y_dim=self.y_dim,
                                   output_dim=self.encoder_dims[-1]*2,
                                   width=width,
                                   nonlinearity=self.nonlinearity,
                                   num_layers=num_layers,)
        
    def compute_posterior(self, X_c: torch.Tensor, y_c: torch.Tensor):
        if X_c is None or y_c is None:
            raise ValueError("NP requires a non-empty context set.")
        r_mean, r_log_std = self.encoder(torch.cat((X_c, y_c), dim=-1)).mean(0).chunk(2) # each of shape (r_dim,)
        r_log_std = r_log_std - 1.0 # for stability, ensure we begin training with small variances in latent variable's posterior.
        return torch.distributions.Normal(r_mean, 0.0001 + 0.9999*r_log_std.exp())


class ABNP(BaseLNP):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        num_layers = len(self.encoder_dims)
        width = max(self.encoder_dims)
        enc_dims = num_layers * [width]
        for i in range(num_layers):
            if self.encoder_dims[i] != enc_dims[i]:
                warnings.warn(f"User gave weird guff as ANP encoder architecture. Simplifying to {enc_dims}.")
                break
            self.encoder_dims = enc_dims

        self.encoder = Transformer(x_dim=self.x_dim,
                                   y_dim=self.y_dim,
                                   output_dim=self.encoder_dims[-1]*2,
                                   width=width,
                                   nonlinearity=self.nonlinearity,
                                   num_layers=num_layers,)
        
        self.p_mean = torch.zeros((self.r_dim,))
        self.p_std = torch.ones((self.r_dim,))
        
    def compute_posterior(self, X_c: torch.Tensor, y_c: torch.Tensor):
        if X_c is None or y_c is None: # if empty context set, use prior predictive
            return torch.distributions.Normal(self.p_mean, self.p_std)

        # if non-empty context set, proceed as normal in an NP:
        m, raw_s = self.encoder(torch.cat((X_c, y_c), dim=-1)).chunk(2, dim=-1) # each of shape (n_t, r_dim)
        s = 0.0001 + 0.9999*(raw_s - 1.0).exp() # the most stable operation in the history of stable operations, maybe ever.

        # below implements equation (8) of Volpp paper on Bayesian context aggregation for NPs.
        q_std = (self.p_std.pow(-2) + s.pow(-2).sum(0)).pow(-0.5)
        d = m - self.p_mean.unsqueeze(0).repeat((m.shape[0], 1))
        q_mean = self.p_mean + q_std * (d / s.pow(2)).sum(0)
        return torch.distributions.Normal(q_mean, q_std)
