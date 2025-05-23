import torch
from torch import nn
from typing import List, Optional
from . import likelihoods
from .priors import FCNetworkwisePrior
from .bnn_layers import BDNPLayer

class BDNP(nn.Module):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 hidden_dims: List[int],
                 prior_type: int,
                 likelihood: nn.Module,
                 inf_dims: List[int]=None,
                 use_final_layer_targets=False,
                 use_final_layer_noise=False,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                ):
        super().__init__()
        dims = [x_dim] + hidden_dims + [y_dim]

        if prior_type == 3:
            prior = FCNetworkwisePrior(dims)
        else:
            prior = None

        if (use_final_layer_targets or use_final_layer_noise) and not isinstance(likelihood, likelihoods.GaussianLikelihood):
            raise ValueError("Houston, we have a problem. User is trying to do Bayesian linear regression at output layer under a non-Gaussian noise model.")

        self.layers = nn.ModuleList()
        for i in range(len(dims)):
            if i == len(dims) - 1:
                break
            ta = (i == len(dims)-2 and use_final_layer_targets)
            gn = (i == len(dims)-2 and use_final_layer_noise)
            sp = (prior_type == 0 and scale_prior == True)
            self.layers.append(BDNPLayer(x_dim,
                                         y_dim,
                                         dims[i],
                                         dims[i+1],
                                         prior_type=prior_type,
                                         inf_dims=inf_dims,
                                         targets_available=ta,
                                         global_noise=gn,
                                         scale_prior=sp,
                                         nonlinearity=nonlinearity,
                                         first_layer=(i==0),
                                         final_layer=(i==len(dims)-2)
                                        )
                              )

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.prior_type = prior_type
        self.prior = prior
        self.likelihood = likelihood
        self.dims = dims
        self.inf_dims = inf_dims
        self.use_final_layer_targets = use_final_layer_targets
        self.use_final_layer_noise = use_final_layer_noise
        self.scale_prior = scale_prior
        self.nonlinearity = nonlinearity

    def trainable_prior(self, flag: bool, just_mean: bool = False):
        if self.prior_type == 3:
            self.prior.trainable(flag, just_mean=just_mean)
        else:
            for l in self.layers:
                l.prior.trainable(flag, just_mean=just_mean)

    def forward(self, Xt, Xc=None, Yc=None, return_kl=False, num_samples=1, update_prev=False, save_stuff=False):
        # Xt shape (Nt, x_dim), Xc shape (Nc, x_dim)
        Xt_prev = Xt.clone().unsqueeze(0).repeat((num_samples, 1, 1))
        if Xc is None:
            Xc_prev = None
        else:
            Xc_prev = Xc.clone()
            Xc_prev = Xc_prev.unsqueeze(0).repeat((num_samples, 1, 1))

        cum_kl = torch.tensor(0.0)
        prev_weights = []
        for i, layer in enumerate(self.layers):
            lcp = None
            if self.prior_type == 3:
                lcp = self.prior(previous_layer_weights=prev_weights, first_layer=(i==0))
            ols = None
            if i == len(self.dims) - 2:
                if self.use_final_layer_noise:
                    ols = self.likelihood.sigmas.log()
            outputs = layer(Xt_prev,
                            Xc_prev,
                            Xc,
                            Yc,
                            layerwise_conditional_prior=lcp,
                            output_log_sigmas=ols,
                            return_kl=return_kl,
                            return_weights=(self.prior_type==3),
                            num_samples=num_samples,
                            update_prev=update_prev,
                            save_stuff=save_stuff              
                           )
            Xt_prev, Xc_prev = outputs[0], outputs[1]
            if return_kl:
                cum_kl += outputs[2]
                if self.prior_type == 3:
                    prev_weights.append(outputs[3])
            elif self.prior_type == 3:
                prev_weights.append(outputs[2])
        
        return Xt_prev, Xc_prev, cum_kl
    
    def minibatched_posterior_sample(self, context_dataloader):
        pass

    def loss(self, Xc, Yc, Xt, Yt, num_samples=1, use_kl=True):
        pred_t, pred_c, kl = self(Xt, Xc=Xc, Yc=Yc, return_kl=use_kl, num_samples=num_samples)
        e_ll = self.likelihood.log_prob(pred_t, Yt).mean(0).sum() # average over samples, sum over batch
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
