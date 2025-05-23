import torch
from torch import nn
from typing import List, Optional

from .priors import FCWeightwisePrior, FCUnitwisePrior, FCLayerwisePrior, FCNetworkwisePrior
from .inference import compute_unitwise_posteriors, compute_layerwise_posterior
from networks.base_architectures import MLP

class BDNPLayer(nn.Module):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 d_in: int,
                 d_out: int,
                 prior_type: int,
                 inf_dims: Optional[List[int]]=None,
                 targets_available=False,
                 global_noise=False,
                 scale_prior=False,
                 nonlinearity=nn.ReLU(),
                 first_layer=False,
                 final_layer=False,
                ):
        super().__init__()

        prior_types = [FCWeightwisePrior,
                       FCUnitwisePrior,
                       FCLayerwisePrior,
                       FCNetworkwisePrior]
        assert prior_type in [0, 1, 2, 3]
        if prior_type == 0:
            prior = prior_types[0](d_in, d_out, scale_prior)
        if prior_type == 3:
            prior = None
        else:
            prior = prior_types[prior_type](d_in, d_out)

        self.inf_net = None
        if not (global_noise and targets_available): # if we need an inference network to predict something
            if inf_dims is None: # ensure user has specified the inference network architecture
                    raise ValueError("User must specify inference network layer dimensions if amortising inference.")
            if (not global_noise) and (not targets_available):
                out_dims = 2*d_out # regular case of predicting targets and their uncertainties (log sigmas)
            elif global_noise or targets_available:
                out_dims = d_out # inference network just predicts log sigmas or targets
            self.inf_net = MLP([x_dim + y_dim] + inf_dims + [out_dims], nonlinearity=nonlinearity)

        self.d_in = d_in
        self.d_out = d_out
        self.prior_type = prior_type
        self.prior = prior
        self.targets_available = targets_available
        self.global_noise = global_noise
        self.nonlinearity = nonlinearity
        self.first_layer = first_layer
        self.final_layer = final_layer
        self.prev_W = None
        self.prev_q_w = None

    def compute_posterior(self, X, Y, log_sigmas, layerwise_conditional_prior=None, update_prev=False):
        if update_prev and self.final_layer:
            q_w = compute_unitwise_posteriors(X, Y, log_sigmas, self.prev_q_w)
        elif self.prior_type in [0, 1]:
            q_w = compute_unitwise_posteriors(X, Y, log_sigmas, self.prior())
        elif self.prior_type == 2:
            q_w = compute_layerwise_posterior(X, Y, log_sigmas, self.prior())
        elif self.prior_type == 3:
            if layerwise_conditional_prior is None:
                raise ValueError("User has chosen to use networkwise prior but not passed layerwise conditional.")
            q_w = compute_layerwise_posterior(X, Y, log_sigmas, layerwise_conditional_prior)

        return q_w


    def forward(self,
                Xt_prev_l,
                Xc_prev_l=None,
                Xc=None, 
                Yc=None,
                layerwise_conditional_prior=None,
                output_log_sigmas=None,
                return_kl=False,
                return_weights=False,
                num_samples=1,
                update_prev=False,
                save_stuff=False
               ):
        if Xc is None or Yc is None or Xc_prev_l is None: # use prior as posterior if no context data, unless doing sequential stuff
            if layerwise_conditional_prior is None:
                if update_prev and self.final_layer:
                    q_w = self.prev_q_w
                else:
                    q_w = self.prior(num_repeats=num_samples)
            else:
                p_w = layerwise_conditional_prior
                if p_w.mean.shape[0] == num_samples:
                    q_w = p_w
                else:
                    m = p_w.mean.repeat((num_samples, 1))
                    S = p_w.covariance_matrix.repeat((num_samples, 1, 1))
                    q_w = torch.distributions.MultivariateNormal(m, S)
        else: 
            if Xc is None or Yc is None or Xc_prev_l is None:
                raise ValueError("User must specify either all three of Xc, Yc, Xc_prev_l or none of them to each layer.")
            
            z = torch.cat((Xc, Yc), dim=-1) # shape (N, x_dim+y_dim)
            if self.targets_available and self.global_noise:
                Yc_l = Yc
                if output_log_sigmas is None:
                    raise ValueError("User must specify the observation noise if using global_noise=True.")
                log_sigmas = output_log_sigmas.unsqueeze(0).repeat(Xc.shape[0], 1) # shape (N, y_dim)
            elif self.targets_available:
                Yc_l = Yc
                log_sigmas = self.inf_net(z) - 2
            elif self.global_noise:
                Yc_l = self.inf_net(z)
                if output_log_sigmas is None:
                    raise ValueError("User must specify the observation noise if using global_noise=True.")
                log_sigmas = output_log_sigmas.unsqueeze(0).repeat(Xc.shape[0], 1) # shape (N, y_dim)
            else:
                Yc_l, log_sigmas = self.inf_net(z).chunk(chunks=2, dim=-1) # each of shape (N, y_dim)
                log_sigmas = log_sigmas - 2
            
            # handle nonlinearities and biases for context inputs
            if self.first_layer:
                Xc_prev_l_phi = torch.cat((Xc_prev_l, torch.ones((*Xc_prev_l.shape[:2], 1))), dim=-1) 
            else:
                Xc_prev_l_phi = torch.cat((self.nonlinearity(Xc_prev_l), torch.ones((*Xc_prev_l.shape[:2], 1))), dim=-1)

            q_w = self.compute_posterior(Xc_prev_l_phi, Yc_l, log_sigmas, layerwise_conditional_prior=layerwise_conditional_prior, update_prev=update_prev)
        if update_prev and not self.final_layer:
            W = self.prev_W
        else:
            W = q_w.rsample()
        if save_stuff:
            self.prev_q_w =  q_w
            self.prev_W = W.clone().detach()

        samples = W.shape[0]
        if len(W.shape) == 2: # shape (samples, d_out*(d_in+1))
            W = W.reshape((samples, self.d_out, self.d_in+1)).transpose(-2, -1)
        else: # shape (samples, d_out, d_in+1)
            W = W.transpose(-2, -1)
        # now weights are shape (samples, d_in+1, d_out)

        # handle nonlinearities and biases for target inputs
        if self.first_layer:
            Xt_prev_l_phi = torch.cat((Xt_prev_l, torch.ones((*Xt_prev_l.shape[:2], 1))), dim=-1) 
        else:
            Xt_prev_l_phi = torch.cat((self.nonlinearity(Xt_prev_l), torch.ones((*Xt_prev_l.shape[:2], 1))), dim=-1)
        # now Xt_prev_l_phi has shape (samples, Nt, d_in+1)

        Xt_l = Xt_prev_l_phi @ W # shape (samples, Nt, d_out)
        Xc_l = None
        if Xc_prev_l is not None:
            Xc_l = Xc_prev_l_phi @ W

        outputs = [Xt_l, Xc_l]

        # if we are training, compute KL divergence here
        if return_kl:
            if self.prior_type == 0:
                p_mu = self.prior().mean.unsqueeze(0) # shape (1, d_out, d_in+1)
                p_var = self.prior().variance.unsqueeze(0) # shape (1, d_out, d_in+1)
                p = torch.distributions.MultivariateNormal(p_mu, p_var.diag_embed())
            elif self.prior_type == 3:
                p = layerwise_conditional_prior
            else:
                p = self.prior()
            
            kl = torch.distributions.kl_divergence(q_w, p).mean(0).sum() # average over samples, sum over independent bits of W

            outputs.append(kl)

        if return_weights:
            outputs.append(W)

        return outputs

