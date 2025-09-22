import torch
from torch import nn
from typing import List, Optional
from itertools import accumulate
from . import likelihoods
from .priors import FCNetworkwisePrior
from .amortised_layers import AmortisedLinearLayer, AmortisedAttentionBlock

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
                 residual: bool = False,
                 pyramid_inf_net = False,
                 inf_transformer_width: Optional[int] = None,
                 inf_transformer_layers: Optional[int] = None,
                 inf_net_use_act: bool = False,
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
            sp = (prior_type in [0, 1] and scale_prior == True)
            if pyramid_inf_net:
                inf_dims = hidden_dims[:i+1]
            self.layers.append(AmortisedLinearLayer(x_dim,
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
                                         final_layer=(i==len(dims)-2),
                                         residual=residual,
                                         inf_transformer_width=inf_transformer_width,
                                         inf_transformer_layers=inf_transformer_layers,
                                         inf_net_use_act=(inf_net_use_act and i!=0)
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

    def set_prior_trainability(self, proportion: float, from_front: bool = False):
        assert proportion >= 0.0
        assert proportion <= 1.0
        if self.prior_type not in [0, 1]:
            raise NotImplementedError(f"User wants to set partial prior trainability with prior type {self.prior_type}. Only implemented for prior types 0 and 1.")
        dims = self.dims
        weights_per_layer = [(dims[i]+1) * dims[i+1] for i in range(len(dims) - 1)]
        cumulative_weights_per_layer = [0] + list(accumulate(weights_per_layer))
        num_weights = cumulative_weights_per_layer[-1]

        self.trainable_prior(False)

        if from_front:
            stop_weight = int(proportion * num_weights)
            for i, layer in enumerate(self.layers):
                if stop_weight >= cumulative_weights_per_layer[i+1]:
                    layer.prior.trainable(True)
                    if stop_weight == cumulative_weights_per_layer[i+1]:
                        break
                elif stop_weight > cumulative_weights_per_layer[i]:
                    layer.prior.partially_trainable(stop_weight - cumulative_weights_per_layer[i])
                    break
        else:
            start_weight = num_weights - int(proportion * num_weights)
            for i, layer in enumerate(self.layers):
                if start_weight <= cumulative_weights_per_layer[i]:
                    layer.prior.trainable(True)
                elif start_weight < cumulative_weights_per_layer[i+1]:
                    layer.prior.partially_trainable(cumulative_weights_per_layer[i+1] - start_weight)
        

    def forward(self, Xt, Xc=None, Yc=None, return_kl=False, num_samples=1, update_prev=False, save_stuff=False, batch_size=None):
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
            if batch_size is not None:
                assert self.prior_type in [0, 1]
                assert self.use_final_layer_noise == False
                assert Xc is not None
                if torch.is_grad_enabled(): # if we are training
                    if batch_size >= Xc.shape[0]:
                        grad_batch_idx = 0
                    else:
                        num_batches = -(-Xc.shape[0] // batch_size)  # works for integers, same as math.ceil(a / b)
                        grad_batch_idx = torch.randint(0, num_batches, (1,)).item()
                else:
                    grad_batch_idx = None
                outputs = layer.minibatched_forward(Xt_prev,
                                                    Xc_prev,
                                                    Xc,
                                                    Yc,
                                                    return_kl=return_kl,
                                                    num_samples=num_samples,
                                                    batch_size=batch_size,
                                                    grad_batch_idx=grad_batch_idx
                                                   )
            else:
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
                if self.prior_type == 3:
                    prev_weights.append(outputs[-1])


            Xt_prev, Xc_prev = outputs[0], outputs[1]
            if return_kl:
                cum_kl += outputs[2]
        
        return self.likelihood(Xt_prev), self.likelihood(Xc_prev), cum_kl

    def loss(self, Xc, Yc, Xt=None, Yt=None, num_samples=1, use_kl=True, logsumexp=False, pp_avi=False, batch_size=None, **kwargs):
        metrics = {}
        if Xt is None:
            Xt, Yt = Xc, Yc
        pred_t, pred_c, kl = self(Xt, Xc=Xc, Yc=Yc, return_kl=use_kl, num_samples=num_samples, batch_size=batch_size)
        if pp_avi:
            trgt_ppl = self.likelihood.log_prob(pred_t, Yt).sum(-1).sum(-1).logsumexp(0) - torch.tensor(num_samples).log()
            ctxt_ell = self.likelihood.log_prob(pred_c, Yc).mean(0).sum() # average over samples, sum over batch
            elbo = ctxt_ell - kl
            loss = elbo + trgt_ppl
            metrics["pp_avi"] = loss.detach().item()
            metrics["elbo"] = elbo.detach().item()
            metrics["trgt_ppl"] = trgt_ppl.detach().item()
            metrics["ctxt_ell"] = ctxt_ell.detach().item()
            metrics["kl"] = kl.detach().item()
        elif logsumexp: # log-sum-exp over samples, sum over batch
            # estimates log expected likelihood (i.e. log posterior predictive)
            loss = self.likelihood.log_prob(pred_t, Yt).sum(-1).sum(-1).logsumexp(0) - torch.tensor(num_samples).log()
            metrics["ppl"] = loss.detach().item()
        else:
            # estimates expected log likelihood
            ell = self.likelihood.log_prob(pred_t, Yt).mean(0).sum() # average over samples, sum over batch
            loss = ell - kl
            if use_kl:
                metrics["elbo"] = loss.detach().item()
                metrics["ell"] = ell.detach().item()
                metrics["kl"] = kl.detach().item()
            else:
                metrics["ell"] = loss.detach().item()

        if self.y_dim == 1 and isinstance(self.likelihood, likelihoods.GaussianLikelihood):
            if self.likelihood.raw_sigmas.requires_grad:
                metrics['sigma_y'] = self.likelihood.sigmas.detach().item()

        return - loss, metrics



class BDAM(nn.Module):
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 num_blocks: int,
                 d_emb: int,
                 likelihood: nn.Module,
                 num_heads: int = 8,
                 inf_dims: Optional[List[int]]=None,
                 nonlinearity: nn.Module = nn.ReLU(),
                 use_final_layer_targets: bool = False,
                 use_final_layer_noise: bool = False,
                 inf_net_use_act: bool = False,
                ):
        super().__init__()
        self.input_layer = AmortisedLinearLayer(x_dim, y_dim, d_in=x_dim, d_out=d_emb, prior_type=1, inf_dims=inf_dims, nonlinearity=nn.Identity())

        bdap_blocks = []
        for _ in range(num_blocks):
            bdap_blocks.append(
                AmortisedAttentionBlock(x_dim, y_dim, d_emb, num_heads=num_heads, inf_dims=inf_dims, nonlinearity=nonlinearity, inf_net_use_act=inf_net_use_act)
            )
        self.bdap_blocks = nn.ModuleList(bdap_blocks)
        
        self.prediction_head = AmortisedLinearLayer(x_dim,
                                         y_dim,
                                         d_in=d_emb,
                                         d_out=y_dim,
                                         prior_type=1,
                                         inf_dims=inf_dims,
                                         nonlinearity=nonlinearity,
                                         final_layer=True,
                                         targets_available=use_final_layer_targets,
                                         global_noise=use_final_layer_noise,
                                         inf_net_use_act=inf_net_use_act,
                                        )
        
        self.likelihood = likelihood
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.prior_type = 1
        self.inf_dims = inf_dims
        self.use_final_layer_targets = use_final_layer_targets
        self.use_final_layer_noise = use_final_layer_noise
        self.nonlinearity = nonlinearity

    def trainable_prior(self, flag: bool, just_mean: bool = False):
        self.input_layer.prior.trainable(flag, just_mean=just_mean)
        self.prediction_head.prior.trainable(flag, just_mean=just_mean)
        for block in self.bdap_blocks:
                block.trainable_prior(flag, just_mean=just_mean)
        
    def forward(self, Xt, Xc=None, Yc=None, return_kl=False, num_samples=1, update_prev=False, save_stuff=False):

        # Xt shape (Nt, x_dim), Xc shape (Nc, x_dim)
        Xt = Xt.clone().unsqueeze(0).repeat((num_samples, 1, 1))
        if Xc is not None:
            Xc_rep = Xc.clone().unsqueeze(0).repeat((num_samples, 1, 1))
        else:
            Xc_rep = None

        il_outputs = self.input_layer(Xt, Xc_rep, Xc, Yc, return_kl=return_kl, num_samples=num_samples, update_prev=update_prev, save_stuff=save_stuff)

        cum_kl = torch.tensor(0.0)
        if return_kl:
            cum_kl += il_outputs[2]
        Xt_prev, Xc_prev = il_outputs[:2]

        for bdap_block in self.bdap_blocks:
            block_outputs = bdap_block(Xt_prev, Xc_prev, Xc, Yc, return_kl=return_kl, num_samples=num_samples, update_prev=update_prev, save_stuff=save_stuff)
            Xt_prev, Xc_prev = block_outputs[:2]
            if return_kl:
                cum_kl += block_outputs[2]
        
        ols = None
        if self.use_final_layer_noise:
            ols = self.likelihood.sigmas.log()
        ph_outputs = self.prediction_head(Xt_prev,
                                          Xc_prev,
                                          Xc,
                                          Yc,
                                          output_log_sigmas=ols,
                                          return_kl=return_kl,
                                          num_samples=num_samples,
                                          update_prev=update_prev,
                                          save_stuff=save_stuff
                                         )
        if return_kl:
            cum_kl += ph_outputs[2]

        Xt_final, Xc_final = ph_outputs[:2]
        
        return Xt_final, Xc_final, cum_kl
    

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
        