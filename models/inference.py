import torch
from torch import nn
from .tensors import stable_inversion

def compute_unitwise_posteriors(X, Y, log_sigmas, prior):
    # X should have already been passed through a nonlinearity
    # X is shape (samples, N, d_in+1)
    # Y and log_sigmas are both shape (N, d_out) or shape (samples, N, d_out) (the latter if inf_net_use_act==True)
    diagonal = False
    if isinstance(prior, torch.distributions.Normal):
        diagonal = True
    
    Lambda_d_l = (1 / ((2*log_sigmas).exp()+1e-6)).transpose(-2, -1).diag_embed() # shape (d_out, N, N) or (samples, d_out, N, N)
    if len(Lambda_d_l.shape) == 2:
        Lambda_d_l = Lambda_d_l.unsqueeze(0)
    mu_d_l = prior.mean # shape (d_out, d_in+1)

    if len(Y.shape) == 2:
        Y = Y.unsqueeze(0)

    if diagonal:
        Sigma_d_l = prior.variance.diag_embed() # shape (d_out, d_in+1, d_in+1)
        Sigma_d_l_inv = (1 / prior.variance).diag_embed() # same shape
    else:
        Sigma_d_l = prior.covariance_matrix
        Sigma_d_l_inv = stable_inversion(Sigma_d_l)

    if len(Sigma_d_l_inv) == 3:
        Sigma_d_l_inv = Sigma_d_l_inv.unsqueeze(0)

    S_d_l_inv = Sigma_d_l_inv + X.transpose(-2, -1).unsqueeze(1) @ Lambda_d_l @ X.unsqueeze(1) # shape (samples, d_out, d_in+1, d_in_+1)
    S_d_l = stable_inversion(S_d_l_inv) # shape (samples, d_out, d_in+1, d_in+1)
    m_d_l = (S_d_l @ (Sigma_d_l_inv @ mu_d_l.unsqueeze(-1) + X.transpose(-2, -1).unsqueeze(1) @ Lambda_d_l @ Y.transpose(-2, -1).unsqueeze(-1))).squeeze(-1)
    # m_d_l is shape (samples, d_out, d_in+2)
    return torch.distributions.MultivariateNormal(m_d_l, S_d_l) # object shape (samples, d_out, d_in+1)


def compute_layerwise_posterior(X, Y, log_sigmas, prior):
    # X should have already been passed through a nonlinearity
    # X is shape (samples, N, d_in)
    # Y and log_sigmas are both shape (N, d_out)

    lambdas = (1 / ((2*log_sigmas).exp()+1e-6))
    Lambda_d_l = lambdas.transpose(-2, -1).diag_embed() # shape (d_out, N, N)
    # below is only needed if we were to explicitly work with \chi
    # Lambda_l = lambdas.transpose(-2, -1).flatten().diag_embed() # shape (N*d_out, N*d_out)

    mu_l = prior.mean # shape (samples, (d_in+1)*d_out)
    Sigma_l = prior.covariance_matrix
    Sigma_l_inv = stable_inversion(Sigma_l) # shape (samples, (d_in+1)*d_out, (d_in+1)*d_out)

    Phi_d_l = X.transpose(-2, -1).unsqueeze(1) @ Lambda_d_l.unsqueeze(0) @ X.unsqueeze(1) # shape (samples, d_out, d_in+1, d_in+1)
    Phi_l = torch.cat([torch.block_diag(*Phi_d_l[i]).unsqueeze(0) for i in range(Phi_d_l.shape[0])], dim=0) # shape (samples, d_out*(d_in+1), d_out*(d_in+1))

    Y_tilde = Y * lambdas
    phi_l = (X.transpose(-2, -1) @ Y_tilde.unsqueeze(0)).flatten(start_dim=1) # shape (samples, (d_in+1)*d_out)

    S_l_inv = Sigma_l_inv + Phi_l
    S_l = stable_inversion(S_l_inv) # shape (samples, (d_in+1)*d_out, (d_in+1)*d_out)

    m_l = (S_l @ ((Sigma_l_inv @ mu_l.unsqueeze(-1)).squeeze(-1) + phi_l).unsqueeze(-1)).squeeze(-1) # shape (samples, (d_in+1)*d_out)

    return torch.distributions.MultivariateNormal(m_l, S_l) # object shape (samples, d_out*(d_in+1))
