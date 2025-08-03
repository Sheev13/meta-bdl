from typing import List, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from utils.bnn_prior import GaussianBNNPrior

class MetaDataset(Dataset):
    def __init__(self, datasets: List[Any]):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]


def ctxt_trgt_split(X: torch.Tensor, y: torch.Tensor, ctxt_proportion_range: Optional[Tuple[float]]=None, ctxt_proportion: Optional[float] = None):
    if ctxt_proportion is None:
        if ctxt_proportion_range[1] < ctxt_proportion_range[0]:
            ctxt_proportion_range = ctxt_proportion_range[::-1]
        if ctxt_proportion_range[0] < 0.0:
            raise ValueError("Cannot have a negative proportion of context points.")
        if ctxt_proportion_range[1] > 1.0:
            raise ValueError("Cannot have a proportion of context points that is greater than 1.")
        
        proportion = torch.rand((1,)) * (ctxt_proportion_range[1] - ctxt_proportion_range[0]) + ctxt_proportion_range[0]
    
    else:
        if ctxt_proportion < 0.0:
            raise ValueError("Cannot have a negative proportion of context points.")
        if ctxt_proportion > 1.0:
            raise ValueError("Cannot have a proportion of context points that is greater than 1.")
        
        proportion = ctxt_proportion

    num_ctxt = int(X.shape[0] * proportion)
    inds = torch.randperm(X.shape[0])
    ctxt_i = inds[:num_ctxt]
    trgt_i = inds[num_ctxt:]

    X_c, y_c = X[ctxt_i], y[ctxt_i]
    X_t, y_t = X[trgt_i], y[trgt_i]

    if X_c.shape[0] == 0:
        X_c, y_c = X_t[0], y_t[0]
    elif X_t.shape[0] == 0:
        X_t, y_t = X_c[0], y_c[0]

    return (X_c, y_c, X_t, y_t)


def obtain_me_a_nice_sawtooth_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, noise_range=None, p=1.0, p_range=None, random_linear=False, random_shift=False, m=1, random_gradient=False):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))

    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    if random_gradient:
        m += torch.rand((1,)) * 2 + 0.5 

    if noise_range is not None:
        noise = torch.rand((1,)) * (max(noise_range) - min(noise_range)) + min(noise_range)
    if p_range is not None:
        p = torch.rand((1,)) * (max(p_range) - min(p_range)) + min(p_range)
    
    if random_shift:
        s = torch.rand(1) * p  # random shift in [0, p)
    else:
        s = 0
    f_x = m * (torch.remainder(X + s, p) - 0.5*p)

    if random_linear:
        f_x += torch.randn((1,)) * X / 3 + torch.randn((1,))*0.25

    y = f_x + torch.randn_like(f_x) * noise
    return X, y

def obtain_me_a_nice_heaviside_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, l=1.0):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)
    Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square()) + torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    f_x -= f_x.mean()
    discrete_f_x = torch.where(f_x > 0, 1.0, -1.0)
    return X, discrete_f_x + torch.randn_like(f_x) * noise

def obtain_me_a_nice_gp_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, l=1.0, noise_range=None, l_range=None, kernel='se', p=1.0, p_range=None, binary_2d=False):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    if binary_2d:
        X = torch.rand((n, 2)) * (max(x_range) - min(x_range)) + min(x_range)
    else:
        X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    if noise_range is not None:
        noise = torch.rand((1,)) * (max(noise_range) - min(noise_range)) + min(noise_range)
    if l_range is not None:
        l = torch.rand((1,)) * (max(l_range) - min(l_range)) + min(l_range)
    if p_range is not None:
        p = torch.rand((1,)) * (max(p_range) - min(p_range)) + min(p_range)
    
    if kernel == 'se':
        Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square())
    elif kernel == 'per':
        Sigma = torch.exp(-2.0 * torch.sin(torch.pi * torch.cdist(X, X) / p).square() / l**2) 
    Sigma += torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    if binary_2d:
        y = torch.distributions.Bernoulli(logits=f_x * 2.5).sample()
    else:
        y = f_x + torch.randn_like(f_x) * noise
    return X, y

def obtain_me_a_nice_bnn_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, hidden_dims=[20, 20], scale_prior=True, nonlinearity=torch.nn.Tanh()):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    bnn_prior = GaussianBNNPrior(1, 1, hidden_dims, scale_prior=scale_prior, nonlinearity=nonlinearity)
    f_x = bnn_prior(X, num_samples=1).squeeze(0)
    y = f_x + torch.randn_like(f_x) * noise

    return X, y
