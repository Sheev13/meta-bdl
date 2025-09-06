import torch
from torch import nn
from models.baselines.swag_bnn.swag_mlp import SWAG_BNN
from tqdm import tqdm
from collections import defaultdict
from typing import Optional
import warnings
import sys

def pretrain(model: SWAG_BNN,
             X: torch.Tensor,
             Y: torch.Tensor,
             training_steps: int = 10_000,
             learning_rate: float = 1e-2,
):
    w_sample = model.sample_from_prior()
    W = nn.Parameter(w_sample, requires_grad=True)
    optimiser = torch.optim.Adam([W], lr=learning_rate)
    tracker = defaultdict(list)
    pbar = tqdm(range(training_steps), file=sys.stdout)

    for step in pbar:
        optimiser.zero_grad()
        neg_loss, metrics = model.log_posterior(X, Y, W)
        loss = - neg_loss
        
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                if p.grad.data.isnan().any():
                    p.grad.data = torch.nan_to_num(p.grad.data)
                    warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")

        # perform gradient update step.
        optimiser.step()      

        # store metrics.
        for key, value in metrics.items():
            tracker[key].append(float(value))

        pbar.set_postfix(metrics)

    model.set_pretrained(W.detach())

    return tracker

def run_SWAG(model: SWAG_BNN,
         X: torch.Tensor,
         Y: torch.Tensor,
         learning_rate: float = 5e-2,
         swa_steps: int = 100, # number of times we update SWA statistics
         c: int = 25, # frequency for stochastic weight updates
         ):
    W_bar = model.W_pretrained.clone()
    W_2_bar = model.W_pretrained.clone().pow(2)
    W = nn.Parameter(model.W_pretrained.clone(), requires_grad=True)
    D_list = []

    optimiser = torch.optim.SGD([W], lr=learning_rate)
    tracker = defaultdict(list)

    if swa_steps < model.K:
        raise ValueError("User wants to perform fewer SWA updates than the rank of SWAG approximate posterior. For me, impossible.")

    training_steps = c * swa_steps
    pbar = tqdm(range(training_steps), file=sys.stdout)

    for i in pbar:
        optimiser.zero_grad()
        neg_loss, metrics = model.log_posterior(X, Y, W)
        loss = - neg_loss
        loss.backward()
        if W.grad is not None:
            if W.grad.data.isnan().any():
                W.grad.data = torch.nan_to_num(W.grad.data)
                warnings.warn("Warning: NaN gradients encountered. Proceeded by setting them to zero.")
        torch.nn.utils.clip_grad_norm_([W], 10.0) 
        optimiser.step() 

        if i % c == 0:
            W_curr = W.detach().clone()
            if W_curr.isnan().any():
                W_curr = torch.nan_to_num(W_curr, nan=0.0, posinf=1e3, neginf=-1e3)
            n = (i / c) + 1 # + 1 since we already include the pretrained weights.
            W_bar = (n * W_bar + W_curr) / (n+1)
            W_2_bar = (n * W_2_bar + W_curr.pow(2)) / (n+1)
            if len(D_list) == model.K:
                D_list.pop(0)
            D_list.append(W_curr - W_bar)

        # store metrics.
        for key, value in metrics.items():
            if key == 'swa_step':
                continue
            tracker[key].append(float(value))

        metrics['swa_step'] = n
        pbar.set_postfix(metrics)

    model.W_swa.data = W_bar
    model.Sigma_diag.data = W_2_bar - W_bar.pow(2)
    model.Sigma_diag.data += 1e-5 # jitter
    model.D.data = torch.stack(D_list).T # shape (num_weights, K)

    return tracker

