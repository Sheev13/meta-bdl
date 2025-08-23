import torch
from torch import nn
from models.baselines.mcmc_bnn.base_mcmc_mlp import MCMC_BNN
from tqdm import tqdm
from collections import defaultdict
from typing import Optional
import warnings
import sys

def subsample(X: torch.Tensor, Y: torch.Tensor, b: int):
    n = X.shape[-1] # number of datapoints
    inds = torch.randperm(n)
    batch_inds = inds[:b]
    return X[batch_inds], Y[batch_inds]

def run_mcmc(model: MCMC_BNN,
            X: torch.Tensor,
            Y: torch.Tensor,
            algorithm: None,
            steps: int = 1000,
            step_size: float = 1e-4,
            minibatch_size: Optional[int] = None, # whether to do minibatching, and if so what size batches
            metropolis_adjusted: bool = False, # include accept-reject step or not
            **hmc_kwargs,
            ):
    if algorithm is None:
        raise ValueError("User failed to specify which type of MCMC to perform.")
    else:
        if algorithm.lower() not in ['hmc', 'lmc']:
            raise ValueError("User failed to specify a valid MCMC algorithm. Options are HMC or LMC.")
    if algorithm.lower() == 'hmc':
        hmc = True
    else:
        hmc = False
        hmc_kwargs = {}

    if minibatch_size is not None:
        if metropolis_adjusted:
            warnings.warn("User has tried to do stochastic gradient MCMC while also including MH accept-reject step.\nDefaulting to omit the accept-reject step.")
            metropolis_adjusted = False
        full_X, full_Y = X.clone(), Y.clone()

    num_w = model.num_weights
    posterior_samples = torch.zeros((steps, num_w))
    acceptance_counter = 0
    W = model.sample_from_prior()
    if hmc:
        P = torch.randn_like(W)

    iter_pbar = tqdm(range(steps), file=sys.stdout)
    tracker = defaultdict(list)

    for step in iter_pbar:
        if minibatch_size is not None:
            X, Y = subsample(full_X, full_Y, minibatch_size)
        metrics = defaultdict(float)
        current_stuff = [W]
        if hmc:
            current_stuff.append(P)
        proposed_stuff = model.get_proposal(X, Y, *current_stuff, step_size=step_size, **hmc_kwargs)
        if metropolis_adjusted:
            log_alpha = model.compute_log_acceptance(
                X, Y, *current_stuff, *proposed_stuff, step_size=step_size
            )
            u = torch.rand((1,))
            if u < log_alpha.exp():
                # accept the sample
                posterior_samples[step] = proposed_stuff[0]
                acceptance_counter += 1
            else:
                # reject the sample
                posterior_samples[step] = current_stuff[0]
        else:
            posterior_samples[step] = proposed_stuff[0]
            acceptance_counter += 1

        with torch.no_grad():
            metrics["log-lik"] = model.log_likelihood(X, Y, posterior_samples[step]).item()
            metrics["log-prior"] = model.log_prior(W).item()
            metrics["log potential"] = - (metrics["log-lik"] + metrics["log-prior"])
        
            if metropolis_adjusted:
                metrics["average acceptance"] = acceptance_counter / (step + 1)
        iter_pbar.set_postfix(metrics)

        for key, value in metrics.items():
            tracker[key].append(float(value))

    return posterior_samples, tracker