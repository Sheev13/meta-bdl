import torch
from typing import List

def autocorrelation(samples: torch.Tensor, lag: int = 0) -> torch.Tensor:
    # samples should have shape (num_samples, num_weights)
    means = torch.mean(samples, dim=0)
    variances = torch.var(samples, dim=0)
    lagged_samples = torch.roll(samples, -lag, 0)
    covariances = ((samples - means) * (lagged_samples - means)).mean(dim=0)
    autocorrelations = covariances / variances
    return autocorrelations

def autocorrelation_array(samples: torch.Tensor, max_lag: int = 30) -> List:
    with torch.no_grad():
        return [autocorrelation(samples, lag=lag).mean().item() for lag in range(max_lag)]