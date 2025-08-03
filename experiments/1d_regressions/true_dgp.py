import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import List, Optional, Tuple

import models
from utils.training import train_meta_model
from utils.data_utils import ctxt_trgt_split, obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_gp_dataset_please
from base_networks.base_architectures import Sin, SharpTanh





def sawtooth_function(x_range=[-2.0, 2.0], p=1.0, random_linear=False, random_shift=False, m=1, random_gradient=False, n=200):
    X = torch.linspace(*x_range, n)

    if random_gradient:
        m += torch.rand((1,)) * 2 + 0.5
    
    if random_shift:
        s = torch.rand(1) * p  # random shift in [0, p)
    else:
        s = 0

    f_x = m * (torch.remainder(X + s, p) - 0.5*p)

    if random_linear:
        f_x += torch.randn((1,)) * X / 3 + torch.randn((1,))*0.25

    return X, f_x



def heaviside_function(x_range=[-4.0, 4.0], l=1.0, n=200):
    X = torch.linspace(*x_range, n).unsqueeze(-1)

    Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square()) + torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    f_x -= f_x.mean()
    discrete_f_x = torch.where(f_x > 0, 1.0, -1.0)

    return X.squeeze(), discrete_f_x.squeeze()

def get_function_samples(function, num_samples=100, granularity=200):
    assert function.lower() in ['sawtooth', 'heaviside']
    if function.lower() == 'sawtooth':
        func_kwargs = {'x_range': [-2.0, 2.0],
                       'p': 0.75,
                       'random_linear': True,
                       'random_shift': False,
                       'm': 1.33,
                       'random_gradient': False,
                       'n': granularity}
        func = sawtooth_function
    else:
        func_kwargs = {'x_range': [-4.0, 4.0], 'l': 1, 'n': granularity}
        func = heaviside_function

    func_samps = [func(**func_kwargs) for _ in range(num_samples)]

    return  func_samps



def main(num_samples=100, granularity=200, use_gpu=False):

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + f"/figs/true_dgp").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/true_dgp/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/true_dgp/pngs").mkdir(parents=True, exist_ok=True)

    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)

    for function in ['sawtooth', 'heaviside']:

        func_samps = get_function_samples(function, num_samples=num_samples, granularity=granularity)
        if function.lower() == 'sawtooth':
            x_lim = [-2.0, 2.0]
            y_lim = [-2.0, 2.0]
        else:
            x_lim = [-4.0, 4.0]
            y_lim = [-4.0, 4.0]

        for (X, y) in func_samps:
            plt.plot(X.cpu(), y.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/true_dgp/pdfs/{function}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/true_dgp/pngs/{function}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="True Data-Generating-Process Samples")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of function samples to generate.')
    parser.add_argument('--granularity', type=int, default=200, help='Number of points at which to evaluate each function.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')

    args = parser.parse_args()
    main(**vars(args))