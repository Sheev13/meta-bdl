import torch
import matplotlib.pyplot as plt
from tueplots import bundles
from tqdm import tqdm
import sys
from pathlib import Path
import neurokit2 as nk
import numpy as np
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import List, Optional, Tuple

import models
from utils.training import train_meta_model
from utils.data_utils import ctxt_trgt_split, obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_gp_dataset_please, obtain_me_a_nice_bnn_dataset_please
from utils.bnn_prior import GaussianBNNPrior
from base_networks.base_architectures import Sin, SharpTanh

plt.rcParams.update(bundles.iclr2024(nrows=1, ncols=1))


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

def bnn_function(x_range=[-4.0, 4.0], n=200, hidden_dims=[128, 128], scale_prior=True, nonlinearity=torch.nn.ReLU()):
    X = torch.linspace(*x_range, n).unsqueeze(-1)
    bnn_prior = GaussianBNNPrior(1, 1, hidden_dims, scale_prior=scale_prior, nonlinearity=nonlinearity)
    f_x = bnn_prior(X, num_samples=1).squeeze(0)

    return X.squeeze(), f_x.squeeze()

def ecg_function(
    fs=100,
    noise=0.001,
    n_range=None,
    x_range=None,
):
    """Generate one synthetic ECG waveform and return (X, Y) torch tensors."""
    d = 6.0
    signal = nk.ecg_simulate(
        duration=8.0,
        sampling_rate=fs,
        heart_rate=20,
        noise=noise,
        method='simple'
    )
    signal -= np.mean(signal)
    signal /= (2*np.max(signal))

    single_wave_block = signal[int(3.5*fs):int(6.5*fs)] # 3s long
    zeros_block = np.concatenate(3*[signal[int(7.0*fs):]], axis=0) # 3s long
    split = int(np.random.uniform(0.0, 3.0) * fs)
    full = np.concatenate((zeros_block[:split], single_wave_block, zeros_block[split:]), axis=0)

    # normalize and convert to tensors
    t = np.linspace(-d/2, d/2, len(full), endpoint=False)
    X = torch.tensor(t, dtype=torch.float64).unsqueeze(1)
    Y = torch.tensor(full, dtype=torch.float64).unsqueeze(1)

    if x_range is not None:
        gooduns = ((X >= min(x_range)) & (X <= max(x_range)))
        X = X[gooduns]
        Y = Y[gooduns]

    if n_range is not None:
        n = torch.randint(low=min(n_range), high=min(max(n_range), X.shape[0]), size=(1,))
        inds = torch.randperm(X.shape[0])[:n]
        return X[inds].unsqueeze(-1), Y[inds].unsqueeze(-1)

    return X, Y

def get_function_samples(function, num_samples=100, granularity=200):
    assert function.lower() in ['sawtooth', 'heaviside', 'bnn', 'ecg']
    if function.lower() == 'sawtooth':
        func_kwargs = {'x_range': [-2.0, 2.0],
                       'p': 0.75,
                       'random_linear': True,
                       'random_shift': False,
                       'm': 1.33,
                       'random_gradient': False,
                       'n': granularity}
        func = sawtooth_function
    elif function.lower() == 'heaviside':
        func_kwargs = {'x_range': [-4.0, 4.0], 'l': 1, 'n': granularity}
        func = heaviside_function
    elif function.lower() == 'bnn':
        func_kwargs = {'x_range': [-4.0, 4.0], 'n': granularity}
        func = bnn_function
    elif function.lower() == 'ecg':
        func_kwargs = {'noise': 0.0}
        func = ecg_function

    func_samps = [func(**func_kwargs) for _ in range(num_samples)]

    return  func_samps



def build_meta_dataset(num_datasets=10_000, n_range=[40, 100], function_type='sawtooth', x_range=[-5.0, 5.0], **bnn_kwargs):
    md = []
    assert function_type.lower() in ['sawtooth', 'ecg', 'heaviside', 'bnn']

    if function_type.lower() == 'sawtooth':
        dataset_func = obtain_me_a_nice_sawtooth_dataset_please
        data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
    elif function_type.lower() == 'heaviside':
        dataset_func = obtain_me_a_nice_heaviside_dataset_please
        data_hypers = {'x_range': x_range, 'l': 1, 'noise': 0.01}
    elif function_type.lower() == 'bnn':
        dataset_func = obtain_me_a_nice_bnn_dataset_please
        data_hypers = {'x_range': x_range, **bnn_kwargs}
    elif function_type.lower() == 'ecg':
        dataset_func = ecg_function
        data_hypers = {'x_range': x_range}

    for _ in range(num_datasets):
        X, y = dataset_func(n_range=n_range, **data_hypers)
        md.append((X, y))
    
    return md


def main():
    PATH = str(Path(__file__).resolve().parent)
    torch.set_default_dtype(torch.float64)

    total_size = bundles.iclr2024(nrows=3, ncols=5)["figure.figsize"]
    cell_width  = total_size[0] / (5/3)
    cell_height = total_size[1] / (3/3)

    Path(PATH + f"/figs/results").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/sawtooth").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/sawtooth/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/sawtooth/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/bnn").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/bnn/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/bnn/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/heaviside").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/heaviside/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/heaviside/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/ecg").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/ecg/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/results/ecg/pdfs").mkdir(parents=True, exist_ok=True)

    for codename, function_type in zip(['ursula', 'vincent', 'thomas', 'zadok'], ['sawtooth', 'bnn', 'heaviside', 'ecg']):
    # for codename, function_type in zip(['zadok'], ['ecg']):
        bdnp = torch.load(PATH + f'/saved_models/bdnp-{codename}', weights_only=False, map_location=torch.device('cpu'))

        if function_type == 'sawtooth':
            x_lim = [-2.0, 2.0]
            y_lim = [-2.0, 2.0]
        elif function_type == 'heaviside':
            x_lim = [-4.0, 4.0]
            y_lim = [-2.0, 2.0]
        elif function_type == 'bnn':
            x_lim = [-4.0, 4.0]
            y_lim = [-4.0, 4.0]
        elif function_type == 'ecg':
            x_lim = [-2.1, 1.1]
            y_lim = [-0.4, 0.7]

        xs = torch.linspace(x_lim[0], x_lim[1], 250).unsqueeze(-1)
        samps = 100
        
        # true DGP samples:
        func_samps = get_function_samples(function_type, num_samples=samps, granularity=250)
        fig, ax = plt.subplots(1, 1, figsize=(cell_width, cell_height))
        for (X, y) in func_samps:
            ax.plot(X.cpu(), y.cpu(), linewidth=0.5, color='C0', alpha=0.3)
        ax.grid()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        plt.savefig(PATH + f"/figs/results/{function_type}/pngs/dgp.png", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/results/{function_type}/pdfs/dgp.pdf", bbox_inches="tight")
        plt.close()

        # prior samples:
        with torch.no_grad():
            prior_samps = bdnp(xs, None, None, num_samples=samps)[0]

        fig, ax = plt.subplots(1, 1, figsize=(cell_width, cell_height))
        ax.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), prior_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.3)
        ax.grid()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        plt.savefig(PATH + f"/figs/results/{function_type}/pngs/prior-predictive.png", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/results/{function_type}/pdfs/prior-predictive.pdf", bbox_inches="tight")
        plt.close()

        # posterior stuff
        test_md = build_meta_dataset(num_datasets=5,
                                    n_range=[10, 11],
                                    function_type=function_type,
                                    x_range=x_lim,
                                    )
        print(f"Plotting {function_type} predictions...")
        for i, (X, y) in tqdm(enumerate(test_md)):
            Path(PATH + f"/figs/results/{function_type}/pngs/{i}").mkdir(parents=True, exist_ok=True)
            Path(PATH + f"/figs/results/{function_type}/pdfs/{i}").mkdir(parents=True, exist_ok=True)
            for j in range(1, 5):
                X_c, y_c = X.clone()[0:j,:], y.clone()[0:j,:]
                with torch.no_grad():
                    pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
                fig, ax = plt.subplots(1, 1, figsize=(cell_width, cell_height))
                ax.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.3)
                ax.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
                ax.grid()
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='both', which='both', length=0)
                plt.savefig(PATH + f"/figs/results/{function_type}/pngs/{i}/{j}-points.png", bbox_inches="tight")
                plt.savefig(PATH + f"/figs/results/{function_type}/pdfs/{i}/{j}-points.pdf", bbox_inches="tight")
                plt.close()
        print("All done.")

if __name__ == "__main__":
    main()