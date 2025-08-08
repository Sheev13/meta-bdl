import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import List, Optional, Tuple
from collections import defaultdict

import models
from models import baselines
from utils.training import train_meta_model, train_variational_model
from utils.bnn_prior import GaussianBNNPrior
from utils.data_utils import obtain_me_a_nice_bnn_dataset_please, obtain_me_a_nice_gp_dataset_please


    
def main(codename=None,
         model_name=None,
         dataset='bnn',
         hidden_dims=[20, 20],
         lml_mc_samples=10_000,
         num_sigma_ys=10,
         min_sigma_y=0.01,
         max_sigma_y=0.1,
         num_bdnp_datasets=1,
         seed=69,
         scale_prior=True,
         nonlinearity=nn.Tanh(),
         use_gpu=False,
         ):
    
    args_dict = locals()

    if codename is None:
        raise ValueError("User failed to specify a codename for this training run.")
    else:
        codename = codename.lower()

    if model_name is None:
        raise ValueError("User failed to specify which model to run ELBO experiment for.")
    elif model_name.lower() not in ['mfvi', 'ucvi', 'lcvi', 'fcvi', 'givi', 'bdnp', 'meta_bdnp', 'mc']:
        raise ValueError(f"User has specified an unrecognised model with which to run the ELBO experiment: '{model_name}'")
    
    if dataset.lower() not in ['bnn', 'gp']:
        raise ValueError("User failed to specify a valid dataset option out of 'bnn' or 'gp'.")
    

    if nonlinearity.lower() == 'relu':
        nonlinearity = torch.nn.ReLU()
    elif nonlinearity.lower() == 'tanh':
        nonlinearity = torch.nn.Tanh()
    elif nonlinearity.lower() == 'sigmoid':
        nonlinearity = torch.nn.Sigmoid()
    elif 'leaky' in nonlinearity.lower():
        nonlinearity = torch.nn.LeakyReLU()
    elif nonlinearity.lower() == 'swish' or nonlinearity.lower() == 'silu':
        nonlinearity = torch.nn.SiLU()
    else:
        raise NotImplementedError("Conversion to torch.nn module not yet implemented for provided nonlinearity string.")



    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)
        print("device type: ", device)

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(seed)
    if dataset == 'bnn':
        data_generating_func = obtain_me_a_nice_bnn_dataset_please
        data_generating_kwargs = {'x_range': [-4.0, 4.0],
                                  'noise': 0.05,
                                  'hidden_dims': hidden_dims,
                                  'scale_prior': scale_prior,
                                  'nonlinearity': nonlinearity}
    elif dataset == 'gp':
        data_generating_func = obtain_me_a_nice_gp_dataset_please
        data_generating_kwargs = {'x_range': [-4.0, 4.0],
                                  'noise': 0.05,
                                  'l': 1.0,
                                  'kernel': 'se',}
        
    X, Y = data_generating_func(n_range=[10, 11], **data_generating_kwargs)
        
    sigma_y_list = torch.linspace(min_sigma_y, max_sigma_y, num_sigma_ys)

    lik = models.GaussianLikelihood(y_dim=1, sigma_y=min_sigma_y, train=False)
    bnn_kwargs = {'x_dim': 1,
                  'y_dim': 1,
                  'hidden_dims': hidden_dims,
                  'likelihood': lik,
                  'scale_prior': scale_prior,
                  'nonlinearity': nonlinearity}
    
    bdnp_kwargs = {'x_dim': 1,
                   'y_dim': 1,
                   'hidden_dims': hidden_dims,
                   'prior_type': 0,
                   'likelihood': lik,
                   'inf_dims': hidden_dims, 
                   'use_final_layer_targets': True,
                   'use_final_layer_noise': True,
                   'scale_prior': scale_prior,
                   'nonlinearity': nonlinearity}
    
    if model_name == 'mfvi':
        model = baselines.MFVIBNN(**bnn_kwargs)
    elif model_name == 'ucvi':
        model = baselines.UCVIBNN(**bnn_kwargs)
    elif model_name == 'lcvi':
        model = baselines.LCVIBNN(**bnn_kwargs)
    elif model_name == 'fcvi':
        model = baselines.FCVIBNN(**bnn_kwargs)
    elif model_name == 'givi':
        model = baselines.GIVIBNN(**bnn_kwargs, num_inducing=5)
    elif model_name == 'bdnp' or model_name == 'meta_bdnp':
        model = models.BDNP(**bdnp_kwargs)
    elif model_name == 'mc':
        model = GaussianBNNPrior(**bnn_kwargs)

    if model_name == 'meta_bdnp':
        md = [data_generating_func(n_range=[5, 50], **data_generating_kwargs) for _ in range(num_bdnp_datasets)]
        training_alg = train_meta_model
        training_kwargs = {'md': md,
                           'batch_size': 5,
                           'learning_rate': 5e-3,
                           'final_learning_rate': 1e-3,
                           'num_samples': 8,
                           'loss_function': 'avi',
                           'device_agnostic': True}
    else:
        training_alg = train_variational_model
        training_kwargs = {'dataset': (X, Y),
                           'learning_rate': 1e-2,
                           'final_learning_rate': 5e-3,
                           'num_samples': 8,
                           'device_agnostic': True}

    results = defaultdict(list)
    training_metrics = defaultdict(list)

    for i, sigma_y in enumerate(sigma_y_list):
        print(f"On step {i} of {num_sigma_ys}")
        model.likelihood.raw_sigmas.data = sigma_y.log()

        if model_name != 'mc':
            if i == 0:
                training_stint = training_alg(model, training_steps=5_000, **training_kwargs)
            else:
                training_stint = training_alg(model, training_steps=1_000, **training_kwargs)

        for key, value in training_stint.items():
            training_metrics[key].extend(value)

        with torch.no_grad():
            if 'bdnp' in model_name.lower():
                neg_elbo, _ = model.loss(X, Y, X, Y, num_samples=10_000)
            
            if model_name == 'mc':
                lml = model.log_marginal_likelihood(X, Y, lml_mc_samples)
                results['mc'].append(lml.cpu().float())
            elif 'bdnp' in model_name.lower():
                neg_elbo, _ = model.loss(X, Y, X, Y, num_samples=10_000)
                results[model_name].append(-neg_elbo)
            else:
                neg_elbo, _ = model.loss(X, Y, num_samples=10_000)
                results[model_name].append(-neg_elbo)
    

    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    omitted_steps = 0
    for i, (key, value) in enumerate(training_metrics.items()):
        axes[i].plot(value[omitted_steps:])
        axes[i].set_xlabel(key)
        axes[i].grid()
        if key == 'elbo':
            axes[i].set_ylim([-5000, 500])
        elif key == 'e_ll':
            axes[i].set_ylim([-4000, 1000])
        elif key == 'kl':
            axes[i].set_ylim([0, 2000])

    plt.savefig(PATH + f"/figs/{codename}/pdfs/{model_name}-training.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/{codename}/pngs/{model_name}-training.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP experiment 2")
    parser.add_argument('--codename', type=str, default=None, help='Codename for training run')
    parser.add_argument('--model_name', type=str, default=None, help='Which model to run the experiment for.')
    parser.add_argument('--dataset', type=str, default='bnn', help='Type of function/dataset')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[20, 20], help='hidden layer dimensions of BNNs')
    parser.add_argument('--lml_mc_samples', type=int, default=10_000, help='Number of Monte Carlo samples for MC log marginal likelihood estimation')
    parser.add_argument('--num_sigma_ys', type=int, default=10, help='Number of different likelihood noise variances to evaluate.')
    parser.add_argument('--min_sigma_y', type=float, default=0.01, help='Lower value of sigma_y range.')
    parser.add_argument('--max_sigma_y', type=float, default=0.1, help='Upper value of sigma_y range.')
    parser.add_argument('--num_bdnp_datasets', type=int, default=100_000, help='Number of datasets in meta-dataset')
    parser.add_argument('--seed', type=int, default=69, help='Manually-specified random seed.')
    parser.add_argument('--scale_prior', action='store_true', help='Whether to use an input-dimension-scaled prior.')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')

    args = parser.parse_args()
    main(**vars(args))
        
