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
         learning_rate=1e-2,
         final_learning_rate=1e-3,
         num_sigma_ys=10,
         min_sigma_y=0.01,
         max_sigma_y=0.1,
         num_bdnp_datasets=1,
         seed=69,
         scale_prior=True,
         nonlinearity=nn.Tanh(),
         use_gpu=False,
         use_shared_dataset=False,
         ):
    
    args_dict = locals()

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + f"/run_configs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/run_configs/{seed}").mkdir(parents=True, exist_ok=True)
    with open(PATH + f"/run_configs/{seed}/{codename}.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

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
    else:
        device = torch.device('cpu')
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    print("device type: ", device)

    Path(PATH + "/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{seed}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{seed}/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{seed}/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{seed}/{codename}/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/data").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/results").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/results/{seed}").mkdir(parents=True, exist_ok=True)

    if dataset == 'bnn':
        data_generating_func = obtain_me_a_nice_bnn_dataset_please
        data_generating_kwargs = {'x_range': [-4.0, 4.0],
                                'noise': 0.1,
                                'hidden_dims': hidden_dims,
                                'scale_prior': scale_prior,
                                'nonlinearity': nonlinearity}
    elif dataset == 'gp':
        data_generating_func = obtain_me_a_nice_gp_dataset_please
        data_generating_kwargs = {'x_range': [-4.0, 4.0],
                                'noise': 0.1,
                                'l': 1.0,
                                'kernel': 'se',}
    
    if use_shared_dataset:
        data = torch.load(PATH + f"/shared_datasets/{dataset}.pt", weights_only=False)
        X, Y = data["X"].to(device=device, dtype=torch.float64), data["Y"].to(device=device, dtype=torch.float64)
    else:
        torch.manual_seed(69) # constant seed for this part only to ensure dame dataset every time
        if use_gpu:
            torch.cuda.manual_seed(69)
        X, Y = data_generating_func(n_range=[21, 42], **data_generating_kwargs)

    plt.scatter(X.cpu(), Y.cpu(), color='C1', zorder=10000)
    plt.grid()
    plt.xlim([-4.0, 4.0])
    plt.ylim([-4.0, 4.0])
    plt.savefig(PATH + f"/figs/data/{model_name}-{dataset}.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/data/{model_name}-{dataset}.png", bbox_inches="tight")
    plt.close()
        
    sigma_y_list = torch.logspace(torch.tensor(min_sigma_y).log10(), torch.tensor(max_sigma_y).log10(), num_sigma_ys)

    if 'bdnp' in model_name.lower():
        model_kwargs = {'x_dim': 1,
                        'y_dim': 1,
                        'hidden_dims': hidden_dims,
                        'prior_type': 0,
                        'inf_dims': [50, 50],
                        'use_final_layer_targets': True,
                        'use_final_layer_noise': False,
                        'scale_prior': scale_prior,
                        'nonlinearity': nonlinearity}
    else:
        model_kwargs = {'x_dim': 1,
                        'y_dim': 1,
                        'hidden_dims': hidden_dims,
                        'scale_prior': scale_prior,
                        'nonlinearity': nonlinearity}
        if model_name.lower() == 'givi':
            model_kwargs['num_inducing'] = 10
    
    if model_name == 'mfvi':
        model_class = baselines.MFVIBNN
    elif model_name == 'ucvi':
        model_class = baselines.UCVIBNN
    elif model_name == 'lcvi':
        model_class = baselines.LCVIBNN
    elif model_name == 'fcvi':
        model_class = baselines.FCVIBNN
    elif model_name == 'givi':
        model_class = baselines.GIVIBNN
    elif 'bdnp' in model_name.lower():
        model_class = models.BNNP
    elif model_name == 'mc':
        model_class = GaussianBNNPrior

    torch.manual_seed(seed) # vary seed here for model training variability
    if use_gpu:
        torch.cuda.manual_seed(seed)

    if model_name == 'meta_bdnp':
        training_alg = train_meta_model
        training_kwargs = {'md': None, # meta dataset passed in the training loop below.
                           'batch_size': 5,
                           'learning_rate': learning_rate,
                           'final_learning_rate': final_learning_rate,
                           'num_samples': 8,
                           'loss_function': 'pp-avi',
                           'ctxt_proportion_range': [0.7, 0.9],
                           'device_agnostic': True}
    else:
        training_alg = train_variational_model
        training_kwargs = {'dataset': (X, Y),
                           'learning_rate': learning_rate,
                           'final_learning_rate': final_learning_rate,
                           'num_samples': 8,
                           'device_agnostic': True}
        if model_name.lower() == 'givi':
            training_kwargs['retain_graph'] = True

    results = {}

    print(f"Model type: {model_name}. Seed: {seed}")
    # core experiment here, i.e. actually train and evaluate models.
    for i, sigma_y in enumerate(sigma_y_list):
        print(f"On step {i+1} of {num_sigma_ys}. sigma_y={sigma_y}")

        lik = models.GaussianLikelihood(y_dim=1, sigma_y=sigma_y, train=False)

        if model_name != 'mc':
            model_kwargs['likelihood'] = lik
            model = model_class(**model_kwargs)
            if model_name.lower() == 'givi':
                model.init_inducing_points(X)
            if model_name.lower() == 'meta_bdnp':
                md = [data_generating_func(n_range=[5, 50], **data_generating_kwargs) for _ in range(num_bdnp_datasets)]
                training_kwargs['md'] = md
            training_stint = training_alg(model, training_steps=20_000, **training_kwargs)
            

            fig, axes = plt.subplots(1, len(training_stint), figsize=(3*len(training_stint), 1))
            omitted_steps = 0
            for j, (key, value) in enumerate(training_stint.items()):
                axes[j].set_title(f"sigma_y={sigma_y.item()}")
                axes[j].plot(value[omitted_steps:])
                axes[j].set_xlabel(key)
                axes[j].grid()
                if key == 'elbo':
                    axes[j].set_ylim([-1000, 100])
                elif key == 'e_ll':
                    axes[j].set_ylim([-1000, 200])
                elif key == 'kl':
                    axes[j].set_ylim([0, 500])

            plt.savefig(PATH + f"/figs/training/{seed}/{codename}/pdfs/stint_{i}.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/figs/training/{seed}/{codename}/pngs/stint_{i}.png", bbox_inches="tight")
            plt.close()


        # collect ELBO/LML results
        with torch.no_grad():
            k = sigma_y.item()
            if model_name == 'mc':
                if i < 6:
                    s = 100_000_000 # crank this up to 100M or so when using cluster.
                else:
                    s = 10_000_000 # for larger sigma_y fewer samples are fine, even as few as e.g. 10k.
                model_kwargs['likelihood'] = lik
                model = model_class(**model_kwargs)
                lml = model.log_marginal_likelihood(X, Y, s)
                results[k] = lml.cpu().item()
            elif 'bdnp' in model_name.lower():
                neg_elbo, _ = model.loss(X, Y, X, Y, num_samples=10_000)
                results[k] = -neg_elbo.cpu().item()
            else:
                neg_elbo, _ = model.loss(X, Y, num_samples=10_000)
                results[k] = -neg_elbo.cpu().item()
        print(f"Result: {results[k]}")


    with open(PATH + f"/results/{seed}/{codename}.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP experiment 2")
    parser.add_argument('--codename', type=str, default=None, help='Codename for training run')
    parser.add_argument('--model_name', type=str, default=None, help='Which model to run the experiment for.')
    parser.add_argument('--dataset', type=str, default='bnn', help='Type of function/dataset')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[20, 20], help='hidden layer dimensions of BNNs')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--num_sigma_ys', type=int, default=40, help='Number of different likelihood noise variances to evaluate.')
    parser.add_argument('--min_sigma_y', type=float, default=0.01, help='Lower value of sigma_y range.')
    parser.add_argument('--max_sigma_y', type=float, default=10.0, help='Upper value of sigma_y range.')
    parser.add_argument('--num_bdnp_datasets', type=int, default=50_000, help='Number of datasets in meta-dataset')
    parser.add_argument('--seed', type=int, default=69, help='Manually-specified random seed.')
    parser.add_argument('--scale_prior', action='store_true', help='Whether to use an input-dimension-scaled prior (defaults to False).')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')
    parser.add_argument('--use_shared_dataset', action='store_true', help='Load pre-made dataset rather than constructing one from scratch (defaults to False).')

    args = parser.parse_args()
    main(**vars(args))
        
