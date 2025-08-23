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
from utils.data_utils import ctxt_trgt_split, obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_gp_dataset_please, obtain_me_a_nice_bnn_dataset_please
from base_networks.base_architectures import Sin, SharpTanh


def build_meta_dataset(num_datasets=10_000, n_range=[40, 100], function_type='sawtooth', x_range=[-5.0, 5.0], **bnn_kwargs):
    md = []
    assert function_type.lower() in ['sawtooth', 'gp', 'heaviside', 'bnn']

    if function_type.lower() == 'sawtooth':
        dataset_func = obtain_me_a_nice_sawtooth_dataset_please
        data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
    elif function_type.lower() == 'gp':
        dataset_func = obtain_me_a_nice_gp_dataset_please
        data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': x_range}
    elif function_type.lower() == 'heaviside':
        dataset_func = obtain_me_a_nice_heaviside_dataset_please
        data_hypers = {'x_range': x_range, 'l': 1, 'noise': 0.01}
    elif function_type.lower() == 'bnn':
        dataset_func = obtain_me_a_nice_bnn_dataset_please
        data_hypers = {'x_range': x_range, **bnn_kwargs}

    for _ in range(num_datasets):
        X, y = dataset_func(n_range=n_range, **data_hypers)
        md.append((X, y))
    
    return md

def init_bdnp(architecture=[250, 250, 250], nonlinearity='relu', residual=False, trainable_likelihood_noise=True, init_likelihood_noise=0.1, transformer_layers=None, transformer_width=None, use_act=False):
    lik = models.GaussianLikelihood(y_dim=1, sigma_y=init_likelihood_noise, train=trainable_likelihood_noise, sigma_y_upper_bound=0.3)

    if nonlinearity.lower() == 'relu':
        nl = torch.nn.ReLU()
    elif nonlinearity.lower() == 'sin':
        nl = Sin()
    elif nonlinearity.lower() == 'tanh':
        nl = torch.nn.Tanh()
    elif nonlinearity.lower() == 'sigmoid':
        nl = torch.nn.Sigmoid()
    elif 'leaky' in nonlinearity.lower():
        nl = torch.nn.LeakyReLU()
    elif nonlinearity.lower() == 'swish' or nonlinearity.lower() == 'silu':
        nl = torch.nn.SiLU()
    elif nonlinearity.lower() == 'sharptanh':
        nl = SharpTanh()
    else:
        raise NotImplementedError("Conversion to torch.nn module not yet implemented for provided nonlinearity string.")

    bdnp = models.BDNP(x_dim=1,
                   y_dim=1,
                   hidden_dims=architecture,
                   prior_type=1,
                   likelihood=lik,
                   inf_dims=architecture, 
                   use_final_layer_targets=True,
                   use_final_layer_noise=False,
                   scale_prior=True,
                   nonlinearity=nl,
                   residual=residual,
                   inf_transformer_layers=transformer_layers,
                   inf_transformer_width=transformer_width,
                   inf_net_use_act=use_act
                   )
    
    return bdnp
    
def main(
        codename=None,
        num_datasets=10_000,
        n_range=[40, 100],
        function_type='sawtooth',
        architecture=[250, 250, 250],
        nonlinearity='relu',
        residual=False,
        trainable_likelihood_noise=True,
        init_likelihood_noise=0.2,
        transformer_layers=None,
        transformer_width=None,
        use_act=False,
        training_steps=10_000,
        batch_size=5,
        learning_rate=1e-3,
        final_learning_rate=5e-5,
        loss_function='p-avi',
        num_samples=1,
        release_prior_at_step=100,
        ctxt_proportion_range=(0.7, 0.9),
        train_new_model=False,
        use_gpu=True,
):
    args_dict = locals()

    if codename is None:
        raise ValueError("User failed to specify a codename for this training run.")
    else:
        codename = codename.lower()

    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)
        print("device type: ", device)
        # model.to(device, dtype=torch.float32)
        # print("Moving dataset to device...")
        # md = [(X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)) for (X, y) in md]
        # print("Done.")

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)

    if train_new_model:
        with open(PATH + f"/training-configs/{codename}-config.json", 'w') as f:
            json.dump(args_dict, f, indent=4)

        bnn_kwargs = {}
        if function_type.lower() == 'bnn':
            bnn_kwargs = {'nonlinearity': torch.nn.ReLU(), 'hidden_dims': architecture}

        md = build_meta_dataset(
            num_datasets=num_datasets,
            n_range=n_range,
            function_type=function_type,
            **bnn_kwargs,
        )

        bdnp = init_bdnp(
            architecture=architecture,
            nonlinearity=nonlinearity,
            residual=residual,
            trainable_likelihood_noise=trainable_likelihood_noise,
            init_likelihood_noise=init_likelihood_noise,
            transformer_layers=transformer_layers,
            transformer_width=transformer_width,
            use_act=use_act,
        )

        training_metrics = train_meta_model(
            bdnp,
            md,
            training_steps=training_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_learning_rate=final_learning_rate,
            num_samples=num_samples,
            loss_function=loss_function,
            release_prior_at_step=release_prior_at_step,
            ctxt_proportion_range=ctxt_proportion_range,
            device_agnostic=True,
        )

        torch.save(bdnp, PATH + f'/saved_models/bdnp-{codename}')

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

        plt.savefig(PATH + f"/figs/{codename}/pdfs/training.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/training.png", bbox_inches="tight")
        plt.close()



    else: # use an already-trained model
        bdnp = torch.load(PATH + f'/saved_models/bdnp-{codename}', weights_only=False)

    
    xs = torch.linspace(-4.0, 4.0, 250).unsqueeze(-1)
    samps = 100

    if function_type == 'sawtooth':
        x_lim = [-2.0, 2.0]
        y_lim = [-2.0, 2.0]
    else:
        x_lim = [-4.0, 4.0]
        y_lim = [-4.0, 4.0]

    # prior samples:
    with torch.no_grad():
        prior_samps = bdnp(xs, None, None, num_samples=samps)[0]

    plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), prior_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
    plt.grid()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig(PATH + f"/figs/{codename}/pdfs/prior-predictive.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/{codename}/pngs/prior-predictive.png", bbox_inches="tight")
    plt.close()

    # single-datapoint samples:
    test_md = build_meta_dataset(num_datasets=5,
                                 n_range=[1, 2],
                                 function_type=function_type,
                                 x_range=[-3.5, 3.5]
                                 )
    for i, (X, y) in enumerate(test_md):
        X_c, y_c = X.clone(), y.clone()
        with torch.no_grad():
            pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
        
        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/{codename}/pdfs/one-point-predictive-{i}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/one-point-predictive-{i}.png", bbox_inches="tight")
        plt.close()

    # multiple-datapoint samples:
    n_range=[2, 10]
    test_md = build_meta_dataset(num_datasets=10,
                                 n_range=n_range,
                                 function_type=function_type,
                                 x_range=[-4.0, 4.0]
                                 )
    for i, (X, y) in enumerate(test_md):
        X_c, y_c = X.clone(), y.clone()
        with torch.no_grad():
            pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
        
        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/{codename}/pdfs/multi-point-predictive-{i}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/multi-point-predictive-{i}.png", bbox_inches="tight")
        plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP experiment 1")
    parser.add_argument('--codename', type=str, default=None, help='Codename for training run')
    parser.add_argument('--num_datasets', type=int, default=100_000, help='Number of datasets in meta-dataset')
    parser.add_argument('--n_range', type=int, nargs='+', default=[40, 100], help='Range of datapoints in each dataset')
    parser.add_argument('--function_type', type=str, default='sawtooth', help='Type of function/dataset')
    parser.add_argument('--architecture', type=int, nargs='+', default=[250, 250, 250], help='Hidden layer dims of BDNP and inference nets')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')
    parser.add_argument('--residual', action='store_true', help='Is the primary BDNP network residual?')
    parser.add_argument('--trainable_likelihood_noise', action='store_true', help='Whether the likelihood noise is learned or not.')
    parser.add_argument('--init_likelihood_noise', type=float, default=0.2, help='(initial) value of the likelihood noise std.')
    parser.add_argument('--transformer_layers', type=int, default=None, help='Number of attention blocks in AttBDNP inference nets')
    parser.add_argument('--transformer_width', type=int, default=None, help='Representation dimension of AttBNDP inference nets')
    parser.add_argument('--use_act', action='store_true', help='Pass current layer activations to inference nets?')
    parser.add_argument('--training_steps', type=int, default=20_000, help='The number of training steps')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of datasets used to estimate objective at each step')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--loss_function', type=str, default='p-avi', help='Objective function (vi or npvi)')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of MC samples to estimate expected log likelihood.')
    parser.add_argument('--release_prior_at_step', type=int, default=0, help='Training step at which prior parameters start being optimised')
    parser.add_argument('--ctxt_proportion_range', type=float, nargs='+', default=[0.1, 0.9], help='Range of context set/full set proportion for each sampled task')
    parser.add_argument('--train_new_model', action='store_true', help='Train a new BDNP, or load a pre-trained one.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')

    args = parser.parse_args()
    main(**vars(args))