import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json

import models
from utils.training import train_meta_model
from utils.data_utils import obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_gp_dataset_please
from base_networks.base_architectures import Sin, SharpTanh


def build_meta_dataset(num_datasets=10_000, n_range=[40, 100], function_type='sawtooth', x_range=[-5.0, 5.0]):
    md = []
    assert function_type.lower() in ['sawtooth', 'gp', 'heaviside']

    if function_type.lower() == 'sawtooth':
        dataset_func = obtain_me_a_nice_sawtooth_dataset_please
        data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
    elif function_type.lower() == 'gp':
        dataset_func = obtain_me_a_nice_gp_dataset_please
        data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': x_range}
    elif function_type.lower() == 'heaviside':
        dataset_func = obtain_me_a_nice_heaviside_dataset_please
        data_hypers = {'x_range': x_range, 'l': 1, 'noise': 0.01}

    for _ in range(num_datasets):
        X, y = dataset_func(n_range=n_range, **data_hypers)
        md.append((X, y))
    
    return md

def init_bdnp(architecture=[250, 250, 250], nonlinearity='relu'):
    lik = models.GaussianLikelihood(y_dim=1, sigma_y=0.05, train=False, sigma_y_upper_bound=0.5)

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
                   )
    
    return bdnp
    
def main(num_datasets=100_000,
         function_type='sawtooth',
         architecture=[48, 48],
         training_steps=20_000,
         learning_rate=5e-3,
         final_learning_rate=5e-5,
         loss_function='pp-avi',
         use_gpu=True,
        ):
    args_dict = locals()

    codename = function_type.lower() + "".join([f'_{i}' for i in architecture]) # e.g. 'sawtooth_48_48'

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    
    print(f"Learning prior for {function_type} data.")

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + "/saved_models").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/training_configs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)

    with open(PATH + f"/training_configs/{codename}-config.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    md = build_meta_dataset(
        num_datasets=num_datasets,
        n_range=[40, 100],
        function_type=function_type
    )

    if function_type == 'sawtooth':
        nl = 'relu'
    elif function_type == 'gp':
        nl = 'silu'
    elif function_type == 'heaviside':
        nl = 'tanh'
        
    bdnp = init_bdnp(
        architecture=architecture,
        nonlinearity=nl,
    )
    bdnp.trainable_prior(True)

    training_metrics = train_meta_model(
        bdnp,
        md,
        training_steps=training_steps,
        batch_size=5,
        learning_rate=learning_rate,
        final_learning_rate=final_learning_rate,
        num_samples=32,
        loss_function=loss_function,
        ctxt_proportion_range=(0.1, 0.6),
        device_agnostic=True,
    )

    torch.save(bdnp, PATH + f'/saved_models/{codename}')

    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    omitted_steps = 100
    for i, (key, value) in enumerate(training_metrics.items()):
        axes[i].plot(value[omitted_steps:])
        axes[i].set_xlabel(key)
        axes[i].grid()

    plt.savefig(PATH + f"/figs/{codename}/pdfs/training.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/{codename}/pngs/training.png", bbox_inches="tight")
    plt.close()

    
    xs = torch.linspace(-4.0, 4.0, 250).unsqueeze(-1)
    samps = 100

    if function_type == 'sawtooth':
        x_lim = [-2.0, 2.0]
        y_lim = [-2.0, 2.0]
    else:
        x_lim = [-4.0, 4.0]
        y_lim = [-4.0, 4.0]
    if function_type == 'heaviside': 
        y_lim = [-2.0, 2.0]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior transfer experiment")
    parser.add_argument('--num_datasets', type=int, default=100_000, help='Number of datasets in meta-dataset')
    parser.add_argument('--function_type', type=str, default='sawtooth', help='Type of function/dataset')
    parser.add_argument('--architecture', type=int, nargs='+', default=[48, 48], help='Hidden layer dims of BDNP and inference nets')
    parser.add_argument('--training_steps', type=int, default=100_000, help='The number of training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--loss_function', type=str, default='pp-avi', help='Objective function (vi or npvi)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')

    args = parser.parse_args()
    main(**vars(args))