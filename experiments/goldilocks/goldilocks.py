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
from base_networks.base_architectures import Sin, SharpTanh


def init_bdnp(architecture=[64, 64, 64], nonlinearity='silu'):
    lik = models.GaussianLikelihood(y_dim=1, sigma_y=0.1, train=True)

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

    bdnp = models.BDNP(x_dim=3,
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

def main(model: str = None,
         dataset: str = None,
         prior_trainability: float = None,
         training_steps: int = 30_000,
         learning_rate: float = 5e-3,
         final_learning_rate: float = 5e-5,
         seed: int = None,
         use_gpu: bool = False):
    # args_dict = locals()

    model = model.lower()
    assert model in ['swag', 'mfvi', 'givi', 'bdnp', 'np', 'tnp', 'anp'] # eventually include convnp

    if model is None:
        raise ValueError("User failed to specify which model to use.")
    if dataset is None:
        raise ValueError("User failed to specify which dataset to use.")
    if seed is None:
        raise ValueError("User failed to specify which seed to use.")
    if (model == 'bdnp') and (prior_trainability is None):
        raise ValueError("User failed to specify how much of the BDNP prior to train.")

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    model_codename = model
    if model == 'bdnp':
        model_codename += f"_{prior_trainability}"

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}/pngs").mkdir(parents=True, exist_ok=True)

    ##### done to here #####
    # define model classes, model kwargs, training funcs, training kwwargs, seed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goldilocks experiment.")
    parser.add_argument('--model', type=str, default=None, help='run codename')
    parser.add_argument('--prior_trainability', type=float, default=None, help='Proportion of weights whose prior is trainable.')
    parser.add_argument('--training_steps', type=int, default=30_000, help='The number of training steps')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--seed', type=int, default=None, help='Seed number for repeat trials.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one available. Default False.')

    args = parser.parse_args()
    main(**vars(args))