import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import List
from utils.data_utils import obtain_me_a_nice_bnn_dataset_please, obtain_me_a_nice_gp_dataset_please


def main(dataset: str = 'bnn', hidden_dims: List[int] = [20, 20], scale_prior: bool = True, nonlinearity: str = 'relu'):    

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

    PATH = str(Path(__file__).resolve().parent)

    torch.manual_seed(69) # constant seed for this part only to ensure dame dataset every time
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
        
    X, Y = data_generating_func(n_range=[21, 42], **data_generating_kwargs)
    plt.scatter(X.cpu(), Y.cpu(), color='C1', zorder=10000)
    plt.grid()
    plt.xlim([-4.0, 4.0])
    plt.ylim([-4.0, 4.0])
    plt.savefig(PATH + f"/figs/data/shared-{dataset}-dataset.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/data/shared-{dataset}-dataset.png", bbox_inches="tight")
    plt.close()

    # save data
    Path(PATH + f"/shared_datasets").mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "Y": Y}, PATH + f"/shared_datasets/{dataset}.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data for BDNP experiment 2")
    parser.add_argument('--dataset', type=str, default='bnn', help='Type of function/dataset')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[20, 20], help='hidden layer dimensions of BNNs')
    parser.add_argument('--scale_prior', action='store_false', help='Whether to use an input-dimension-scaled prior (defaults to True).')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')

    args = parser.parse_args()
    main(**vars(args))