import torch
import matplotlib.pyplot as plt
from tueplots import bundles
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
from utils.bnn_prior import GaussianBNNPrior
from base_networks.base_architectures import Sin, SharpTanh


def main():
    PATH = str(Path(__file__).resolve().parent)
    torch.set_default_dtype(torch.float64)

    Path(PATH + f"/james_figs").mkdir(parents=True, exist_ok=True)

    for codename, function_type in zip(['ursula', 'vincent', 'thomas'], ['sawtooth', 'bnn', 'heaviside']):
        bdnp = torch.load(PATH + f'/saved_models/bdnp-{codename}', weights_only=False, map_location=torch.device('cpu'))

        num_layers = len(bdnp.dims) - 1
        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*3, 3))
        for i in range(num_layers):
            Sigmas = bdnp.layers[i].prior.Sigmas.detach() # shape (d_out, d_in+1, d_in+1)
            eigvals = torch.linalg.eigvalsh(Sigmas).flatten()

            ax[i].hist(eigvals.numpy(), bins=50, color='steelblue', alpha=0.7)
            ax[i].set_title(f'Layer {i+1}')
            ax[i].set_xlabel('Eigenvalue')
            ax[i].set_ylabel('Count')

        fig.suptitle(function_type)
        plt.tight_layout()
        plt.savefig(PATH + f"/james_figs/{function_type}.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()