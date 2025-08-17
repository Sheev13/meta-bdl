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
from collections import defaultdict

def main(modelname_codename: json.loads):

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs/combined").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 1, figsize=(10.5, 3))
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    axes.grid(alpha=0.5)
    mc_y = None
    for i, (model, code) in enumerate(modelname_codename):
        with open(PATH + f"/results/{code}.json", "r") as f:
            results = json.load(f)
        x = [float(k) for k in results.keys()]
        y = list(results.values())
        if model.lower() == 'mc':
            mc_y = y
        c = f"C{i}"
        m = marker_styles[i]
        lab = model.upper()
        if model.lower() == 'mc':
            lab = 'LML'
        if model.lower() == 'meta_bdnp':
            lab = 'BDNP (meta)'
        axes.scatter(x, y, label=lab, color=c, marker=m, zorder=1000, s=10)
        axes.plot(x, y, color=c)
        axes.set_xlabel(r'$\sigma_y$')
        axes.set_ylabel('ELBO/LML')
        axes.set_xscale('log')
        axes.set_ylim([-125, 25])
        axes.set_xlim([0.01, 10.0])

    axes.spines['top'].set_alpha(0.5)
    axes.spines['right'].set_alpha(0.5)
    axes.spines['bottom'].set_alpha(0.5)
    axes.spines['left'].set_alpha(0.5)
    axes.legend(ncol=4)

    plt.savefig(PATH + f"/figs/combined/elbo-results.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/combined/elbo-results.png", bbox_inches="tight")
    plt.close()

    if mc_y is not None:
        print("Plotting KL curves too since ground-truth LML available.")
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 3))
        marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
        axes.grid(alpha=0.5)
        for i, (model, code) in enumerate(modelname_codename):
            if model == 'mc':
                continue
            with open(PATH + f"/results/{code}.json", "r") as f:
                results = json.load(f)
            x = [float(k) for k in results.keys()]
            y = [mc_y[i] - elbo for i, elbo in enumerate(list(results.values()))]
            c = f"C{i}"
            m = marker_styles[i]
            lab = model.upper()
            if model.lower() == 'mc':
                lab = 'LML'
            if model.lower() == 'meta_bdnp':
                lab = 'BDNP (meta)'
            axes.scatter(x, y, label=lab, color=c, marker=m, zorder=1000, s=10)
            axes.plot(x, y, color=c)
            axes.set_xlabel(r'$\sigma_y$')
            axes.set_ylabel(r'$KL[q(\mathbf{W}|\mathcal{D})\|p(\mathbf{W}|\mathcal{D})]$')
            axes.set_xscale('log')
            axes.set_ylim([0, 65])
            axes.set_xlim([0.01, 10.0])

        axes.spines['top'].set_alpha(0.5)
        axes.spines['right'].set_alpha(0.5)
        axes.spines['bottom'].set_alpha(0.5)
        axes.spines['left'].set_alpha(0.5)
        axes.legend(ncol=4)

        plt.savefig(PATH + f"/figs/combined/kl-results.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/combined/kl-results.png", bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP experiment 2")
    parser.add_argument('--modelname_codename', type=json.loads, required=True, help='Modelname-Codename pairs to be included in plot.')

    args = parser.parse_args()
    main(**vars(args))