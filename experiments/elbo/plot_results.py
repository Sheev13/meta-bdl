import torch
import numpy as np
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

def get_colours():
    colours = {'givi': 'grey', 'mc': 'black'}

    vi_cmap = plt.cm.cool
    vi_colours = [vi_cmap(i) for i in np.linspace(0.0, 0.6, 4)]
    vi_methods = ['mfvi', 'ucvi', 'lcvi', 'fcvi']
    for i in range(4):
        colours[vi_methods[i]] = vi_colours[i]

    bdnp_cmap = plt.cm.copper
    bdnp_colours = [bdnp_cmap(i) for i in [0.5, 0.7]]
    bdnp_methods = ['bdnp', 'meta_bdnp']
    for i in range(2):
        colours[bdnp_methods[i]] = bdnp_colours[i]
    
    return colours


def main():
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/figs/results").mkdir(parents=True, exist_ok=True)
    seeds = [21, 42, 69, 420]
    modelname_codename = {"bdnp": "bdnp_final",
                          "fcvi": "fcvi_final",
                          "givi": "givi_final",
                          "lcvi": "lcvi_final",
                          "mc": "mc_final",
                          "meta_bdnp": "meta_bdnp_final",
                          "mfvi": "mfvi_final",
                          "ucvi": "ucvi_final"
                          }

    fig, axes = plt.subplots(1, 1, figsize=(10.5, 3))
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    colours = get_colours()
    axes.grid(alpha=0.5)
    mc_y = None
    for i, (model, code) in enumerate(modelname_codename.items()):
        ys = []
        for seed in seeds:
            with open(PATH + f"/results/{seed}/{code}.json", "r") as f:
                results = json.load(f)
            x = [float(k) for k in results.keys()]
            y = list(results.values())
            ys.append(y)
        ys = torch.tensor(ys)
        y_means = ys.mean(0)
        y_stds = ys.std(0)
        if model.lower() == 'mc':
            mc_y = y_means
        c = colours[model]
        m = marker_styles[i]
        lab = model.upper()
        if model.lower() == 'mc':
            lab = 'LML'
        if model.lower() == 'meta_bdnp':
            lab = 'BDNP (meta)'
        axes.scatter(x, y_means.tolist(), label=lab, color=c, marker=m, zorder=1000, s=10)
        axes.plot(x, y_means.tolist(), color=c)
        axes.fill_between(x, (y_means-2*y_stds).tolist(), (y_means+2*y_stds).tolist(), color=c, alpha=0.2)
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

    plt.savefig(PATH + f"/figs/results/elbo.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/results/elbo.png", bbox_inches="tight")
    plt.close()

    if mc_y is not None:
        print("Plotting KL curves too since ground-truth LML available.")
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 3))
        marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
        axes.grid(alpha=0.5)
        for i, (model, code) in enumerate(modelname_codename.items()):
            if model == 'mc':
                continue
            ys = []
            for seed in seeds:
                with open(PATH + f"/results/{seed}/{code}.json", "r") as f:
                    results = json.load(f)
                x = [float(k) for k in results.keys()]
                y = [mc_y[i] - elbo for i, elbo in enumerate(list(results.values()))]
                ys.append(y)
            ys = torch.tensor(ys)
            y_means = ys.mean(0)
            y_stds = ys.std(0)
            c = colours[model]
            m = marker_styles[i]
            lab = model.upper()
            if model.lower() == 'mc':
                lab = 'LML'
            if model.lower() == 'meta_bdnp':
                lab = 'BDNP (meta)'
            axes.scatter(x, y_means.tolist(), label=lab, color=c, marker=m, zorder=1000, s=10)
            axes.plot(x, y_means.tolist(), color=c)
            axes.fill_between(x, (y_means-2*y_stds).tolist(), (y_means+2*y_stds).tolist(), color=c, alpha=0.2)
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

        plt.savefig(PATH + f"/figs/results/kl.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/results/kl.png", bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    main()