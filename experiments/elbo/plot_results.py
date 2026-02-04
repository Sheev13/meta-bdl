import torch
import numpy as np
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
from collections import defaultdict

plt.rcParams.update(bundles.iclr2024(rel_width=0.8))
total_size = bundles.iclr2024(rel_width=0.8)["figure.figsize"]
cell_width = total_size[0]
cell_height = total_size[1] / 2

def get_colours():
    colours = {'mc': 'black'}

    # vi_cmap = plt.cm.cool
    # vi_colours = [vi_cmap(i) for i in np.linspace(0.0, 0.6, 4)]
    # vi_methods = ['mfvi', 'ucvi', 'lcvi', 'fcvi']
    # for i in range(4):
    #     colours[vi_methods[i]] = vi_colours[i]

    # bnnp_cmap = plt.cm.copper
    # bnnp_colours = [bnnp_cmap(i) for i in [0.5, 0.7]]
    # bnnp_methods = ['bnnp', 'meta_bnnp']
    # for i in range(2):
    #     colours[bnnp_methods[i]] = bnnp_colours[i]

    cmap = plt.cm.RdYlGn
    methods = ['mfvi', 'ucvi', 'lcvi', 'fcvi', 'spare', 'givi', 'meta_bnnp', 'bnnp']
    cmap_colours = [cmap(i) for i in np.linspace(0.0, 1.0, len(methods))]
    for i in range(len(methods)):
        if methods[i] != 'spare':
            colours[methods[i]] = cmap_colours[i]
    
    return colours


def main():
    x_lim = 0.03
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/figs/results").mkdir(parents=True, exist_ok=True)
    seeds = [21, 42, 69, 420]
    modelname_codename = {"bnnp": "bdnp_final",
                          "fcvi": "fcvi_final",
                          "givi": "givi_final",
                          "lcvi": "lcvi_final",
                          "mc": "mc_final",
                          "meta_bnnp": "meta_bdnp_final",
                          "mfvi": "mfvi_final",
                          "ucvi": "ucvi_final"
                          }

    # fig, axes = plt.subplots(1, 1, figsize=(7, 2))
    fig, axes = plt.subplots(1, 1, figsize=(cell_width, cell_height))
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
        if model.lower() == 'meta_bnnp':
            lab = 'BNNP (meta)'
        axes.scatter(x, y_means.tolist(), label=lab, color=c, marker=m, zorder=1000, s=20)
        axes.plot(x, y_means.tolist(), color=c)
        axes.fill_between(x, (y_means-2*y_stds).tolist(), (y_means+2*y_stds).tolist(), color=c, alpha=0.2)
        axes.set_xlabel(r'$\sigma_y$')
        # axes.set_ylabel('ELBO/LML')
        axes.set_xscale('log')
        axes.set_ylim([-80, 20])
        axes.set_xlim([max(0.01, x_lim), 10.0])

    axes.spines['top'].set_alpha(0.5)
    axes.spines['right'].set_alpha(0.5)
    axes.spines['bottom'].set_alpha(0.5)
    axes.spines['left'].set_alpha(0.5)
    # axes.legend(ncol=2)

    plt.savefig(PATH + f"/figs/results/elbo.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/results/elbo.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(cell_width, cell_height))
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    axes.grid(alpha=0.5)
    for i, (model, code) in enumerate(modelname_codename.items()):
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
        if model == 'mc':
            y_stds = torch.zeros_like(y_means)
        c = colours[model]
        m = marker_styles[i]
        lab = model.upper()
        if model.lower() == 'mc':
            lab = 'LML'
        if model.lower() == 'meta_bnnp':
            lab = 'BNNP (meta)'
        axes.scatter(x, y_means.tolist(), label=lab, color=c, marker=m, zorder=1000, s=20)
        axes.plot(x, y_means.tolist(), color=c)
        axes.fill_between(x, (y_means-2*y_stds).tolist(), (y_means+2*y_stds).tolist(), color=c, alpha=0.2)
        axes.set_xlabel(r'$\sigma_y$')
        # axes.set_ylabel(r'$KL[q(\mathbf{W}|\mathcal{D})\|p(\mathbf{W}|\mathcal{D})]$')
        axes.set_xscale('log')
        axes.set_ylim([0, 50])
        axes.set_xlim([max(0.01, x_lim), 10.0])

    axes.spines['top'].set_alpha(0.5)
    axes.spines['right'].set_alpha(0.5)
    axes.spines['bottom'].set_alpha(0.5)
    axes.spines['left'].set_alpha(0.5)
    axes.legend(ncol=2)

    plt.savefig(PATH + f"/figs/results/kl.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/results/kl.png", bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()