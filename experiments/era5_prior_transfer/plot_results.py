import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
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

plt.rcParams.update(bundles.iclr2024(rel_width=0.4))
total_size = bundles.iclr2024(rel_width=0.4)["figure.figsize"]
cell_width = total_size[0]
cell_height = total_size[1]# / 2


def main(codename='firefly', swissless=True):
    methods = ['swag', 'mfvi', 'lmc', 'givi', 'bdnp', 'hmc']
    PATH = str(Path(__file__).resolve().parent)
    swsls = "_swissless" if swissless else ""

    positions = np.arange(len(methods)) * 2.0  # space out models
    width = 0.8  # width of each box
    marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']

    for metric in  ['ppd', 'mae']:

        fig, ax = plt.subplots(figsize=(cell_width, cell_height))
        for j, method in enumerate(methods):
            try:
                with open(PATH + f"/{method}/bnn/results{swsls}.json", "r") as f:
                    bnn_prior_results = json.load(f)[metric]
                m_bnn = np.mean(bnn_prior_results)
                se_bnn = np.std(bnn_prior_results, ddof=1) / np.sqrt(len(bnn_prior_results))
                ax.errorbar(positions[j]-width/2, m_bnn, yerr=se_bnn, fmt='o',
                            color='tab:brown', capsize=2.5, markersize=3)
                # ax.boxplot(bnn_prior_results, positions=[positions[j] - width/2], widths=width, patch_artist=True,
                #         boxprops=dict(facecolor=colours[method]), medianprops=dict(color='red'))
            except:
                FileNotFoundError
            try:
                with open(PATH + f"/{method}/{codename}/results{swsls}.json", "r") as f:
                    learned_prior_results = json.load(f)[metric]
                m_learn = np.mean(learned_prior_results)
                se_learn = np.std(learned_prior_results, ddof=1) / np.sqrt(len(learned_prior_results))
                ax.errorbar(positions[j]+width/2, m_learn, yerr=se_learn, fmt='D',
                            color='tab:blue', capsize=2.5, markersize=3) 
                # ax.boxplot(learned_prior_results, positions=[positions[j] + width/2], widths=width, patch_artist=True,
                #         boxprops=dict(facecolor=colours[method]), medianprops=dict(color='green'))
            except:
                FileNotFoundError   

        if metric == 'mae':
            ax.set_ylim(bottom=0.0)
        else:
            # ax.set_title('Era5')
            pass
        ax.set_xticks(positions)
        labels = [method.upper() for method in methods]
        for i in range(len(labels)):
            if labels[i] == 'HMC':
                labels[i] = 'SGHMC'
            if labels[i] == 'LMC':
                labels[i] = 'SGLD'
            if labels[i] == 'BDNP':
                labels[i] = 'BNNP'
        ax.set_xticklabels(labels)
        xticks = ax.get_xticklabels()
        text = xticks[-1]
        trans = offset_copy(text.get_transform(), x=0.05, y=0, fig=fig)
        text.set_transform(trans)

        for j, pos in enumerate(positions):
            ax.axvspan(pos - 1.0, pos + 1.0, color="gray", alpha=0.1 if j % 2 == 0 else 0)
        ylab = metric.upper() + " (↑)" if metric == 'ppd' else metric.upper() + " (↓)"
        # ax.set_ylabel(ylab, rotation=0, labelpad=20)
        # ax.set_title('Era5')
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        legend_elements = [
            Line2D([0], [0], marker='o', color='tab:brown', label='Standard prior',
                markersize=3, linestyle='None'),  # brown dots
            Line2D([0], [0], marker='D', color='tab:blue', label='Learned prior',
                markersize=3, linestyle='None')   # blue diamonds
        ]
        if metric == 'mae':
            ax.legend(handles=legend_elements)

        Path(PATH + "/figs/results").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/results/bnn-vs-{codename}").mkdir(parents=True, exist_ok=True)
        plt.savefig(PATH + f"/figs/results/bnn-vs-{codename}/{metric}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/results/bnn-vs-{codename}/{metric}.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()