import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

plt.rcParams.update(bundles.iclr2024(rel_width=0.75))
total_size = bundles.iclr2024(rel_width=0.75)["figure.figsize"]
cell_width = total_size[0]
cell_height = total_size[1] / 2

SEEDS = [21, 42, 69, 420]

def modelnames_to_labels(models):
    gandalf = []
    for m in models:
        if m.startswith("bdnp_"):
            p = m[5:]
            gandalf.append(p)
            # gandalf.append(f"BDNP ({p})")
        elif m == "bnp":
            gandalf.append("BNP  ")
        elif m == "ar-tnp":
            gandalf.append("  AR-TNP")
        else:
            gandalf.append(m.upper())
    return gandalf

def load_results(model_dir):
    """Load results (dict with keys 'ppd' and 'mae') for all seeds in a folder.
    If any seed file is missing, return None.
    """
    results = {metric: [] for metric in ["ppd", "mae"]}
    for seed in SEEDS:
        path = model_dir + f"/{seed}.json"
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        for metric in results:
            results[metric].append(data[metric])
    return results


def compute_stats(values):
    """Return mean and standard error of the mean."""
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(len(arr) - 1)
    return mean, sem


def main(exclude_bnns=False):
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/figs/results").mkdir(parents=True, exist_ok=True)

    for dataset in ['paul15', 'abalone']:

        Path(PATH + f"/figs/results/{dataset}").mkdir(parents=True, exist_ok=True)

        # collect all result folders
        all_folders = [f for f in Path(PATH + f"/results/{dataset}").iterdir() if f.is_dir()]

        # separate baselines and bdnp
        bdnp_folders = [f.name for f in all_folders if f.name.startswith("bdnp_")]
        bdnp_folders_sorted = sorted(bdnp_folders, key=lambda x: float(x.split("_")[1]))
        model_order = bdnp_folders_sorted + ["np", "bnp", "ar-tnp"]
        if not exclude_bnns:
            model_order = ["mfvi", "givi"] + model_order
            # model_order = ["givi"] + model_order

        metrics_data = {"ppd": [], "mae": []}
        colours = []
        model_order_clean = []
        bdnp_idx = []

        for i, model in enumerate(model_order):
            if model.startswith("bdnp"):
                bdnp_idx.append(i)
            model_dir = PATH + f"/results/{dataset}/{model}"
            results = load_results(model_dir)
            if results is None:
                continue

            for metric in metrics_data:
                mean, sem = compute_stats(results[metric])
                metrics_data[metric].append((mean, sem))

            if model in ["mfvi", "givi"]:
                cval = 0.0
            elif model in ["np", "bnp", "ar-tnp"]:
                cval = 1.0
            else:
                cval = float(model.split("_")[1])
            colours.append(plt.cm.plasma(cval*0.8))
            model_order_clean.append(model)

        for metric in ["mae", "ppd"]:
            means = [m[0] for m in metrics_data[metric]]
            sems = [m[1] for m in metrics_data[metric]]

            # fig, ax = plt.subplots(figsize=(7.5, 1.875))
            fig, ax = plt.subplots(figsize=(cell_width, cell_height))
            x = np.arange(len(model_order_clean))
            for xi, mean, sem, color in zip(x, means, sems, colours):
                ax.errorbar(
                    xi, mean, yerr=sem, fmt="o", markersize=4.5,
                    capsize=3.5, elinewidth=1.2, color=color, zorder=2
                )
            # ax.scatter(x, means, c=colours, s=80, zorder=2)
            bdnp_ms = [means[ind] for ind in bdnp_idx]
            bdnp_xs = [x[ind] for ind in bdnp_idx]
            grey = 'gray' # American bollocks
            ax.plot(bdnp_xs, bdnp_ms, alpha=0.75, ls=':', lw=1.0, color=grey)

            # store xticklabel size for manually setting BDNP label to this size
            ax.set_xticks(x)
            labels = modelnames_to_labels(model_order_clean)
            ax.set_xticklabels(labels)
            ticklabel = ax.get_xticklabels()[0]
            label_size = ticklabel.get_size()

            # ensure BNP and AR-TNP are spaced apart, also MFVI and GIVI
            xticks = ax.get_xticklabels()
            mfvi_label = xticks[0]
            mfvi_trans = offset_copy(mfvi_label.get_transform(), x=-0.03, y=0, fig=fig)
            mfvi_label.set_transform(mfvi_trans)
            bnp_label, artnp_label = xticks[-2:]
            bnp_trans = offset_copy(bnp_label.get_transform(), x=-0.03, y=0, fig=fig)
            bnp_label.set_transform(bnp_trans)
            artnp_trans = offset_copy(artnp_label.get_transform(), x=0.03, y=0, fig=fig)
            artnp_label.set_transform(artnp_trans)

            mid = (bdnp_idx[0]-0.5 + bdnp_idx[-1]+0.5) / 2
            h = -0.25
            ax.annotate(
                "BDNP",
                xy=(mid, h - 0.075),
                xycoords=('data', 'axes fraction'),
                ha='center', va='top', fontsize=label_size,
                annotation_clip=False
            )
            # Draw bracket line
            ax.plot(
                [bdnp_idx[0]-0.3, bdnp_idx[0]-0.3, bdnp_idx[-1]+0.3, bdnp_idx[-1]+0.3],
                [h, h - 0.03, h - 0.03, h],
                transform=ax.get_xaxis_transform(),
                color='black', clip_on=False
            )

            if metric == "mae":
                # ax.set_ylabel("mae (↓)", rotation=0, labelpad=20)
                ax.set_ylim(bottom=0.0)
            else:
                # ax.set_ylabel("ppd (↑)", rotation=0, labelpad=20)
                pass
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)

            plt.savefig(PATH + f"/figs/results/{dataset}/{metric}.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/figs/results/{dataset}/{metric}.png", bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()