import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils.data_utils import scrambled_ctxt_trgt_to_grid

def main():
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/figs/real_examples").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/figs/real_examples/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/figs/real_examples/pdfs").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        X_normed, y_normed = torch.load(PATH + "/data/train_sets.pt", weights_only=False)[torch.randint(0, 6000, (1,))]
        X_means, X_stds = torch.load(PATH + "/data/X_norm_consts.pt", weights_only=False)
        y_mean, y_std = torch.load(PATH + "/data/y_norm_consts.pt", weights_only=False)
        X = X_normed * X_stds + X_means
        y = y_normed * y_std + y_mean
        # cm = plt.cm.inferno

        # plt.scatter(X[:,0], X[:,1], color=cm(y))
        # plt.savefig(PATH + f"/figs/test.png", bbox_inches="tight")
        # plt.close()

        xs = X[:,:2]
        ys = y.unsqueeze(0)
        xx1, xx2, Y = scrambled_ctxt_trgt_to_grid(xs, ys)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # # Add geographic context
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
        ax.set_extent([5, 12, 45, 50]) 

        # # Plot rainfall
        im = ax.pcolormesh(xx1, xx2, Y.squeeze(0), cmap="Blues", shading="auto")

        # # Add colorbar
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")

        plt.savefig(PATH + f"/figs/real_examples/pngs/{i}.png", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/real_examples/pdfs/{i}.pdf", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()