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

import models
from utils.training import train_meta_model
from utils.data_utils import scrambled_ctxt_trgt_to_grid
from base_networks.base_architectures import Sin, SharpTanh



def init_bdnp(architecture=[48, 48], nonlinearity='silu'):
    lik = models.GaussianLikelihood(y_dim=1, sigma_y=0.05, train=True)

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
    
def main(codename=None,
         architecture=[48, 48],
         nonlinearity='silu',
         training_steps=30_000,
         learning_rate=5e-3,
         final_learning_rate=5e-5,
         use_gpu=False,
         bnn_prior=False,
         use_pretrained=False,
        ):
    args_dict = locals()

    if codename is None:
        raise ValueError("User needs to specify a codename for this prior learning run.")

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    PATH = str(Path(__file__).resolve().parent)

    if use_pretrained:
        bdnp = torch.load(PATH + f'/saved_models/{codename}')
        md = torch.load(PATH + "/data/train_sets.pt", weights_only=False, map_location="cpu")

    else:

        Path(PATH + "/saved_models").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/training_configs").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/bnn").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/bnn/pdfs").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/bnn/pngs").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)

        with open(PATH + f"/training_configs/{codename}-config.json", 'w') as f:
            json.dump(args_dict, f, indent=4)

        md = torch.load(PATH + "/data/train_sets.pt", weights_only=False, map_location="cpu")

        bdnp = init_bdnp(
            architecture=architecture,
            nonlinearity=nonlinearity,
        )
        if not bnn_prior:
            bdnp.trainable_prior(True)

        ######## visualise prior predictive samples with standard guff prior ########
        for i in range(20):
            X_normed, _ = md[torch.randint(low=0, high=len(md), size=(1,))]
            X_normed_dev = X_normed.to(device)
            with torch.no_grad():
                pred_y_normed = bdnp(X_normed_dev, Xc=None, Yc=None, num_samples=1, batch_size=None)[0].squeeze(0)
            X_means, X_stds = torch.load(PATH + "/data/X_norm_consts.pt", weights_only=False, map_location="cpu")
            y_mean, y_std = torch.load(PATH + "/data/y_norm_consts.pt", weights_only=False, map_location="cpu")
            X = X_normed * X_stds + X_means
            pred_y = pred_y_normed * y_std + y_mean

            xs = X[:,:2]
            ys = pred_y.unsqueeze(0)
            xx1, xx2, Y = scrambled_ctxt_trgt_to_grid(xs, ys)
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
            ax.set_extent([5, 12, 45, 50])
            im = ax.pcolormesh(xx1, xx2, Y.squeeze(0), cmap="Blues", shading="auto")
            cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
            plt.savefig(PATH + f"/figs/bnn/pngs/{i}.png", bbox_inches="tight")
            plt.savefig(PATH + f"/figs/bnn/pdfs/{i}.pdf", bbox_inches="tight")
            plt.close()

        ############# okay now train ###########

        training_metrics = train_meta_model(
            bdnp,
            md,
            training_steps=training_steps,
            batch_size=5,
            learning_rate=learning_rate,
            final_learning_rate=final_learning_rate,
            num_samples=16,
            loss_function='pp-avi',
            ctxt_proportion_range=(0.01, 0.6),
            # within_task_batch_size=512,
            task_subsample_fraction=0.25,
            device_agnostic=True,
            dataset_on_cpu=True,
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

    ##################### visualise prior predictive samples with trained prior ##############
    for i in range(20):
        X_normed, _ = md[torch.randint(low=0, high=len(md), size=(1,))]
        X_normed_dev = X_normed.to(device)
        with torch.no_grad():
            pred_y_normed = bdnp(X_normed_dev, Xc=None, Yc=None, num_samples=1, batch_size=None)[0].squeeze(0)
        X_means, X_stds = torch.load(PATH + "/data/X_norm_consts.pt", weights_only=False, map_location="cpu")
        y_mean, y_std = torch.load(PATH + "/data/y_norm_consts.pt", weights_only=False, map_location="cpu")
        X = X_normed * X_stds + X_means
        pred_y = pred_y_normed * y_std + y_mean

        xs = X[:,:2]
        ys = pred_y.unsqueeze(0)
        xx1, xx2, Y = scrambled_ctxt_trgt_to_grid(xs, ys)
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
        ax.set_extent([5, 12, 45, 50])  
        im = ax.pcolormesh(xx1, xx2, Y.squeeze(0), cmap="Blues", shading="auto")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
        plt.savefig(PATH + f"/figs/{codename}/pngs/{i}.png", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pdfs/{i}.pdf", bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior transfer experiment on era5 dataset.")
    parser.add_argument('--codename', type=str, default=None, help='run codename')
    parser.add_argument('--architecture', type=int, nargs='+', default=[64, 64, 64], help='Hidden layer dims of BDNP and inference nets')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')
    parser.add_argument('--training_steps', type=int, default=30_000, help='The number of training steps')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one available. Default False.')
    parser.add_argument('--bnn_prior', action='store_true', help='Whether to fix prior to BNN standard one. Default False.')
    parser.add_argument('--use_pretrained', action='store_true', help='Use a pretrained BDNP with matching codename. Default False.')

    args = parser.parse_args()
    main(**vars(args))