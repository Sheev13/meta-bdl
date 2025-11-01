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
from models import baselines
from utils.training import train_variational_model
from utils.data_utils import scrambled_ctxt_trgt_to_grid, scrambled_sprs_to_masked_grid
from utils.mcmc_utils import autocorrelation_array
from base_networks.base_architectures import Sin, SharpTanh


def main(prior=None,
         model_name=None,
         hidden_dims=[64, 64, 64],
         swissless=False,
         use_gpu=False,
         ):
    
    args_dict = locals()

    print("model: ", model_name, " prior type: ", prior)
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + "/training_configs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{model_name}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{model_name}/{prior}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{model_name}/{prior}/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{model_name}/{prior}/figs/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{model_name}/{prior}/figs/pdfs").mkdir(parents=True, exist_ok=True)

    with open(PATH + f"/training_configs/{model_name}-{prior}-config.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    if prior is None:
        raise ValueError("User failed to specify the prior to use. Total fool.")
    saved_models_path = Path(__file__).parent / "saved_models"
    if prior != 'bnn':
        file = Path(saved_models_path / prior)
        if not file.is_file():
            raise ValueError("User failed to provide prior name corresponding to valid pretrained BDNP.")

    assert model_name.lower() in ['mfvi', 'givi', 'hmc', 'lmc', 'bdnp', 'swag']
    
    # if mfvi, givi, hmc, or lmc we train on each task and then evaluate, regardless of prior
    # if bdnp with fancy prior, do no training but just evaluate on tasks
    # if bdnp with bnn prior, treat it the same as other bnns
    
    nl = torch.nn.ReLU()

    model_kwargs = {'x_dim': 3,
                    'y_dim': 1,
                    'hidden_dims': hidden_dims,
                    'scale_prior': True,
                    'nonlinearity': nl}
    if model_name.lower() == 'givi':
        model_kwargs['num_inducing'] = 128
    elif model_name.lower() == 'swag':
        model_kwargs['K'] = 64
    
    if model_name == 'mfvi':
        model_class = baselines.MFVIBNN
    elif model_name == 'givi':
        model_class = baselines.GIVIBNN
    elif model_name == 'hmc':
        model_class = baselines.HMC_BNN
    elif model_name == 'lmc':
        model_class = baselines.HMC_BNN # LMC is HMC with a single leapfrog step
    elif model_name == 'swag':
        model_class = baselines.SWAG_BNN

    if model_name.lower() == 'bdnp':
        model = torch.load(PATH + f'/saved_models/{prior}', weights_only=False)


    ######## Evaluation time mothafucka ########

    torch.manual_seed(69)
    if use_gpu:
        torch.cuda.manual_seed(69)

    if swissless:
        test_sets = torch.load(PATH + f"/data/swissless_test_sets.pt", weights_only=False)
    else:
        test_sets = torch.load(PATH + f"/data/test_sets.pt", weights_only=False)
    num_test_sets = len(test_sets)

    results = {'ppd': torch.zeros((num_test_sets,)), 'mae': torch.zeros((num_test_sets,))}
    for j, test_set in enumerate(test_sets):
        Xc_raw, yc_raw, Xt_raw, yt_raw = test_set
        Xc = Xc_raw.to(device=device, dtype=torch.float64)
        yc = yc_raw.to(device=device, dtype=torch.float64)
        Xt = Xt_raw.to(device=device, dtype=torch.float64)
        yt = yt_raw.to(device=device, dtype=torch.float64)
        xs = torch.cat((Xc, Xt), dim=0)
        if swissless:
            Path(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless").mkdir(parents=True, exist_ok=True)
            Path(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless").mkdir(parents=True, exist_ok=True)
        else:
            Path(PATH + f"/{model_name}/{prior}/figs/pngs/{j}").mkdir(parents=True, exist_ok=True)
            Path(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}").mkdir(parents=True, exist_ok=True)

        # initialise model if not pretrained BDNP
        if model_name.lower() != 'bdnp':
            lik = models.GaussianLikelihood(1, sigma_y=0.05)
            model_kwargs['likelihood'] = lik
            model = model_class(**model_kwargs)
            # adopt prior if using pre-trained one.
            if prior.lower() != 'bnn':
                pretrained_bdnp = torch.load(PATH + f'/saved_models/{prior}', weights_only=False)
                layerwise_priors = []
                for layer in pretrained_bdnp.layers:
                    m = layer.prior.mus.detach()
                    if layer.prior_type == 1:
                        S = layer.prior.Sigmas.detach()
                    elif layer.prior_type == 0:
                        S = layer.prior.sigmas.detach().diag_embed()
                    else:
                        raise ValueError(f"pre-trained BDNP has unsupported type of prior for adoption into {model_name} BNN.")
                    layerwise_priors.append((m, S))
                model.adopt_prior(layerwise_priors)

        elif model_name.lower() == 'bdnp':
            model = torch.load(PATH + f'/saved_models/{prior}', weights_only=False)

        if model_name.lower() in ['mfvi', 'givi']:
            retain_graph = False
            if model_name.lower() == 'givi':
                retain_graph = True
                model.init_inducing_points(Xc)
            training_metrics = train_variational_model(model,
                                                       (Xc, yc),
                                                       training_steps=75_000,
                                                       learning_rate=5e-3,
                                                       final_learning_rate=1e-4,
                                                       num_samples=8,
                                                       device_agnostic=True,
                                                       retain_graph=retain_graph)

            num_samples = 1000
            with torch.no_grad():
                pred_samps = model(xs, num_samples=100)
                pred_yt = model(Xt, num_samples=num_samples)


        elif model_name.lower() in ['lmc', 'hmc']:
            if model_name.lower() == 'hmc':
                step_size = 1e-4
                steps = 5_000
                burn = 2_000
                thin = 50
                leapfrog_steps = 100
            # else:
            #     step_size = 1e-5
            #     steps = 500_000
            #     burn = 25_000
            #     thin = 10_000
            # raw_samples, training_metrics = baselines.run_mcmc(model,
            #                             Xc,
            #                             yc,
            #                             algorithm=model_name.lower(),
            #                             steps=steps,
            #                             step_size=step_size,
            #                             minibatch_size=None, # full-batch
            #                             metropolis_adjusted=True,
            #                             leapfrog_steps=100) # leapfrog_steps is silently ignored for LMC
            else:
                step_size = 5e-4
                steps = 200_000
                burn = 50_000
                thin = 2_500
                leapfrog_steps = 1

            raw_samples, training_metrics = baselines.run_mcmc(model,
                                                               Xc,
                                                               yc,
                                                               algorithm='hmc', # LMC code is buggy, so we do HMC with 1 leapfrog step for LMC
                                                               steps=steps,
                                                               step_size=step_size,
                                                               minibatch_size=128, # SG-(HMC/LD)
                                                               metropolis_adjusted=False,
                                                               leapfrog_steps=leapfrog_steps)
            
            burned_in_samples = raw_samples[burn:] # do burn-in and thinning here
            samples = burned_in_samples[::thin]

            # below is nice but always makes cuda run out of memory :/
            # training_metrics['autocorrelation'] = autocorrelation_array(raw_samples, max_lag=50)       
            
            with torch.no_grad():
                pred_yt = model.batch_forward(Xt, samples)
                pred_samps = model.batch_forward(xs, samples)
                num_samples = pred_yt.shape[0]


        elif model_name.lower() == 'swag':
            pretraining_metrics = baselines.pretrain(model, Xc, yc, training_steps=10_000, learning_rate=1e-3)
            training_metrics = baselines.run_SWAG(model, Xc, yc, learning_rate=5e-3, swa_steps=200, c=25)
            num_samples=1000
            with torch.no_grad():
                pred_samps = model.bma_forward(xs, num_samples=100)
                pred_yt = model.bma_forward(Xt, num_samples=num_samples)
    

        ## handle pre-trained bdnp ##
        elif model_name.lower() == 'bdnp':
            # fine-tune
            model.likelihood.raw_sigmas.data = torch.log(torch.tensor(0.05)) # set observation noise to 0.05
            model.likelihood.raw_sigmas.requires_grad = False
            training_metrics = train_variational_model(model,
                                            (Xc, yc),
                                            training_steps=25_000,
                                            learning_rate=5e-4,
                                            final_learning_rate=1e-4,
                                            num_samples=8,
                                            device_agnostic=True)

            num_samples = 1000
            with torch.no_grad():
                pred_samps = model(xs, Xc, yc, num_samples=100, batch_size=50)[0]
                pred_yt = model(Xt, Xc, yc, num_samples=num_samples, batch_size=50)[0]


        fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
        omitted_steps = 100
        for i, (key, value) in enumerate(training_metrics.items()):
            axes[i].plot(value[omitted_steps:])
            axes[i].set_xlabel(key)
            axes[i].grid()

        if swissless:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/training.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/training.png", bbox_inches="tight")
        else:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/training.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/training.png", bbox_inches="tight")
        plt.close()

        if model_name.lower() == 'swag':
            fig, axes = plt.subplots(1, len(pretraining_metrics), figsize=(3*len(pretraining_metrics), 1))
            omitted_steps = 0
            for i, (key, value) in enumerate(pretraining_metrics.items()):
                axes[i].plot(value[omitted_steps:])
                axes[i].set_xlabel(key)
                axes[i].grid()

            if swissless:
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/pretraining.pdf", bbox_inches="tight")
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/pretraining.png", bbox_inches="tight")
            else:
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/pretraining.pdf", bbox_inches="tight")
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/pretraining.png", bbox_inches="tight")
            plt.close()



        ###### evaluation code common to all models. #######

        # log per-datapoint average posterior predictive density
        results['ppd'][j] = (model.likelihood.log_prob(pred_yt, yt).sum(-1).logsumexp((0, 1)) - torch.tensor(num_samples * yt.shape[0]).log()).item()

        # normalise data for MAE and plotting:
        X_means, X_stds = torch.load(PATH + "/data/X_norm_consts.pt", weights_only=False)
        y_mean, y_std = torch.load(PATH + "/data/y_norm_consts.pt", weights_only=False)
        X_means = X_means.to(device=device, dtype=torch.float64)
        X_stds = X_stds.to(device=device, dtype=torch.float64)
        y_mean = y_mean.to(device=device, dtype=torch.float64)
        y_std = y_std.to(device=device, dtype=torch.float64)
        pred_yt = pred_yt * y_std + y_mean
        pred_samps = pred_samps * y_std + y_mean
        yt = yt * y_std + y_mean
        yc = yc * y_std + y_mean
        Xc = Xc * X_stds + X_means
        xs = xs * X_stds + X_means

        # per-datapoint average mean-absolute error between true targets and predictive mean.
        results['mae'][j] = ((pred_yt.mean(0) - yt).abs().mean()).item()


        # plot posterior predictive samples/means/stds etc
        xs = xs[:,:2]
        ys = pred_samps
        xx1, xx2, Y = scrambled_ctxt_trgt_to_grid(xs, ys)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)
        ax.set_extent([5, 12, 45, 50])  
        im = ax.pcolormesh(xx1, xx2, Y.mean(0), cmap="Blues", shading="auto")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
        if swissless:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/pred_mean.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/pred_mean.png", bbox_inches="tight")
        else:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/pred_mean.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/pred_mean.png", bbox_inches="tight")
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)
        ax.set_extent([5, 12, 45, 50])  
        im = ax.pcolormesh(xx1, xx2, Y.std(0), cmap="inferno", shading="auto")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precip. std (mm)")
        if swissless:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/pred_std.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/pred_std.png", bbox_inches="tight")
        else:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/pred_std.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/pred_std.png", bbox_inches="tight")
        plt.close()

        for k in range(10):
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)
            ax.set_extent([5, 12, 45, 50])  
            im = ax.pcolormesh(xx1, xx2, Y[k], cmap="Blues", shading="auto")
            cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
            if swissless:
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/sample-{k}.pdf", bbox_inches="tight")
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/pred-{k}.png", bbox_inches="tight")
            else:
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/sample-{k}.pdf", bbox_inches="tight")
                plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/pred-{k}.png", bbox_inches="tight")
            plt.close()

        
        # plot task, i.e. full version and masked version
        true_ys = torch.cat((yc, yt), dim=0)
        xx1, xx2, true_Y = scrambled_ctxt_trgt_to_grid(xs, true_ys.unsqueeze(0))

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)
        ax.set_extent([5, 12, 45, 50])  
        im = ax.pcolormesh(xx1, xx2, true_Y.squeeze(0), cmap="Blues", shading="auto")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
        if swissless:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/full.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/full.png", bbox_inches="tight")
        else:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/full.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/full.png", bbox_inches="tight")
        plt.close()


        masked_Y = scrambled_sprs_to_masked_grid(Xc[:,:2], yc, xx1, xx2)
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0)
        ax.set_extent([5, 12, 45, 50])
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="maroon")
        im = ax.pcolormesh(xx1, xx2, masked_Y, cmap=cmap, shading="auto")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")
        if swissless:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}_swissless/ctxt.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}_swissless/ctxt.png", bbox_inches="tight")
        else:
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pdfs/{j}/ctxt.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{model_name}/{prior}/figs/pngs/{j}/ctxt.png", bbox_inches="tight")
        plt.close()

    # store results
    if swissless:
        results_path = PATH + f"/{model_name}/{prior}/results_swissless.json"
    else:
        results_path = PATH + f"/{model_name}/{prior}/results.json"
    with open(results_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior transfer era5.")
    parser.add_argument('--prior', type=str, default=None, help='Type of prior.')
    parser.add_argument('--model_name', type=str, default=None, help='Type of BNN approximate inference algorithm.')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64], help='Hidden layer dims of BNNs.')
    parser.add_argument('--swissless', action='store_true', help='Whether to omit Swiss data from test set context sets.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available.')

    args = parser.parse_args()
    main(**vars(args))