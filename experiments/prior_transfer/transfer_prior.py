import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json

import models
from models import baselines
from utils.training import train_variational_model
from utils.data_utils import obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_gp_dataset_please, ctxt_trgt_split
from utils.mcmc_utils import autocorrelation_array
from base_networks.base_architectures import Sin, SharpTanh


def build_meta_dataset(num_datasets=10_000, n_range=[40, 100], function_type='sawtooth', x_range=[-4.0, 4.0]):
    md = []
    assert function_type.lower() in ['sawtooth', 'gp', 'heaviside']

    if function_type.lower() == 'sawtooth':
        dataset_func = obtain_me_a_nice_sawtooth_dataset_please
        data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
    elif function_type.lower() == 'gp':
        dataset_func = obtain_me_a_nice_gp_dataset_please
        data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': x_range}
    elif function_type.lower() == 'heaviside':
        dataset_func = obtain_me_a_nice_heaviside_dataset_please
        data_hypers = {'x_range': x_range, 'l': 1}

    for _ in range(num_datasets):
        X, y = dataset_func(n_range=n_range, **data_hypers)
        md.append((X, y))
    
    return md


def main(prior=None,
         model_name=None,
         hidden_dims=[48, 48],
         num_test_sets=16,
         function_type='sawtooth',
         use_gpu=False,
         use_shared_test_sets=False,
         ):
    
    args_dict = locals()

    print("model: ", model_name, " dataset: ", function_type, " prior type: ", prior)
    
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
    Path(PATH + f"/{function_type}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{function_type}/{model_name}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{function_type}/{model_name}/{prior}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{function_type}/{model_name}/{prior}/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{function_type}/{model_name}/{prior}/figs/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/{function_type}/{model_name}/{prior}/figs/pdfs").mkdir(parents=True, exist_ok=True)

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

    if function_type == 'sawtooth':
        nonlinearity = 'relu'
    elif function_type == 'heaviside':
        nonlinearity = 'tanh'
    elif function_type == 'gp':
        nonlinearity = 'silu'

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
    
    # if mfvi, givi, hmc, or lmc we train on each task and then evaluate, regardless of prior
    # if bdnp with fancy prior, do no training but just evaluate on tasks
    # if bdnp with bnn prior, treat it the same as other bnns

    if model_name.lower() == 'mfvi':
        lik = models.GaussianLikelihood(1, sigma_y=0.1)
    else:
        lik = models.GaussianLikelihood(1, sigma_y=0.05)

    if 'bdnp' in model_name.lower():
        model_kwargs = {'x_dim': 1,
                        'y_dim': 1,
                        'likelihood': lik,
                        'hidden_dims': hidden_dims,
                        'prior_type': 1,
                        'inf_dims': hidden_dims,
                        'use_final_layer_targets': True,
                        'use_final_layer_noise': False,
                        'scale_prior': True,
                        'nonlinearity': nl}
    else:
        model_kwargs = {'x_dim': 1,
                        'y_dim': 1,
                        'likelihood': lik,
                        'hidden_dims': hidden_dims,
                        'scale_prior': True,
                        'nonlinearity': nl}
        if model_name.lower() == 'givi':
            model_kwargs['num_inducing'] = 10
        elif model_name.lower() == 'swag':
            model_kwargs['K'] = 64
    
    if model_name == 'mfvi':
        model_class = baselines.MFVIBNN
    elif model_name == 'givi':
        model_class = baselines.GIVIBNN
    elif model_name == 'bdnp':
        model_class = models.BDNP
    elif model_name == 'hmc':
        model_class = baselines.HMC_BNN
    elif model_name == 'lmc':
        model_class = baselines.HMC_BNN # LMC is HMC with a single leapfrog step
    elif model_name == 'swag':
        model_class = baselines.SWAG_BNN

    if (model_name.lower() == 'bdnp') and (prior.lower() != 'bnn'):
        model = torch.load(PATH + f'/saved_models/{prior}', weights_only=False)


    ######## Evaluation time mothafucka ########

    torch.manual_seed(69)
    if use_gpu:
        torch.cuda.manual_seed(69)

    if use_shared_test_sets:
        test_sets = torch.load(PATH + f"/shared_test_sets/{function_type}.pt", weights_only=False)
        assert len(test_sets) >= num_test_sets
    else:
        test_sets = build_meta_dataset(num_datasets=num_test_sets, n_range=[50, 51], function_type=function_type)
    xs = torch.linspace(-4.0, 4.0, 200).unsqueeze(-1)
    results = {'ppd': torch.zeros((num_test_sets,)), 'mae': torch.zeros((num_test_sets,))}
    for j, test_set in enumerate(test_sets):
        if use_shared_test_sets:
            Xc_raw, yc_raw, Xt_raw, yt_raw = test_set
            Xc = Xc_raw.to(device=device, dtype=torch.float64)
            yc = yc_raw.to(device=device, dtype=torch.float64)
            Xt = Xt_raw.to(device=device, dtype=torch.float64)
            yt = yt_raw.to(device=device, dtype=torch.float64)
        else:
            Xc, yc, Xt, yt = ctxt_trgt_split(*test_set, ctxt_proportion_range=[0.05, 0.5])
        Path(PATH + f"/{function_type}/{model_name}/{prior}/figs/pngs/{j}").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/{function_type}/{model_name}/{prior}/figs/pdfs/{j}").mkdir(parents=True, exist_ok=True)

        # initialise model if not pretrained BDNP
        if not ((model_name.lower() == 'bdnp') and (prior.lower() != 'bnn')):
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

        if model_name.lower() in ['mfvi', 'givi'] or ((model_name.lower() == 'bdnp') and (prior.lower() == 'bnn')):
            retain_graph = False
            if model_name.lower() == 'givi':
                retain_graph = True
            training_metrics = train_variational_model(model,
                                                       (Xc, yc),
                                                       training_steps=15_000,
                                                       learning_rate=5e-3,
                                                       final_learning_rate=1e-4,
                                                       num_samples=8,
                                                       device_agnostic=True,
                                                       retain_graph=retain_graph)

            num_samples = 1000
            with torch.no_grad():
                if model_name.lower() == 'bdnp':
                    pred_samps = model(xs, Xc, yc, num_samples=100)[0]
                    pred_yt = model(Xt, Xc, yc, num_samples=num_samples)[0]
                else:
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
                step_size = 1e-4
                steps = 250_000
                burn = 50_000
                thin = 5_000
                leapfrog_steps = 1

            raw_samples, training_metrics = baselines.run_mcmc(model,
                                                               Xc,
                                                               yc,
                                                               algorithm='hmc', # LMC code is buggy, so we do HMC with 1 leapfrog step for LMC
                                                               steps=steps,
                                                               step_size=step_size,
                                                               minibatch_size=None, # full-batch
                                                               metropolis_adjusted=True,
                                                               leapfrog_steps=leapfrog_steps)
            
            burned_in_samples = raw_samples[burn:] # do burn-in and thinning here
            samples = burned_in_samples[::thin]

            training_metrics['autocorrelation'] = autocorrelation_array(raw_samples, max_lag=50)            
            
            with torch.no_grad():
                pred_yt = model.batch_forward(Xt, samples)
                pred_samps = model.batch_forward(xs, samples)
                num_samples = pred_yt.shape[0]


        elif model_name.lower() == 'swag':
            pretraining_metrics = baselines.pretrain(model, Xc, yc, training_steps=5_000, learning_rate=5e-3)
            training_metrics = baselines.run_SWAG(model, Xc, yc, learning_rate=1e-2, swa_steps=100, c=25)
            num_samples=1000
            with torch.no_grad():
                pred_samps = model.bma_forward(xs, num_samples=100)
                pred_yt = model.bma_forward(Xt, num_samples=num_samples)


        ## handle pre-trained bdnp ##
        if model_name.lower() == 'bdnp' and prior != 'bnn':
            num_samples = 1000
            with torch.no_grad():
                if model_name.lower() == 'bdnp':
                    pred_samps = model(xs, Xc, yc, num_samples=100)[0]
                    pred_yt = model(Xt, Xc, yc, num_samples=num_samples)[0]

        
        else:        ###### plot training metrics for all other models (i.e. not the pre-trained BDNP case) ######
            fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
            omitted_steps = 0
            for i, (key, value) in enumerate(training_metrics.items()):
                axes[i].plot(value[omitted_steps:])
                axes[i].set_xlabel(key)
                axes[i].grid()
                if key == 'elbo':
                    axes[i].set_ylim([-5000, 500])
                elif key == 'e_ll':
                    axes[i].set_ylim([-4000, 1000])
                elif key == 'kl':
                    axes[i].set_ylim([0, 2000])

            plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pdfs/{j}/training.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pngs/{j}/training.png", bbox_inches="tight")
            plt.close()

        if model_name.lower() == 'swag':
            fig, axes = plt.subplots(1, len(pretraining_metrics), figsize=(3*len(pretraining_metrics), 1))
            omitted_steps = 0
            for i, (key, value) in enumerate(pretraining_metrics.items()):
                axes[i].plot(value[omitted_steps:])
                axes[i].set_xlabel(key)
                axes[i].grid()

            plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pdfs/{j}/pretraining.pdf", bbox_inches="tight")
            plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pngs/{j}/pretraining.png", bbox_inches="tight")
            plt.close()



        ###### evaluation code common to all models.
        plt.plot(xs.unsqueeze(0).repeat((pred_samps.shape[0], 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(Xc.cpu(), yc.cpu(), color='C1', zorder=10000)
        plt.grid()
        lim = [-4.0, 4.0]
        if function_type.lower() == 'sawtooth':
            lim = [-2.0, 2.0]
        plt.xlim(lim)
        plt.ylim(lim)
        plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pdfs/{j}/predictive.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/{function_type}/{model_name}/{prior}/figs/pngs/{j}/predictive.png", bbox_inches="tight")
        plt.close() 

        # log per-datapoint average posterior predictive density
        results['ppd'][j] = (model.likelihood.log_prob(pred_yt, yt).sum(-1).logsumexp((0, 1)) - torch.tensor(num_samples * yt.shape[0]).log()).item()
        # per-datapoint average mean-absolute error between true targets and predictive mean.
        results['mae'][j] = ((pred_yt.mean(0) - yt).abs().mean()).item()


    # store results
    with open(PATH + f"/{function_type}/{model_name}/{prior}/results.json", 'w') as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prior transfer experiment")
    parser.add_argument('--prior', type=str, default=None, help='Type of prior.')
    parser.add_argument('--model_name', type=str, default=None, help='Type of BNN approximate inference algorithm.')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[48, 48], help='Hidden layer dims of BNNs.')
    parser.add_argument('--num_test_sets', type=int, default=16, help='Number of test datasets for evaluation.')
    parser.add_argument('--function_type', type=str, default='sawtooth', help='Type of function/dataset.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available.')
    parser.add_argument('--use_shared_test_sets', action='store_true', help='Use pre-made test datasets rather than constructing new ones on the fly.')

    args = parser.parse_args()
    main(**vars(args))