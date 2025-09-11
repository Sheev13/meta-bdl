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
from utils.training import train_meta_model, train_variational_model


def main(model: str = None,
         dataset: str = None,
         prior_trainability: float = None,
         training_steps: int = 30_000,
         learning_rate: float = 5e-3,
         final_learning_rate: float = 1e-4,
         seed: int = None,
         use_gpu: bool = False):
    args_dict = locals()

    print("model: ", model, " seed: ", seed, "prior trainability: ", prior_trainability)

    model = model.lower()
    assert model in ['mfvi', 'givi', 'bdnp', 'np', 'bnp', 'abnp', 'anp'] # no conv(c)np since there are 7 > 3 input dims.
    if model in ['swag', 'mfvi', 'givi']:
        model_type = 'bnn'
    elif model == 'bdnp':
        model_type = 'bdnp'
    else:
        model_type = 'np'

    if model is None:
        raise ValueError("User failed to specify which model to use.")
    if dataset is None:
        raise ValueError("User failed to specify which dataset to use.")
    if seed is None:
        raise ValueError("User failed to specify which seed to use.")
    if (model == 'bdnp') and (prior_trainability is None):
        raise ValueError("User failed to specify how much of the BDNP prior to train.")
    if prior_trainability is not None:
        if prior_trainability < 0.0:
            raise ValueError("User has set fraction of learnable prior parameters to less than 0.0.")
        if prior_trainability > 1.0:
            raise ValueError("User has set fraction of learnable prior parameters to greater than 1.0.")

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    model_codename = model
    if model == 'bdnp':
        model_codename += f"_{prior_trainability}"

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/training/{model_codename}/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/results").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/results/{dataset}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/results/{dataset}/{model_codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/training_configs").mkdir(parents=True, exist_ok=True)

    with open(PATH + f"/training_configs/{model_codename}-config-{seed}.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    ##### done to here #####
    # define model classes, model kwargs, training funcs, training kwargs, seed

    nl = torch.nn.SiLU()
    architecture = [64, 64, 64]

    with open(PATH +f"/data/{dataset}/metadata.json") as f:
        dataset_metadata = json.load(f)
    x_dim = int(dataset_metadata["dimensionality"])

    if model == 'bdnp':
        model_kwargs = {'x_dim': x_dim,
                        'y_dim': 1,
                        'likelihood': models.GaussianLikelihood(1, sigma_y=0.1, train=True),
                        'hidden_dims': architecture,
                        'prior_type': 1,
                        'inf_dims': architecture,
                        'use_final_layer_targets': True,
                        'use_final_layer_noise': False,
                        'scale_prior': True,
                        'nonlinearity': nl}
        m = models.BDNP(**model_kwargs)
        if prior_trainability != 0.0:
            if prior_trainability == 1.0:
                m.trainable_prior(True)
            else:
                m.set_prior_trainability(prior_trainability, from_front=False)

        training_kwargs = {'training_steps': training_steps,
                           'batch_size': 5,
                           'learning_rate': learning_rate,
                           'final_learning_rate': final_learning_rate,
                           'num_samples': 16,
                           'loss_function': 'pp-avi',
                           'ctxt_proportion_range': (0.1, 0.5),
                           'task_subsample_fraction': None,
                           'within_task_batch_size': 250,
                           'device_agnostic': True}

    elif model_type == 'bnn':
        model_kwargs = {'x_dim': x_dim,
                        'y_dim': 1,
                        'likelihood': models.GaussianLikelihood(1, sigma_y=0.1, train=True if model=='givi' else False),
                        'hidden_dims': architecture,
                        'scale_prior': True,
                        'nonlinearity': nl}
        if model == 'givi':
            model_kwargs['num_inducing'] = 128
        
        if model == 'mfvi':
            m = baselines.MFVIBNN(**model_kwargs)
        elif model == 'givi':
            m = baselines.GIVIBNN(**model_kwargs)

        training_kwargs = {'training_steps': training_steps,
                            'learning_rate': learning_rate,
                            'final_learning_rate': final_learning_rate,
                            'num_samples': 8,
                            'device_agnostic': True,
                            'retain_graph': model=='givi'}
            
    elif model_type == 'np':
        model_kwargs = {'x_dim': x_dim,
                        'y_dim': 1,
                        'lik': models.GaussianLikelihood(1, sigma_y=0.1, train=True), # should we really train this?
                        'encoder_dims': architecture,
                        'decoder_dims': architecture,
                        'nonlinearity': nl}
        
        if model == 'np':
            m = baselines.NP(**model_kwargs)
        elif model == 'bnp':
            m = baselines.BNP(**model_kwargs)
        elif model == 'anp':
            m = baselines.ANP(**model_kwargs)
        elif model == 'abnp':
            m = baselines.ABNP(**model_kwargs)

        training_kwargs = {'training_steps': training_steps,
                           'batch_size': 5,
                           'learning_rate': 1e-2,
                           'final_learning_rate': 1e-3,
                           'num_samples': 32, # ignored for conditional family of NPs
                           'loss_function': 'mpl', # this is irrelevant, we just need one of the options that splits context and targets appropriately within train_meta_model
                           'ctxt_proportion_range': (0.1, 0.5),
                           'task_subsample_fraction': None,
                           'device_agnostic': True}
        
    
    ############# Model is now defined. Time to train the thing. ##############

    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)

    Xc_raw, yc_raw, Xt_raw, yt_raw = torch.load(PATH + f"/data/{dataset}/test_set.pt")
    Xc = Xc_raw.to(device=device, dtype=torch.float64)
    yc = yc_raw.to(device=device, dtype=torch.float64)
    Xt = Xt_raw.to(device=device, dtype=torch.float64)
    yt = yt_raw.to(device=device, dtype=torch.float64)
    num_predict_samps = 1000

    if model == 'bdnp' or model_type == 'np':
        md = torch.load(PATH + f"/data/{dataset}/train_sets.pt", weights_only=False)
        training_metrics = train_meta_model(m, md, **training_kwargs)
        with torch.no_grad():
            if model == 'bdnp':
                pred_yt = model(Xt, Xc, yc, num_samples=num_predict_samps, batch_size=50)[0]
            else:
                pred_yt = model(Xt, Xc, yc, num_samples=num_predict_samps)

    elif model_type == 'bnn':
        training_metrics = train_variational_model(m, (Xc, yc), **training_kwargs)
        with torch.no_grad():
            pred_yt = model(Xt, num_samples=num_predict_samps)

    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    for i, (key, value) in enumerate(training_metrics.items()):
        axes[i].plot(value)
        axes[i].set_xlabel(key)
        axes[i].grid()
        if key == 'elbo':
            axes[i].set_ylim([-5000, 500])
        elif key == 'e_ll':
            axes[i].set_ylim([-4000, 1000])
        elif key == 'kl':
            axes[i].set_ylim([0, 2000])
    plt.savefig(PATH + f"/figs/training/{model_codename}/pdfs/training_{seed}.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/training/{model_codename}/pngs/training_{seed}.png", bbox_inches="tight")
    plt.close()


    ############# Model is now trained. Time to evaluate the thing. ##############
    results = {}

    # log per-datapoint average posterior predictive density
    results['ppd'] = (m.likelihood.log_prob(pred_yt, yt).sum(-1).logsumexp((0, 1)) - torch.tensor(num_predict_samps * yt.shape[0]).log()).item()
    # normalise data for MAE and plotting:
    norm_consts = torch.load(PATH + f"/data/{dataset}/norm_consts.pt", weights_only=False)
    y_mean = norm_consts['y_mean']
    y_std = norm_consts['y_std']
    y_mean = y_mean.to(device=device, dtype=torch.float64)
    y_std = y_std.to(device=device, dtype=torch.float64)
    pred_yt = pred_yt * y_std + y_mean
    yt = yt * y_std + y_mean

    # per-datapoint average mean-absolute error between true targets and predictive mean.
    results['mae'] = ((pred_yt.mean(0) - yt).abs().mean()).item()

    with open(PATH + f"/results/{dataset}/{model_codename}/{seed}.json", 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goldilocks experiment.")
    parser.add_argument('--model', type=str, default=None, help='run codename')
    parser.add_argument('--dataset', type=str, default='abalone', help='Name of dataset for which to run the experiment.')
    parser.add_argument('--prior_trainability', type=float, default=None, help='Proportion of weights whose prior is trainable.')
    parser.add_argument('--training_steps', type=int, default=30_000, help='The number of training steps')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--seed', type=int, default=None, help='Seed number for repeat trials.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one available. Default False.')

    args = parser.parse_args()
    main(**vars(args))