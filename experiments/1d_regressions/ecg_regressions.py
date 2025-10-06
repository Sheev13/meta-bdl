import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import Optional, List
import models
from models import baselines
from utils.training import train_meta_model, train_variational_model



def main(codename: Optional[str] = None,
         training_steps: int = 50_000,
         architecture: List[int] = [64, 64, 64],
         use_gpu: bool = False):

    print("codename: ", codename)

    if codename is None:
        raise ValueError("User failed to specify codename.")

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)

    dtp = torch.float64
    torch.set_default_dtype(dtp)

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)

    # define model classes, model kwargs, training funcs, training kwargs, seed

    nl = torch.nn.SiLU()

    with open(PATH +f"/ecg_data/{dataset}/metadata.json") as f:
        dataset_metadata = json.load(f)
    x_dim = int(dataset_metadata["dimensionality"])

    if model == 'bdnp':
        model_kwargs = {'x_dim': x_dim,
                        'y_dim': 1,
                        'likelihood': models.GaussianLikelihood(1, sigma_y=0.5 if dataset == 'paul15' else 0.1, train=True),
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
                m.set_prior_trainability(prior_trainability, from_front=True)

        training_kwargs = {'training_steps': 75_000,
                           'batch_size': 5,
                           'learning_rate': 1e-4,
                           'final_learning_rate': 5e-5,
                           'num_samples': 16,
                           'loss_function': 'pp-avi',
                           'ctxt_proportion_range': (0.1, 0.5),
                           'task_subsample_fraction': 0.5 if dataset == 'qm8' else None,
                           'within_task_batch_size': 512 if dataset == 'qm8' else None,
                           'device_agnostic': True}

    elif model_type == 'bnn':
        model_kwargs = {'x_dim': x_dim,
                        'y_dim': 1,
                        'likelihood': models.GaussianLikelihood(1, sigma_y=0.75 if dataset=='paul15' else 0.1, train=True if model=='givi' else False),
                        'hidden_dims': architecture,
                        'scale_prior': True,
                        'nonlinearity': nl}
        if model == 'givi':
            if dataset == 'abalone':
                n_ind = 128 
            elif dataset == 'paul15':
                n_ind =  192
            elif dataset == 'qm8':
                n_ind = 256
            model_kwargs['num_inducing'] = n_ind
        
        if model == 'mfvi':
            m = baselines.MFVIBNN(**model_kwargs)
        elif model == 'givi':
            m = baselines.GIVIBNN(**model_kwargs)

        training_kwargs = {'training_steps': 50_000 if model == 'givi' else 75_000,
                            'learning_rate': 5e-3,
                            'final_learning_rate': 1e-4,
                            'num_samples': 8,
                            'device_agnostic': True,
                            'retain_graph': model=='givi'}

    elif model_type == 'np':
        if model in ['np', 'bnp']:
            model_kwargs = {'x_dim': x_dim,
                            'y_dim': 1,
                            'lik': models.GaussianLikelihood(1, sigma_y=0.1, train=True), # should we train this?
                            'encoder_dims': [256, 256, 256], # aything less and these NPs are just crap.
                            'decoder_dims': [256, 256, 256],
                            'nonlinearity': nl}
            training_kwargs = {'training_steps': 500_000,
                               'batch_size': 5,
                               'learning_rate': 1e-4,
                               'final_learning_rate': 1e-5,
                               'num_samples': 16, # ignored for conditional family of NPs
                               'loss_function': 'mpl', # this is irrelevant, we just need one of the options that splits context and targets appropriately within train_meta_model
                               'ctxt_proportion_range': (0.1, 0.5),
                               'task_subsample_fraction': None,
                               'device_agnostic': True}
        elif model == 'ar-tnp':
            model_kwargs = {'x_dim': x_dim,
                            'y_dim': 1,
                            'num_layers': 3,
                            'r_dim': 256,
                            'nonlinearity': nl}
            training_kwargs = {'training_steps': 50_000,
                               'batch_size': 5,
                               'learning_rate': 5e-5,
                               'final_learning_rate': 1e-5,
                               'loss_function': 'mpl', # this is irrelevant, we just need one of the options that splits context and targets appropriately within train_meta_model
                               'ctxt_proportion_range': (0.1, 0.5),
                               'task_subsample_fraction': 0.25 if dataset == 'qm8' else None,
                               'device_agnostic': True}
        
        if model == 'np':
            m = baselines.NP(**model_kwargs)
        elif model == 'bnp':
            m = baselines.BNP(**model_kwargs)
        elif model == 'ar-tnp':
            m = baselines.TNP(**model_kwargs)
        
    
    ############# Model is now defined. Time to train the thing. ##############
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)

    if dataset == 'abalone':
        Xc_raw, yc_raw, Xt_raw, yt_raw = torch.load(PATH + f"/data/abalone/test_set.pt")
    elif dataset == 'qm8':
        Xc_raw, yc_raw, Xt_raw, yt_raw = torch.load(PATH + f"/data/qm8/test_sets.pt")[0]
    elif dataset == 'paul15':
        Xc_raw, yc_raw, Xt_raw, yt_raw = torch.load(PATH + f"/data/paul15/test_sets.pt")[-4] # this one has 329 datapoints total
    Xc = Xc_raw.to(device=device, dtype=dtp)
    yc = yc_raw.to(device=device, dtype=dtp)
    Xt = Xt_raw.to(device=device, dtype=dtp)
    yt = yt_raw.to(device=device, dtype=dtp)
    num_predict_samps = 1000

    if model == 'bdnp' or model_type == 'np':
        md = torch.load(PATH + f"/data/{dataset}/train_sets.pt", weights_only=False)
        training_metrics = train_meta_model(m, md, **training_kwargs)
        with torch.no_grad():
            if model == 'bdnp':
                pred_yt = m(Xt, Xc, yc, num_samples=num_predict_samps, batch_size=50)[0]
            elif model in ['np', 'bnp']:
                pred_yt = m(Xt, Xc, yc, num_samples=num_predict_samps)
            elif model == 'ar-tnp':
                print("executing autoregressive TNP forward pass...")
                pred_yt, ll = m.autoregressive_forward(Xt,
                                                       Xc,
                                                       yc,
                                                       yt,
                                                       num_samples=num_predict_samps,
                                                       compute_ll=True,
                                                       verbose=True)
                print("All done my boy.")

    elif model_type == 'bnn':
        training_metrics = train_variational_model(m, (Xc, yc), **training_kwargs)
        if model == 'givi': # circumvent memory limitations due to large number of inducing points
            pred_yt = torch.zeros((1000, Xt.shape[0], 1))
            for p in range(10):
                with torch.no_grad():
                    pred_yt[p*100:(p+1)*100,:,:] = m(Xt, num_samples=100)
        else:
            with torch.no_grad():
                pred_yt = m(Xt, num_samples=num_predict_samps)

    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    for i, (key, value) in enumerate(training_metrics.items()):
        omitted_steps = 250
        if len(training_metrics) == 1:
            a = axes
        else:
            a = axes[i]
        a.plot(value[omitted_steps:])
        a.set_xlabel(key)
        a.grid()
    plt.savefig(PATH + f"/figs/training/{dataset}/{model_codename}/pdfs/training_{seed}.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/training/{dataset}/{model_codename}/pngs/training_{seed}.png", bbox_inches="tight")
    plt.close()
