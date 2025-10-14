import torch
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
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
from base_networks.base_architectures import Sin

print("ECG Regression...")

def random_window(total_l, window_l):
    upper = total_l - window_l
    start = np.random.randint(upper)
    return int(start), int(start+window_l)

def generate_synthetic_ecg_task(
    fs=200,
    duration=4.0,
    hr_mean=45,
    hr_std=0, # optionally remove this for easier meta-task
    noise=0.001,
    n_range=None
):
    """Generate one synthetic ECG waveform and return (X, Y) torch tensors."""
    signal = nk.ecg_simulate(
        duration=duration*2,
        sampling_rate=fs,
        heart_rate=max(hr_mean + np.random.randn() * hr_std, 30.0),
        noise=noise,
        method='simple'
    )

    n_tot = fs * duration * 2
    n_wind = fs * duration
    start_ind, end_ind = random_window(n_tot, n_wind)
    signal = signal[start_ind:end_ind]
    # signal  =signal[:int(n_wind)]

    # normalize and convert to tensors
    signal -= np.mean(signal)
    signal /= (4*np.max(signal))
    t = np.linspace(-duration/2, duration/2, len(signal), endpoint=False)
    X = torch.tensor(t, dtype=torch.float64).unsqueeze(1)
    Y = torch.tensor(signal, dtype=torch.float64).unsqueeze(1)

    if n_range is not None:
        n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
        inds = torch.randperm(len(signal))[:n]
        return X[inds], Y[inds]

    return X, Y


def main(codename: Optional[str] = None):

    if codename is None:
        raise ValueError("User failed to specify codename.")
    
    print("codename: ", codename, ". ECG regressions.")

    if torch.cuda.is_available():
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

    # create dataset
    md = [generate_synthetic_ecg_task() for _ in range(100_000)]

    architecture = [64, 64]

    bdnp = models.BDNP(x_dim=1,
                       y_dim=1,
                       likelihood=models.GaussianLikelihood(1, sigma_y=0.05, train=True),
                       hidden_dims=architecture,
                       prior_type=1,
                       inf_dims=architecture,
                       use_final_layer_targets=True,
                       scale_prior=True,
                    #    nonlinearity=Sin(),
                       nonlinearity = torch.nn.SiLU(),
                      )
    bdnp.trainable_prior(True)

    training_metrics = train_meta_model(bdnp,
                                        md,
                                        training_steps=150_000,
                                        batch_size=5,
                                        learning_rate=1e-3, # change to 1e-2 or 5e-3
                                        final_learning_rate=1e-5,
                                        num_samples=16,
                                        loss_function='pp-avi',
                                        ctxt_proportion_range=(0.0025, 0.25),
                                        # task_subsample_fraction=0.25,
                                        device_agnostic=True,
                                       )
    
    torch.save(bdnp, PATH + f'/saved_models/bdnp-{codename}')

    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    omitted_steps = 100
    for i, (key, value) in enumerate(training_metrics.items()):
        axes[i].plot(value[omitted_steps:])
        axes[i].set_xlabel(key)
        axes[i].grid()

    plt.savefig(PATH + f"/figs/{codename}/pdfs/training.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/{codename}/pngs/training.png", bbox_inches="tight")
    plt.close()


    xs = torch.linspace(-2.0, 2.0, 250).unsqueeze(-1)
    samps = 25

    # x_lim = [0.0, 5.0]
    x_lim = None
    y_lim = [-0.5, 1.0]

    # prior samples:
    with torch.no_grad():
        prior_samps = bdnp(xs, None, None, num_samples=samps)[0]

    plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), prior_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
    plt.grid()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig(PATH + f"/figs/{codename}/pdfs/prior-predictive.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/figs/{codename}/pngs/prior-predictive.png", bbox_inches="tight")
    plt.close()


    # single-datapoint samples:
    test_md = [generate_synthetic_ecg_task(n_range=(1, 2)) for _ in range(5)]
    for i, (X, y) in enumerate(test_md):
        X_c, y_c = X.clone(), y.clone()
        with torch.no_grad():
            pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
        
        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/{codename}/pdfs/one-point-predictive-{i}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/one-point-predictive-{i}.png", bbox_inches="tight")
        plt.close()


    # multiple-datapoint samples:
    test_md = [generate_synthetic_ecg_task(n_range=(2, 10)) for _ in range(5)]
    for i, (X, y) in enumerate(test_md):
        X_c, y_c = X.clone(), y.clone()
        with torch.no_grad():
            pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
        
        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/{codename}/pdfs/multi-point-predictive-{i}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/multi-point-predictive-{i}.png", bbox_inches="tight")
        plt.close()

    # many-datapoint samples:
    test_md = [generate_synthetic_ecg_task(n_range=(50, 200)) for _ in range(10)]
    for i, (X, y) in enumerate(test_md):
        X_c, y_c = X.clone(), y.clone()
        with torch.no_grad():
            pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]
        
        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
        plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(PATH + f"/figs/{codename}/pdfs/many-point-predictive-{i}.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/many-point-predictive-{i}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP ECG")
    parser.add_argument('--codename', type=str, default=None, help='Codename for training run')
    args = parser.parse_args()
    main(**vars(args))