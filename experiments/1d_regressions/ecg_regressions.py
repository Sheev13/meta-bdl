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
    fs=100,
    noise=0.001,
    n_range=None
):
    """Generate one synthetic ECG waveform and return (X, Y) torch tensors."""
    d = 6.0
    signal = nk.ecg_simulate(
        duration=8.0,
        sampling_rate=fs,
        heart_rate=20,
        noise=noise,
        method='simple'
    )
    signal -= np.mean(signal)
    signal /= (2*np.max(signal))

    single_wave_block = signal[int(3.5*fs):int(6.5*fs)] # 3s long
    zeros_block = np.concatenate(3*[signal[int(7.0*fs):]], axis=0) # 3s long
    split = int(np.random.uniform(0.0, 3.0) * fs)
    full = np.concatenate((zeros_block[:split], single_wave_block, zeros_block[split:]), axis=0)

    # normalize and convert to tensors
    t = np.linspace(-d/2, d/2, len(full), endpoint=False)
    X = torch.tensor(t, dtype=torch.float64).unsqueeze(1)
    Y = torch.tensor(full, dtype=torch.float64).unsqueeze(1)

    if n_range is not None:
        n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
        inds = torch.randperm(len(full))[:n]
        return X[inds], Y[inds]

    return X, Y


def main(codename: Optional[str] = None, preload: Optional[str] = None):

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

    if preload is not None:
        bdnp = torch.load(PATH + f"/saved_models/bdnp-{preload}", weights_only=False)
    else:
        architecture = [64, 64]
        bdnp = models.BDNP(x_dim=1,
                        y_dim=1,
                        likelihood=models.GaussianLikelihood(1, sigma_y=0.05, train=True),
                        hidden_dims=architecture,
                        prior_type=1,
                        inf_dims=architecture,
                        use_final_layer_targets=True,
                        scale_prior=True,
                        nonlinearity = torch.nn.Tanh(),
                        )
    bdnp.trainable_prior(True)

    training_metrics = train_meta_model(bdnp,
                                        md,
                                        training_steps=500_000,
                                        batch_size=5,
                                        learning_rate=5e-4,
                                        final_learning_rate=5e-5,
                                        num_samples=24,
                                        loss_function='pp-avi',
                                        ctxt_proportion_range=(2/600, 32/600),
                                        device_agnostic=True,
                                        beta=1.0,
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


    xs = torch.linspace(-3.0, 3.0, 250).unsqueeze(-1)
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
    test_md = [generate_synthetic_ecg_task(n_range=(20, 50)) for _ in range(10)]
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
    parser.add_argument('--preload', type=str, default=None, help='Codename of saved model to preload and continue training')
    args = parser.parse_args()
    main(**vars(args))