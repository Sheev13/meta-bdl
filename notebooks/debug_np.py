import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
import models
from models import baselines
from utils.training import train_meta_model
from utils.data_utils import ctxt_trgt_split, obtain_me_a_nice_gp_dataset_please
import dill


def main(model: str = 'np', use_gpu: bool = False):

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    if model is None:
        raise ValueError("User failed to specify which model to use.")
    assert model.upper() in ['NP', 'LANP', 'BNP', 'ABNP']

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + f"/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{model.lower()}").mkdir(parents=True, exist_ok=True)


    num_datasets = 10_000
    md = []
    # gp_data_hypers = {'l': 1.0, 'kernel': 'per', 'p': 1, 'x_range': [-5.0, 5.0]}
    gp_data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': [-4.0, 4.0]}
    # st_data_hypers = {'p': 1.0, 'random_shift': False, 'random_gradient': False, 'x_range': [-5.0, 5.0]}
    # h_data_hypers = {'x_range': [-5.0, 5.0], 'l': 1}
    for _ in range(num_datasets):
        X, y = obtain_me_a_nice_gp_dataset_please(n_range=[25, 50], **gp_data_hypers)
        # X, y = obtain_me_a_nice_sawtooth_dataset_please(n_range=[40, 100], **st_data_hypers)
        # X, y = obtain_me_a_nice_heaviside_dataset_please(n_range=[40, 100], **h_data_hypers)
        md.append((X, y))

    lik = models.GaussianLikelihood(y_dim=1, sigma_y=0.05, train=False)

    if model.upper() == 'NP':
        model_class = baselines.NP
    elif model.upper() == 'LANP':
        model_class = baselines.LANP
    elif model.upper() == 'BNP':
        model_class = baselines.BNP
    elif model.upper() == 'ABNP':
        model_class = baselines.ABNP

    np = model_class(x_dim=1,
                     y_dim=1,
                     lik=lik,
                     encoder_dims=[256, 256, 256],
                     decoder_dims=[256, 256, 256],
                     nonlinearity=torch.nn.ReLU(),
                     )
    

    training_metrics = train_meta_model(
        np,
        md,
        training_steps=500_000,
        batch_size=5,
        learning_rate=5e-4,
        final_learning_rate=1e-5,
        num_samples=16,
        loss_function='mpl',
        ctxt_proportion_range=[0.1, 0.5],
        device_agnostic=True,
    )
    Path(PATH + f"/saved_models").mkdir(parents=True, exist_ok=True)
    torch.save(np, PATH + f"/saved_models/{model.lower()}.pt", pickle_module=dill)
    fig, axes = plt.subplots(1, len(training_metrics), figsize=(3*len(training_metrics), 1))
    omitted_steps = 100
    if len(training_metrics) == 1:
        for key, value in training_metrics.items():
            axes.plot(value[omitted_steps:])
            axes.set_xlabel(key)
            axes.grid()
    else:
        for i, (key, value) in enumerate(training_metrics.items()):
            axes[i].plot(value[omitted_steps:])
            axes[i].set_xlabel(key)
            axes[i].grid()
        # axes[i].set_ylim([-100, 400])
    plt.savefig(PATH + f"/figs/{model.lower()}/training.png", bbox_inches='tight')
    plt.close()

    gp_data_hypers['x_range'] = [-2.0, 2.0]
    for i in range(20):
        X, y = obtain_me_a_nice_gp_dataset_please(n_range=[1, 40], **gp_data_hypers)
        X_c, y_c = X.clone(), y.clone()
        samps = 100

        xs = torch.linspace(-2.5, 2.5, 200).unsqueeze(-1)
        with torch.no_grad():
            preds = np(xs, X_c, y_c, num_samples=samps).cpu()

        plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), preds.squeeze(-1).T, linewidth=0.5, color='C0', alpha=0.5)
        if X_c is not None:
            plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
        plt.grid()
        plt.xlim([-2.5, 2.5])
        plt.ylim([-5.0, 5.0])
        plt.savefig(PATH + f"/figs/{model.lower()}/{i}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug NP.")
    parser.add_argument('--model', type=str, default=None, help='Which NP to use.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one available. Default False.')
    args = parser.parse_args()
    main(**vars(args))