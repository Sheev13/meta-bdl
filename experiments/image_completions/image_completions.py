import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from typing import List, Optional, Tuple

import models
from utils.training import train_meta_model
from utils.data_utils import img_to_dataset, test_grid, generate_mask, dataset_to_img, vis_ctxt_img
from base_networks.base_architectures import Sin, SharpTanh

def save_image(img, path, cmap=None):
    _, ax = plt.subplots(1, 1, figsize=(3, 3))
    grey = 'gray'
    if cmap is None:
        cm = grey
    else:
        cm = cmap
    ax.imshow(img, cmap=cm)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def build_MNIST_meta_dataset(test=False):
    mnist = datasets.MNIST(
        root="experiments/image_completions/mnist_stuff", train=not test, download=True,
        transform=transforms.ToTensor()
    )

    md = []
    for img, _ in mnist:
        X, y = img_to_dataset(img)
        md.append((X, y))
    
    return md
    
def main(
        codename=None,
        architecture=[250, 250, 250],
        nonlinearity='relu',
        residual=False,
        transformer_layers=None,
        transformer_width=None,
        use_act=False,
        training_steps=10_000,
        batch_size=5,
        learning_rate=1e-3,
        final_learning_rate=5e-5,
        loss_function='pp-avi',
        num_samples=8,
        ctxt_proportion_range=(0.1, 0.9),
        use_pretrained=False,
        use_gpu=True,
):
    args_dict = locals()

    if codename is None:
        raise ValueError("User failed to specify a codename for this training run.")
    else:
        codename = codename.lower()

    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU found, falling back to CPU")
            device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)
        print("device type: ", device)
        # model.to(device, dtype=torch.float32)
        # print("Moving dataset to device...")
        # md = [(X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)) for (X, y) in md]
        # print("Done.")

    PATH = str(Path(__file__).resolve().parent)

    Path(PATH + "/figs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs/prior_samples").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs/prior_samples").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pdfs/posterior_samples").mkdir(parents=True, exist_ok=True)
    Path(PATH + f"/figs/{codename}/pngs/posterior_samples").mkdir(parents=True, exist_ok=True)

    test_md = build_MNIST_meta_dataset(test=True)

    if not use_pretrained: # if training from scratch (default scenario)
        with open(PATH + f"/training-configs/{codename}-config.json", 'w') as f:
            json.dump(args_dict, f, indent=4)

        md = build_MNIST_meta_dataset()

        bdnp = models.BDNP(x_dim=2,
                           y_dim=1,
                           hidden_dims=architecture,
                           prior_type=1,
                           likelihood=models.BernoulliLikelihood(),
                           residual=residual,
                           inf_dims=architecture,
                           use_final_layer_targets=False,
                           transformer_layers=transformer_layers,
                           transformer_width=transformer_width,
                           use_act=use_act,
                           scale_prior=True,
                           nonlinearity=nonlinearity)
        bdnp.trainable_prior(True)

        training_metrics = train_meta_model(
            bdnp,
            md,
            training_steps=training_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_learning_rate=final_learning_rate,
            num_samples=num_samples,
            loss_function=loss_function,
            ctxt_proportion_range=ctxt_proportion_range,
            device_agnostic=True,
        )

        Path(PATH + "/saved_models").mkdir(parents=True, exist_ok=True)
        torch.save(bdnp, PATH + f'/saved_models/bdnp-{codename}')

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

        plt.savefig(PATH + f"/figs/{codename}/pdfs/training.pdf", bbox_inches="tight")
        plt.savefig(PATH + f"/figs/{codename}/pngs/training.png", bbox_inches="tight")
        plt.close()



    else: # if using an already-trained model
        bdnp = torch.load(PATH + f'/saved_models/bdnp-{codename}', weights_only=False)

    X_t = test_grid([28, 28])
    samps = 10

    # prior samples:
    with torch.no_grad():
        prior_samps = bdnp.likelihood(bdnp(X_t, None, None, num_samples=samps)[0]) # shape (10, 784, 1)

    for i in range(samps):
        pred_img = prior_samps[i,:].reshape((28, 28, 1))
        save_image(pred_img, PATH + f"/figs/{codename}/pdfs/prior_samples/sample-{i}.pdf")
        save_image(pred_img, PATH + f"/figs/{codename}/pngs/prior_samples/sample-{i}.png")

    
    # posterior samples:
    proportions = [0.05, 0.1, 0.2, 0.5, 0.75]
    for p in proportions:
        Path(PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}").mkdir(parents=True, exist_ok=True)
        Path(PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}").mkdir(parents=True, exist_ok=True)
        for j in range(5):
            Path(PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}").mkdir(parents=True, exist_ok=True)
            Path(PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}").mkdir(parents=True, exist_ok=True)
            _, Y = test_md.pop()
            img = dataset_to_img(Y)
            mask = generate_mask((28, 28), p)
            ctxt_img = vis_ctxt_img(img, mask)
            X_c, Y_c = img_to_dataset(img, mask=mask)

            save_image(img, PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}/full.pdf")
            save_image(img, PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}/full.png")

            save_image(ctxt_img, PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}/ctxt.pdf")
            save_image(ctxt_img, PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}/ctxt.png")

            with torch.no_grad():
                posterior_samps = bdnp.likelihood(bdnp(X_t, X_c, Y_c, num_samples=samps)[0])

            for i in range(samps):
                pred_img = posterior_samps[i,:].reshape((28, 28, 1))
                save_image(pred_img, PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}/sample-{i}.pdf")
                save_image(pred_img, PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}/sample-{i}.png")

            pp = posterior_samps.mean(0).reshape((28, 28, 1)) # posterior predictive, i.e. posterior predictive/BMA pixelwise class probabilities
            save_image(pp, PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}/pred_mean.pdf")
            save_image(pp, PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}/pred_mean.png")
            std = (pp * (1 - pp)).std() # variance for Bernoulli distribution with probability p is p(1-p)
            save_image(std, PATH + f"/figs/{codename}/pdfs/posterior_samples/ctxt-{p}/image-{j}/pred_std.pdf", cmap='inferno')
            save_image(std, PATH + f"/figs/{codename}/pngs/posterior_samples/ctxt-{p}/image-{j}/pred_std.png", cmap='inferno')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDNP image completions")
    parser.add_argument('--codename', type=str, default=None, help='Codename for training run')
    parser.add_argument('--architecture', type=int, nargs='+', default=[250, 250, 250], help='Hidden layer dims of BDNP and inference nets')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='Elementwise-acting nonlinearity')
    parser.add_argument('--residual', action='store_true', help='Is the primary BDNP network residual?')
    parser.add_argument('--transformer_layers', type=int, default=None, help='Number of attention blocks in AttBDNP inference nets')
    parser.add_argument('--transformer_width', type=int, default=None, help='Representation dimension of AttBNDP inference nets')
    parser.add_argument('--use_act', action='store_true', help='Pass current layer activations to inference nets?')
    parser.add_argument('--training_steps', type=int, default=20_000, help='The number of training steps')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of datasets used to estimate objective at each step')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='(Initial) learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5, help='Final learning rate, linearly tempered')
    parser.add_argument('--loss_function', type=str, default='pp-avi', help='Objective function (vi or npvi)')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of MC samples to estimate expected log likelihood.')
    parser.add_argument('--ctxt_proportion_range', type=float, nargs='+', default=[0.1, 0.9], help='Range of context set/full set proportion for each sampled task')
    parser.add_argument('--use_pretrained', action='store_true', help='Train a new BDNP (default), or load a pre-trained one.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if one is available')

    args = parser.parse_args()
    main(**vars(args))