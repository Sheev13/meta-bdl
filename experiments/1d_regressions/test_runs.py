import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import models
from models import baselines
from utils.training import train_meta_model
from utils.data_utils import ctxt_trgt_split, obtain_me_a_nice_gp_dataset_please, obtain_me_a_nice_sawtooth_dataset_please, obtain_me_a_nice_heaviside_dataset_please

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

codename = 'avi'
print(f"Codename: {codename}")
PATH = str(Path(__file__).resolve().parent)

Path(PATH + f"/test_figs/").mkdir(parents=True, exist_ok=True)
Path(PATH + f"/test_figs/{codename}").mkdir(parents=True, exist_ok=True)
Path(PATH + f"/test_figs/{codename}/pdfs").mkdir(parents=True, exist_ok=True)
Path(PATH + f"/test_figs/{codename}/pngs").mkdir(parents=True, exist_ok=True)

num_datasets = 10_000
md = []
# data_hypers = {'l': 1.0, 'kernel': 'per', 'p': 1, 'x_range': [-5.0, 5.0]}
# data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': [-4.0, 4.0]}
# data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
data_hypers = {'x_range': [-5.0, 5.0], 'l': 1, 'noise': 0.01}
for _ in range(num_datasets):
    # X, y = obtain_me_a_nice_gp_dataset_please(n_range=[10, 100], **data_hypers)
    # X, y = obtain_me_a_nice_sawtooth_dataset_please(n_range=[40, 100], **st_data_hypers)
    X, y = obtain_me_a_nice_heaviside_dataset_please(n_range=[10, 100], **data_hypers)
    md.append((X, y))

lik = models.GaussianLikelihood(y_dim=1, sigma_y=0.1, train=True)
bdnp = models.BDNP(x_dim=1,
                 y_dim=1,
                 hidden_dims=[48, 48],
                 prior_type=1,
                 likelihood=lik,
                 inf_dims=[48, 48],
                 use_final_layer_targets=True,
                 scale_prior=True,
                 nonlinearity=torch.nn.Tanh(),
                 )
bdnp.trainable_prior(True)

training_metrics = train_meta_model(
    bdnp,
    md,
    training_steps=50_000,
    batch_size=5,
    learning_rate=5e-3,
    final_learning_rate=1e-5,
    num_samples=32,
    loss_function='avi',
    ctxt_proportion_range=[0.1, 0.9],
    device_agnostic=True,
)

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
plt.savefig(PATH + f"/test_figs/{codename}/pdfs/training.pdf", bbox_inches="tight")
plt.savefig(PATH + f"/test_figs/{codename}/pngs/training.png", bbox_inches="tight")
plt.close()

xs = torch.linspace(-4.0, 4.0, 250).unsqueeze(-1)
samps = 100

x_lim = [-4.0, 4.0]
y_lim = [-3.0, 3.0]

# prior samples:
with torch.no_grad():
    prior_samps = bdnp(xs, None, None, num_samples=samps)[0]

plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), prior_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
plt.grid()
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.savefig(PATH + f"/test_figs/{codename}/pdfs/prior-predictive.pdf", bbox_inches="tight")
plt.savefig(PATH + f"/test_figs/{codename}/pngs/prior-predictive.png", bbox_inches="tight")
plt.close()

# single-datapoint samples:
test_md = [obtain_me_a_nice_heaviside_dataset_please(n_range=[1,2], **data_hypers) for _ in range(5)]
# test_md = [obtain_me_a_nice_gp_dataset_please(n_range=[1,2], **data_hypers) for _ in range(5)]

for i, (X, y) in enumerate(test_md):
    X_c, y_c = X.clone(), y.clone()
    with torch.no_grad():
        pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]

    plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
    plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
    plt.grid()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig(PATH + f"/test_figs/{codename}/pdfs/one-point-predictive-{i}.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/test_figs/{codename}/pngs/one-point-predictive-{i}.png", bbox_inches="tight")
    plt.close()

# multiple-datapoint samples:
test_md = [obtain_me_a_nice_heaviside_dataset_please(n_range=[2,10], **data_hypers) for _ in range(5)]
# test_md = [obtain_me_a_nice_gp_dataset_please(n_range=[2,10], **data_hypers) for _ in range(5)]
for i, (X, y) in enumerate(test_md):
    X_c, y_c = X.clone(), y.clone()
    with torch.no_grad():
        pred_samps = bdnp(xs, X_c, y_c, num_samples=samps)[0]

    plt.plot(xs.unsqueeze(0).repeat((samps, 1, 1)).squeeze(-1).T.cpu(), pred_samps.squeeze(-1).T.cpu(), linewidth=0.5, color='C0', alpha=0.5)
    plt.scatter(X_c.cpu(), y_c.cpu(), color='C1', zorder=10000)
    plt.grid()
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig(PATH + f"/test_figs/{codename}/pdfs/multi-point-predictive-{i}.pdf", bbox_inches="tight")
    plt.savefig(PATH + f"/test_figs/{codename}/pngs/multi-point-predictive-{i}.png", bbox_inches="tight")
    plt.close()