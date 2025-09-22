from typing import List, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from utils.bnn_prior import GaussianBNNPrior
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

class MetaDataset(Dataset):
    def __init__(self, datasets: List[Any]):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx: int):
        return self.datasets[idx]


def ctxt_trgt_split(X: torch.Tensor, y: torch.Tensor, ctxt_proportion_range: Optional[Tuple[float]]=None, ctxt_proportion: Optional[float] = None):
    if ctxt_proportion is None:
        if ctxt_proportion_range[1] < ctxt_proportion_range[0]:
            ctxt_proportion_range = ctxt_proportion_range[::-1]
        if ctxt_proportion_range[0] < 0.0:
            raise ValueError("Cannot have a negative proportion of context points.")
        if ctxt_proportion_range[1] > 1.0:
            raise ValueError("Cannot have a proportion of context points that is greater than 1.")
        
        proportion = torch.rand((1,)) * (ctxt_proportion_range[1] - ctxt_proportion_range[0]) + ctxt_proportion_range[0]
    
    else:
        if ctxt_proportion < 0.0:
            raise ValueError("Cannot have a negative proportion of context points.")
        if ctxt_proportion > 1.0:
            raise ValueError("Cannot have a proportion of context points that is greater than 1.")
        
        proportion = ctxt_proportion

    num_ctxt = int(X.shape[0] * proportion)
    inds = torch.randperm(X.shape[0])
    ctxt_i = inds[:num_ctxt]
    trgt_i = inds[num_ctxt:]

    X_c, y_c = X[ctxt_i], y[ctxt_i]
    X_t, y_t = X[trgt_i], y[trgt_i]

    if X_c.shape[0] == 0:
        X_c, y_c = X_t[0], y_t[0]
    elif X_t.shape[0] == 0:
        X_t, y_t = X_c[0], y_c[0]

    x_dim, y_dim = X.shape[-1], y.shape[-1]
    if len(X_c.shape) == 1 and X_c.shape[0] == x_dim:
        X_c = X_c.unsqueeze(0)
    if len(y_c.shape) == 1 and y_c.shape[0] == y_dim:
        y_c = y_c.unsqueeze(0)
    if len(X_t.shape) == 1 and X_t.shape[0] == x_dim:
        X_t = X_t.unsqueeze(0)
    if len(y_t.shape) == 1 and y_t.shape[0] == y_dim:
        y_t = y_t.unsqueeze(0)

    return (X_c, y_c, X_t, y_t)


def obtain_me_a_nice_sawtooth_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, noise_range=None, p=1.0, p_range=None, random_linear=False, random_shift=False, m=1, random_gradient=False):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))

    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    if random_gradient:
        m += torch.rand((1,)) * 2 + 0.5 

    if noise_range is not None:
        noise = torch.rand((1,)) * (max(noise_range) - min(noise_range)) + min(noise_range)
    if p_range is not None:
        p = torch.rand((1,)) * (max(p_range) - min(p_range)) + min(p_range)
    
    if random_shift:
        s = torch.rand(1) * p  # random shift in [0, p)
    else:
        s = 0
    f_x = m * (torch.remainder(X + s, p) - 0.5*p)

    if random_linear:
        f_x += torch.randn((1,)) * X / 3 + torch.randn((1,))*0.25

    y = f_x + torch.randn_like(f_x) * noise
    return X, y

def obtain_me_a_nice_heaviside_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, l=1.0):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)
    Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square()) + torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    f_x -= f_x.mean()
    discrete_f_x = torch.where(f_x > 0, 1.0, -1.0)
    return X, discrete_f_x + torch.randn_like(f_x) * noise

def obtain_me_a_nice_gp_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, l=1.0, noise_range=None, l_range=None, kernel='se', p=1.0, p_range=None, binary_2d=False):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    if binary_2d:
        X = torch.rand((n, 2)) * (max(x_range) - min(x_range)) + min(x_range)
    else:
        X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    if noise_range is not None:
        noise = torch.rand((1,)) * (max(noise_range) - min(noise_range)) + min(noise_range)
    if l_range is not None:
        l = torch.rand((1,)) * (max(l_range) - min(l_range)) + min(l_range)
    if p_range is not None:
        p = torch.rand((1,)) * (max(p_range) - min(p_range)) + min(p_range)
    
    if kernel == 'se':
        Sigma = torch.exp(-0.5 * torch.cdist(X/l, X/l, p=2).square())
    elif kernel == 'per':
        Sigma = torch.exp(-2.0 * torch.sin(torch.pi * torch.cdist(X, X) / p).square() / l**2) 
    Sigma += torch.eye(n) * 1e-5
    z = torch.randn((n, 1))
    f_x = torch.linalg.cholesky(Sigma) @ z
    if binary_2d:
        y = torch.distributions.Bernoulli(logits=f_x * 2.5).sample()
    else:
        y = f_x + torch.randn_like(f_x) * noise
    return X, y

def obtain_me_a_nice_bnn_dataset_please(x_range=[-4.0, 4.0], n_range=[5, 100], noise=0.05, hidden_dims=[20, 20], scale_prior=True, nonlinearity=torch.nn.Tanh()):
    n = torch.randint(low=min(n_range), high=max(n_range), size=(1,))
    X = torch.rand((n, 1)) * (max(x_range) - min(x_range)) + min(x_range)

    bnn_prior = GaussianBNNPrior(1, 1, hidden_dims, scale_prior=scale_prior, nonlinearity=nonlinearity)
    f_x = bnn_prior(X, num_samples=1).squeeze(0)
    y = f_x + torch.randn_like(f_x) * noise

    return X, y





########################## Image data utils ##########################

def generate_mask(image_shape: torch.Size, proportion: float):
    mask = torch.zeros(image_shape)
    num_ones = int(mask.numel() * proportion)
    ones_idx = torch.randperm(mask.numel())[:num_ones]
    mask.view(-1)[ones_idx] = 1
    return mask


def vis_ctxt_img(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    assert len(image.shape) == 3
    colours = image.shape[-1]
    assert colours in [1, 3]  # either greyscale or RGB

    mask = mask.bool()
    if colours == 1:
        image = image.repeat((1, 1, 3))
        blue = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(-1),
                torch.zeros_like(mask).unsqueeze(-1),
                torch.ones_like(mask).unsqueeze(-1),
            ),
            dim=-1,
        )
        image = torch.where(mask.unsqueeze(-1).repeat((1, 1, 3)), image, blue)
    elif colours == 3:
        grey = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(-1),
                torch.zeros_like(mask).unsqueeze(-1),
                torch.ones_like(mask).unsqueeze(-1),
            ),
            dim=-1,
        )
        image = torch.where(mask, image, grey)

    return image # shape (h, w, c)


def img_to_dataset(img: torch.Tensor, mask: Optional[torch.Tensor] = None):
    num_x1, num_x2 = img.shape[:2]
    num_pix = num_x1 * num_x2
    x1_range = torch.linspace(-1, 1, num_x1)
    x2_range = torch.linspace(-1, 1, num_x2)
    xm1, xm2 = torch.meshgrid(x1_range, x2_range, indexing="xy")
    x1 = xm1.flatten()
    x2 = xm2.flatten()

    x = torch.stack((x1, x2)).transpose(-1, -2) # shape (784, 2) after transpose
    y = img.reshape((num_pix, -1))

    if mask is not None:
        x = x[mask.flatten().bool(), :]
        y = y[mask.flatten().bool(), :]

    return x, y

def dataset_to_img(Y):
    # Y expected to be of shape (1, 784)
    return Y.reshape(1, 28, 28).permute(1, 2, 0)

def test_grid(image_shape: torch.Size):
    num_x1, num_x2 = image_shape
    x1_range = torch.linspace(-1, 1, num_x1)
    x2_range = torch.linspace(-1, 1, num_x2)
    xm1, xm2 = torch.meshgrid(x1_range, x2_range, indexing="xy")
    x1 = xm1.flatten()
    x2 = xm2.flatten()

    return torch.stack((x1, x2)).transpose(-1, -2)





###################### Era5 visualisation utils ########################


def vis_era5_preds(lons, lats, preds):
    """
    Plot precipitation heatmap over Europe.
    
    lons: 1D array of longitudes
    lats: 1D array of latitudes
    preds: 1D array of predictions aligned with meshgrid(lon, lat)
    """
    # Reshape into grid (lat, lon)
    Lon, Lat = np.meshgrid(lons, lats)
    Z = preds.reshape(len(lats), len(lons))

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add geographic context
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    ax.set_extent([min(lons), max(lons), min(lats), max(lats)])  

    # Plot rainfall
    im = ax.pcolormesh(lons, lats, Z, cmap="Blues", shading="auto")

    # Add colorbar
    cb = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7, label="Precipitation (mm)")

    plt.show()


def scrambled_ctxt_trgt_to_grid(xs: torch.Tensor, ys: torch.Tensor):
    assert xs.shape[1] == 2 # shape (n, 2)
    assert ys.shape[2] == 1 # shape (samples, n, 1)
    x1, x2 = xs[:,0].cpu().numpy(), xs[:,1].cpu().numpy()
    Ys = ys[:,:,0].cpu().numpy()
    x1_uni, x2_uni = np.unique(x1), np.unique(x2)
    Z = np.empty((Ys.shape[0], len(x2_uni), len(x1_uni)))
    x1_to_idx = {val: i for i, val in enumerate(x1_uni)}
    x2_to_idx = {val: i for i, val in enumerate(x2_uni)}

    for i, x1i_x2i in enumerate(zip(x1, x2)):
        x1i, x2i = x1i_x2i
        ix1 = x1_to_idx[x1i]
        ix2 = x2_to_idx[x2i]
        Z[:, ix2, ix1] = Ys[:, i]

    xx1, xx2 = np.meshgrid(x1_uni, x2_uni)

    return xx1, xx2, Z


def scrambled_sprs_to_masked_grid(xs: torch.Tensor, ys: torch.Tensor, xx1: np.ndarray, xx2: np.ndarray):
    # xs and ys correspond to a context set, i.e. m << n
    # xx1 and xx2 come from scrambled_ctxt_trgt_to_grid applied to the full set of points, i.e. all n points.
    assert xs.shape[1] == 2 # shape (m, 2)
    assert ys.shape[1] == 1 # shape (m, 1)
    x1, x2 = xs[:,0].cpu().numpy(), xs[:,1].cpu().numpy()
    Ys = ys[:,0].cpu().numpy()
    x1_uni, x2_uni = np.unique(xx1[0,:]), np.unique(xx2[:,0])
    ix1, ix2 = np.searchsorted(x1_uni, x1), np.searchsorted(x2_uni, x2)
    Z = np.full(xx1.shape, np.nan)
    Z[ix2, ix1] = Ys.ravel()
    Z_masked = np.ma.masked_invalid(Z)

    return Z_masked


    