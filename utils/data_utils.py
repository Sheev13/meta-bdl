from typing import List, Any, Tuple, Optional

import torch
# import torchvision
from torch.utils.data import Dataset
from utils.bnn_prior import GaussianBNNPrior
from tqdm import tqdm

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


# def random_mask(image: torch.Tensor, ratio_range: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     assert len(ratio) == 2
#     ratio = torch.zeros((1,)).uniform_(from=ratio_range[0], to=ratio_range[1])
#     dims = image.shape[-2:]
#     mask = (torch.Tensor(dims).uniform_() < ratio)
#     return image * mask, mask


def vis_ctxt_img(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    assert len(image.shape) == 3
    colours = image.shape[0]
    assert colours in [1, 3]  # either greyscale or RGB

    mask = mask.bool()
    if colours == 1:
        image = image.repeat((3, 1, 1))
        blue = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(0),
                torch.zeros_like(mask).unsqueeze(0),
                torch.ones_like(mask).unsqueeze(0),
            ),
            dim=0,
        )
        image = torch.where(mask, image, blue)
    elif colours == 3:
        grey = torch.cat(
            (
                torch.zeros_like(mask).unsqueeze(0),
                torch.zeros_like(mask).unsqueeze(0),
                torch.ones_like(mask).unsqueeze(0),
            ),
            dim=0,
        )
        image = torch.where(mask, image, grey)

    return image.permute(1, 2, 0)  # permutation needed for matplotlib


def img_for_reg(
    img: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_x1, num_x2 = img.shape[-2:]
    num_pix = num_x1 * num_x2
    x1_range = torch.linspace(-1, 1, num_x1)
    x2_range = torch.linspace(-1, 1, num_x2)
    xm1, xm2 = torch.meshgrid(x1_range, x2_range, indexing="xy")
    x1 = xm1.flatten()
    x2 = xm2.flatten()

    x = torch.stack((x1, x2)).transpose(-1, -2)
    y = img.reshape(-1, num_pix)

    x_c = x[mask.flatten().bool()]
    x_t = x[~mask.flatten().bool()]
    y_c = y[:, mask.flatten().bool()]
    y_t = y[:, ~mask.flatten().bool()]
    return x_c, y_c.T, x_t, y_t.T, x, y.T


def test_grid(image_shape: torch.Size):
    num_x1, num_x2 = image_shape
    num_pix = num_x1 * num_x2
    x1_range = torch.linspace(-1, 1, num_x1)
    x2_range = torch.linspace(-1, 1, num_x2)
    xm1, xm2 = torch.meshgrid(x1_range, x2_range, indexing="xy")
    x1 = xm1.flatten()
    x2 = xm2.flatten()

    return torch.stack((x1, x2)).transpose(-1, -2)


def image_tensor_to_dataset(image: torch.Tensor, mask_proportion_range: Tuple[float]):
    pass
    # _, mask = random_mask(image, mask_proportion_range)
    # return img_for_reg(image, mask)

def build_MNIST_meta_dataset(test: bool = False):
    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    mnist_iter = iter(torch.utils.data.DataLoader(mnist, shuffle=True))

    if test:
        n = 10_000
    else:
        n = 60_000

    md = []
    # for _ in tqdm(range(n)):
    #     img = next(mnist_iter)[0].squeeze(0)
    #     X, Y = some_function(img)
    #     m_d.append((X, Y))

    # return md

    