import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from ucimlrepo import fetch_ucirepo 
from utils.data_utils import ctxt_trgt_split


def cube_split(X, y, target_fraction=0.75, cube_fraction_range=(0.15, 0.25)):
    """
    Splits dataset into context and target sets:
      - Picks cube size so that ~cube_fraction_range of points are inside.
      - All cube points go into target set.
      - Randomly sample from outside points so that target set is ~target_fraction.
    """

    N, D = X.shape

    # === Step 1: determine cube size ===
    absX = X.cpu().abs().numpy()  # symmetric cube around origin
    max_vals = absX.max(axis=0)

    # binary search cube half-width so that fraction in cube is within range
    lo, hi = 0.0, max(max_vals)
    best_width, best_frac = None, None
    for _ in range(50):  # binary search iterations
        mid = (lo + hi) / 2
        in_cube = (absX <= mid).all(axis=1)
        frac = in_cube.mean()
        if cube_fraction_range[0] <= frac <= cube_fraction_range[1]:
            best_width, best_frac = mid, frac
            break
        if frac < cube_fraction_range[0]:
            lo = mid
        else:
            hi = mid
    if best_width is None:  # fallback: pick closest
        mid = (lo + hi) / 2
        in_cube = (absX <= mid).all(axis=1)
        best_width, best_frac = mid, in_cube.mean()

    # final cube membership
    in_cube = (absX <= best_width).all(axis=1)

    # === Step 2: allocate target points ===
    idx_in_cube = np.where(in_cube)[0]
    idx_out_cube = np.where(~in_cube)[0]

    n_target_total = int(target_fraction * N)
    n_inside = len(idx_in_cube)
    n_needed_from_outside = n_target_total - n_inside

    if n_needed_from_outside < 0:
        raise ValueError("Cube fraction is too large relative to target_fraction.")

    idx_outside_sampled = np.random.choice(idx_out_cube, size=n_needed_from_outside, replace=False)

    target_idx = np.concatenate([idx_in_cube, idx_outside_sampled])
    ctxt_idx = np.setdiff1d(np.arange(N), target_idx)

    # === Step 3: create tensors ===
    ctxt = (X[ctxt_idx], y[ctxt_idx])
    trgt = (X[target_idx], y[target_idx])

    return (*ctxt, *trgt), {
        "cube_halfwidth": best_width,
        "cube_frac": best_frac,
        "target_frac": len(target_idx) / N,
        "inside_count": len(idx_in_cube),
        "outside_count_sampled": len(idx_outside_sampled),
        "outside_ctxt_proportion": 1 - (len(idx_outside_sampled) / (X.shape[0] - len(idx_in_cube)))
    }


def main():
    torch.set_default_dtype(torch.float64)

    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/data").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/data/abalone").mkdir(parents=True, exist_ok=True)

    abalone = fetch_ucirepo(id=1)
    features = abalone.data.features
    targs = abalone.data.targets

    features_m = features[features['Sex'] == 'M']
    features_f = features[features['Sex'] == 'F']
    features_i = features[features['Sex'] == 'I']

    targs_m = targs.loc[features_m.index]
    targs_f = targs.loc[features_f.index]
    targs_i = targs.loc[features_i.index]

    # drop sex as an input feature
    features_m = features_m.drop(columns=['Sex'])
    features_f = features_f.drop(columns=['Sex'])
    features_i = features_i.drop(columns=['Sex'])

    X_m = torch.tensor(features_m.values)
    X_f = torch.tensor(features_f.values)
    X_i = torch.tensor(features_i.values)

    y_m = torch.tensor(targs_m.values, dtype=torch.get_default_dtype())
    y_f = torch.tensor(targs_f.values, dtype=torch.get_default_dtype())
    y_i = torch.tensor(targs_i.values, dtype=torch.get_default_dtype())

    # normalise the fellas
    X_cat = torch.cat((X_m, X_f), dim=0)
    X_mean, X_std = X_cat.mean(0), X_cat.std(0)
    X_m = (X_m - X_mean) / X_std
    X_f = (X_f - X_mean) / X_std
    X_i = (X_i - X_mean) / X_std

    y_cat = torch.cat((y_m, y_f), dim=0)
    y_mean, y_std = y_cat.mean(0), y_cat.std(0)
    y_m = (y_m - y_mean) / y_std
    y_f = (y_f - y_mean) / y_std
    y_i = (y_i - y_mean) / y_std

    norm_consts = {}
    norm_consts['X_mean'] = X_mean
    norm_consts['X_std'] = X_std
    norm_consts['y_mean'] = y_mean
    norm_consts['y_std'] = y_std
    torch.save(norm_consts, PATH + "/data/abalone/norm_consts.pt")

    metadata = {}
    metadata['dimensionality'] = X_m.shape[-1]
    metadata['num_male'] = X_m.shape[0]
    metadata['num_female'] = X_f.shape[0]
    metadata['num_infant'] = X_i.shape[0]

    train_sets = [(X_m, y_m), (X_f, y_f)]
    torch.save(train_sets, PATH + "/data/abalone/train_sets.pt")

    test_set, split_metadata = cube_split(X_i, y_i, target_fraction=0.75, cube_fraction_range=(0.15, 0.25))
    torch.save(test_set, PATH + "/data/abalone/test_set.pt")

    with open(PATH + "/data/abalone/metadata.json", 'w') as f:
        json.dump({**metadata, **split_metadata}, f, indent=4)

if __name__ == "__main__":
    main()