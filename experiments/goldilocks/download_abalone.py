import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import argparse
import json
from ucimlrepo import fetch_ucirepo 
from utils.data_utils import ctxt_trgt_split


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
    with open(PATH + "/data/abalone/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    train_sets = [(X_m, y_m), (X_f, y_f)]
    torch.save(train_sets, PATH + "/data/abalone/train_sets.pt")

    test_set = ctxt_trgt_split(X_i, y_i, ctxt_proportion=0.25)
    torch.save(test_set, PATH + "/data/abalone/test_set.pt")

if __name__ == "__main__":
    main()