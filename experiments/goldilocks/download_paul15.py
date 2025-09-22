import scanpy as sc
import torch
import numpy as np
import sys
from pathlib import Path
import json
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
from utils.data_utils import ctxt_trgt_split

def main():
    torch.set_default_dtype(torch.float64)
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/data").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/data/paul15").mkdir(parents=True, exist_ok=True)

    print("Downloading Paul 2015 mouse bone marrow dataset via Scanpy.")
    adata = sc.datasets.paul15()

    sc.pp.scale(adata) 
    sc.tl.pca(adata, n_comps=100) # do PCA to reduce 3400 genes to top 100 eigen genes
    X_np = adata.obsm["X_pca"] 
    if not isinstance(X_np, np.ndarray): 
        X_np = X_np.toarray()
    X = torch.tensor(X_np.copy(), dtype=torch.get_default_dtype())
    X = (X - X.mean(0)) / X.std(0)

    print("Computing pseudotimes.")
    if "dpt_pseudotime" not in adata.obs.columns:
        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata, n_dcs=10)
    y = torch.tensor(adata.obs['dpt_pseudotime'].to_numpy(), dtype=torch.get_default_dtype())
    y_mean, y_std = y.mean(0), y.std(0)
    y = (y - y_mean) / y_std

    # Stage labels (categorical, 19 stages)
    stages = adata.obs['paul15_clusters'].to_numpy()
    unique_stages = np.unique(stages)
    print("Found {} unique stages: {}".format(len(unique_stages), unique_stages))

    # Split into 19 subdatasets
    subdatasets = []
    for stage in unique_stages:
        mask = stages == stage
        X_stage = X[mask]
        y_stage = y[mask].unsqueeze(-1)  # keep as column vector
        subdatasets.append((X_stage, y_stage))

    # Metadata
    metadata = {
        "dimensionality": X.shape[-1],
        "num_datapoints": X.shape[0],
        "num_subdatasets": len(subdatasets),
        "subdataset_sizes": [len(sd[0]) for sd in subdatasets],
        "stage_names": unique_stages.tolist(),
    }

    with open(PATH + "/data/paul15/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Finished preprocessing Paul 2015 dataset.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    norm_consts = {}
    norm_consts['X_mean'] = 0.0
    norm_consts['X_std'] = 1.0
    norm_consts['y_mean'] = y_mean
    norm_consts['y_std'] = y_std
    torch.save(norm_consts, PATH + "/data/paul15/norm_consts.pt")

    train_sets = [subdatasets[i] for i in range(0, len(subdatasets), 2)]
    print("training set sizes: ", [len(sd[0]) for sd in train_sets])
    test_sets = [ctxt_trgt_split(*subdatasets[i], ctxt_proportion=0.2) for i in range(1, len(subdatasets), 2)]
    torch.save(train_sets, PATH + "/data/paul15/train_sets.pt")
    torch.save(test_sets, PATH + "/data/paul15/test_sets.pt")

if __name__ == "__main__":
    main()