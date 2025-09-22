import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import json
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
from utils.data_utils import ctxt_trgt_split

# Create a single generator and reuse it
N_BITS = 100
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

def smiles_to_morgan(smiles_list):
    """
    Convert a list of SMILES into a NumPy array of Morgan fingerprints.
    Returns an array of shape (len(smiles_list), N_BITS).
    """
    X = np.zeros((len(smiles_list), N_BITS))
    
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            # Convert bitstring directly to 0/1 array
            bitstring = fp.ToBitString()
            X[i, :] = np.frombuffer(bitstring.encode('ascii'), dtype=np.uint8) - ord('0')
        # if mol is None, row remains zeros
    return X

def main():
    torch.set_default_dtype(torch.float64)
    PATH = str(Path(__file__).resolve().parent)
    Path(PATH + "/data").mkdir(parents=True, exist_ok=True)
    Path(PATH + "/data/qm8").mkdir(parents=True, exist_ok=True)

    print("downloading QM8 from the world wide web.")
    tasks, datasets, transformers = dc.molnet.load_qm8(featurizer="Raw")
    train_dataset, valid_dataset, test_dataset = datasets

    print("Concatentating pre-split train/test/val sets")
    smiles_list = train_dataset.ids.tolist() + valid_dataset.ids.tolist() + test_dataset.ids.tolist()
    labels = np.vstack([
        np.vstack(train_dataset.y),
        np.vstack(valid_dataset.y),
        np.vstack(test_dataset.y)
    ]).astype(np.float64)

    excitation_tasks = [t for t in tasks if t.startswith('E')]
    print("Excitation tasks:", excitation_tasks)

    print("Splitting tasks via excitation level number.")
    idxs = [tasks.index(t) for t in excitation_tasks]
    labels = labels[:, idxs]

    print("Deleting rows with missing targets.")
    valid_rows = ~np.isnan(labels).any(axis=1)
    labels = labels[valid_rows]
    smiles_list = [s for i, s in enumerate(smiles_list) if valid_rows[i]]

    print("Converting SMILES to length-100 binary vectors.")
    X_np = smiles_to_morgan(smiles_list)
    X = torch.tensor(X_np, dtype=torch.get_default_dtype())
    Y = torch.tensor(labels)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("Performing train/test dataset splits.)")
    test_task_names = ['E2-CAM', 'E1-CC2', 'E2-CC2']
    train_task_names = [t for t in excitation_tasks if t not in test_task_names]

    train_idxs = [excitation_tasks.index(t) for t in train_task_names]
    test_idxs = [excitation_tasks.index(t) for t in test_task_names]

    Y_train = Y[:, train_idxs]  # shape: num_molecules x 5
    Y_test = Y[:, test_idxs]    # shape: num_molecules x 3

    train_sets = [(X, Y_train[:,i:i+1]) for i in range(5)]
    torch.save(train_sets, PATH + "/data/qm8/train_sets.pt")
    test_sets = [ctxt_trgt_split(X, Y_test[:,i:i+1], ctxt_proportion=0.15) for i in range(3)]
    torch.save(test_sets, PATH + "/data/qm8/test_sets.pt")

    # make artificial normalisation constants as data is already in nice range.
    norm_consts = {}
    norm_consts['X_mean'] = 0.0
    norm_consts['X_std'] = 1.0
    norm_consts['y_mean'] = 0.0
    norm_consts['y_std'] = 1.0
    torch.save(norm_consts, PATH + "/data/qm8/norm_consts.pt")

    metadata = {}
    metadata['dimensionality'] = X.shape[-1]
    metadata['num_datapoints'] = X.shape[0]
    metadata['num_tasks'] = Y.shape[-1]

    with open(PATH + "/data/qm8/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Yeeeep looking like we're all finished up here my boy.")

if __name__ == "__main__":
    main()