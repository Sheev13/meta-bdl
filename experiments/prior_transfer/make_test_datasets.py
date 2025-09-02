import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
from utils.data_utils import obtain_me_a_nice_gp_dataset_please, obtain_me_a_nice_heaviside_dataset_please, obtain_me_a_nice_sawtooth_dataset_please, ctxt_trgt_split


def build_meta_dataset(num_datasets=16, n_range=[40, 100], function_type='sawtooth', x_range=[-4.0, 4.0], ctxt_range=[0.05, 0.5]):
    md = []
    assert function_type.lower() in ['sawtooth', 'gp', 'heaviside']

    if function_type.lower() == 'sawtooth':
        dataset_func = obtain_me_a_nice_sawtooth_dataset_please
        data_hypers = {'p': 0.75, 'm': 1.33, 'random_linear': True, 'x_range': [-2.0, 2.0]}
    elif function_type.lower() == 'gp':
        dataset_func = obtain_me_a_nice_gp_dataset_please
        data_hypers = {'l': 0.5, 'kernel': 'se', 'x_range': x_range}
    elif function_type.lower() == 'heaviside':
        dataset_func = obtain_me_a_nice_heaviside_dataset_please
        data_hypers = {'x_range': x_range, 'l': 1}

    for _ in range(num_datasets):
        X, y = dataset_func(n_range=n_range, **data_hypers)
        Xc, yc, Xt, yt = ctxt_trgt_split(X, y, ctxt_proportion_range=ctxt_range)
        md.append((Xc, yc, Xt, yt))
    
    return md


def main():
    PATH = str(Path(__file__).resolve().parent)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(69) # constant seed for this part only to ensure dame dataset every time
        
    for function_type in ['sawtooth', 'gp', 'heaviside']:
        test_sets = build_meta_dataset(num_datasets=16, n_range=[50, 51], function_type=function_type)

        Path(PATH + f"/shared_test_sets").mkdir(parents=True, exist_ok=True)
        torch.save(test_sets, PATH + f"/shared_test_sets/{function_type}.pt")
    

if __name__ == "__main__":
    main()