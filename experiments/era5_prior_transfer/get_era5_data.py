import cdsapi
import xarray as xr
import numpy as np
import torch
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))
import os
from tqdm import tqdm
from utils.data_utils import ctxt_trgt_split



def main():
    PATH = str(Path(__file__).resolve().parent)
    c = cdsapi.Client()

    # Define area (Central Europe) and years
    area = [55, 10, 45, 20]  # N, W, S, E
    years = [str(y) for y in range(2010, 2020)]  # 10 years
    variables = ["total_precipitation", "2m_temperature", "orography"]

    Path(PATH + "/data").mkdir(parents=True, exist_ok=True)
    raw_dir = PATH + "/data/era5_ceurope_raw"
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------
    # 1. Download monthly files
    # ----------------------
    def download_month(year, month):
        target_file = f"{raw_dir}/era5_land_{year}_{month:02d}.nc"
        if not os.path.exists(target_file):
            c.retrieve(
                "reanalysis-era5-land",
                {"variable": ["2m_temperature", "total_precipitation"],
                 "year": str(year),
                 "month": f"{month:02d}",
                 "day": [f"{d:02d}" for d in range(1, 32)],
                 "time": ["00:00", "12:00"],
                 "area": [55, 5, 45, 15],  # North, West, South, East
                 "grid": [0.2, 0.2],     # coarser resolution
                 "format": "netcdf",
                },
                target_file
            )
                
        return target_file

    files = []
    for y in years:
        for m in range(1, 13):
            files.append(download_month(y, m))

    # ----------------------
    # 2. Preprocess: extract datasets
    # ----------------------
    all_X = []
    all_y = []

    for f in files:
        ds = xr.open_dataset(f)

        # Unit conversions
        ds["tp"] = ds["tp"] * 1000      # m → mm
        ds["t2m"] = ds["t2m"] - 273.15  # K → °C

        lon = ds["longitude"].values
        lat = ds["latitude"].values
        oro = ds["orography"].values  # static [lat, lon]
        LON, LAT = np.meshgrid(lon, lat)

        # Only keep 0:00 and 12:00 hours
        time_sel = ds.time[ds.time.dt.hour.isin([0, 12])]

        for t in tqdm(range(len(time_sel)), desc=f"Processing {f}"):
            temp = ds["t2m"].sel(time=time_sel[t]).values
            precip = ds["tp"].sel(time=time_sel[t]).values

            # Flatten spatial grid
            X = np.stack([LON.ravel(), LAT.ravel(), oro.ravel(), temp.ravel()], axis=1)
            y = precip.ravel()

            # Mask invalids
            mask = ~np.isnan(y)
            X, y = X[mask], y[mask]

            all_X.append(X)
            all_y.append(y)

        ds.close()

    # ----------------------
    # 3. Normalize inputs (global mean/std across all datasets)
    # ----------------------
    X_concat = np.vstack(all_X)  # [N_total, 4]
    mean = X_concat.mean(axis=0)
    std = X_concat.std(axis=0)

    m, s = torch.tensor(mean, dtype=torch.float64), torch.tensor(std, dtype=torch.float64)
    torch.save([m, s], PATH + "/data/norm_consts.pt")

    datasets = []
    c = 0
    for X, y in zip(all_X, all_y):
        X_norm = (X - mean) / std
        X_tensor = torch.tensor(X_norm, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.float64)
        if c == 0:
            print(f"Each full dataset has {y_tensor.numel()} datapoints.")
            c = 1
        datasets.append((X_tensor, y_tensor.unsqueeze(-1)))

    # do context/target split, also train/test split.

    full_test_sets = datasets[-16:]
    test_sets = [ctxt_trgt_split(*dataset, ctxt_proportion_range=[0.025, 0.25]) for dataset in full_test_sets]
    train_sets = datasets[:-16]
    
    torch.save(test_sets, PATH + "/data/test_sets.pt")
    torch.save(train_sets, PATH + "/data/train_sets.pt")

if __name__ == "__main__":
    main()