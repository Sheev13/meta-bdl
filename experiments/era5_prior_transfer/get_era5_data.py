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
import zipfile
import pathlib
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point

# define ctxt_trgt_split that never includes context points from within Switzerland
def swissless_ctxt_trgt_split(X, y, X_m, X_s, ctxt_proportion_range=[0.025, 0.25]):

    # Load Natural Earth country boundaries (scale 1:50m is usually fine)
    shpfilename = shpreader.natural_earth(resolution='50m',
                                        category='cultural',
                                        name='admin_0_countries')

    reader = shpreader.Reader(shpfilename)
    countries = reader.records()

    # Get Switzerland polygon(s)
    swiss_geom = None
    for country in countries:
        if country.attributes['NAME_EN'] == 'Switzerland':
            swiss_geom = country.geometry
            break

    if swiss_geom is None:
        raise RuntimeError("Could not find Switzerland geometry")

    # De-normalize longitude and latitude
    X_denorm = X * X_s + X_m
    lon, lat = X_denorm[:, 0].numpy(), X_denorm[:, 1].numpy()

    # Build shapely Points
    points = [Point(lo, la) for lo, la in zip(lon, lat)]

    # Mask: keep only points NOT inside Switzerland
    mask_outside_swiss = np.array([not swiss_geom.contains(p) for p in points])

    eligible_ctxt_indices = np.where(mask_outside_swiss)[0]

    n_total = len(X)
    n_ctxt = np.random.randint(
        int(ctxt_proportion_range[0] * n_total),
        int(ctxt_proportion_range[1] * n_total) + 1
    )

    ctxt_indices = np.random.choice(eligible_ctxt_indices, size=n_ctxt, replace=False)
    trgt_indices = np.setdiff1d(np.arange(n_total), ctxt_indices)

    Xc = torch.tensor(X[ctxt_indices])
    yc = torch.tensor(y[ctxt_indices])
    Xt = torch.tensor(X[trgt_indices])
    yt = torch.tensor(y[trgt_indices])

    return Xc, yc, Xt, yt


def main():
    PATH = str(Path(__file__).resolve().parent)
    c = cdsapi.Client()

    # Define area (Central Europe) and years
    area = [50, 5, 45, 12]  # N, W, S, E
    years = [str(y) for y in range(2010, 2020)]  # 10 years

    Path(PATH + "/data").mkdir(parents=True, exist_ok=True)
    raw_dir = PATH + "/data/era5_ceurope_raw"
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------
    # 1. Download monthly files (only time-varying vars)
    # ----------------------
    def _is_zip(path):
        try:
            with open(path, "rb") as fh:
                sig = fh.read(4)
            return sig == b"PK\x03\x04"
        except FileNotFoundError:
            return False

    def _extract_zip_return_ncs(zip_path, dest_dir):
        """Extract zip to dest_dir and return list of extracted .nc paths."""
        ncs = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".nc"):
                    out_path = pathlib.Path(dest_dir) / pathlib.Path(name).name
                    if not out_path.exists():
                        zf.extract(name, dest_dir)
                        # move to dest_dir root if nested
                        out_extracted = pathlib.Path(dest_dir) / name
                        out_extracted.rename(out_path)
                    ncs.append(str(out_path))
        return ncs

    def download_month(year, month):
        target_file = f"{raw_dir}/era5_land_{year}_{month:02d}.nc"
        if not os.path.exists(target_file):
            c.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": ["2m_temperature", "total_precipitation"],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "12:00"],
                    "area": area,
                    "grid": [0.1, 0.1],
                    "format": "netcdf",
                },
                target_file,
            )
        return target_file

    def open_month_dataset(path):
        """Open a monthly file that may be a ZIP disguised as .nc, and merge if needed."""
        if _is_zip(path):
            nc_paths = _extract_zip_return_ncs(path, raw_dir)
            if not nc_paths:
                raise RuntimeError(f"ZIP contained no .nc files: {path}")
            dsets = [xr.open_dataset(p, engine="netcdf4") for p in nc_paths]
            ds = xr.merge(dsets, compat="override", join="exact")
            return ds
        else:
            try:
                return xr.open_dataset(path, engine="netcdf4")
            except Exception:
                return xr.open_dataset(path, engine="h5netcdf")

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
        ds = open_month_dataset(f)

        # Unit conversions
        ds["tp"] = ds["tp"] * 1000      # m → mm
        ds["t2m"] = ds["t2m"] - 273.15  # K → °C

        lon = ds["longitude"].values
        lat = ds["latitude"].values
        LON, LAT = np.meshgrid(lon, lat)

        # Determine the time dimension dynamically
        time_dim = ds["t2m"].dims[0]
        time_coord = ds["t2m"][time_dim]

        # Select only 0:00 and 12:00 hours
        time_sel = time_coord[time_coord.dt.hour.isin([0, 12])]

        for t in tqdm(range(len(time_sel)), desc=f"Processing {f}", file=sys.stdout):
            temp = ds["t2m"].sel({time_dim: time_sel[t]}).values
            precip = ds["tp"].sel({time_dim: time_sel[t]}).values

            # Flatten spatial grid (LON, LAT, time-varying temp)
            X = np.stack([LON.ravel(), LAT.ravel(), temp.ravel()], axis=1)
            y = precip.ravel()

            mask = ~np.isnan(y)
            X, y = X[mask], y[mask]

            all_X.append(X)
            all_y.append(y)

        ds.close()

    # ----------------------
    # 3. Normalize inputs
    # ----------------------
    # double check normalisation is happening correctly here, e.g. what shape of X_concat is.
    X_concat = np.vstack(all_X)
    X_mean = X_concat.mean(axis=0)
    X_std = X_concat.std(axis=0)

    Y_concat = np.vstack(all_y).flatten()
    y_mean = Y_concat.mean()
    y_std = Y_concat.std()

    X_m, X_s = torch.tensor(X_mean, dtype=torch.float64), torch.tensor(X_std, dtype=torch.float64)
    torch.save([X_m, X_s], PATH + "/data/X_norm_consts.pt")
    y_m, y_s = torch.tensor(y_mean, dtype=torch.float64), torch.tensor(y_std, dtype=torch.float64)
    torch.save([y_m, y_s], PATH + "/data/y_norm_consts.pt")

    datasets = []
    c = 0
    for X, y in zip(all_X, all_y):
        X_tensor = (torch.tensor(X, dtype=torch.float64) - X_m) / X_s
        y_tensor = (torch.tensor(y, dtype=torch.float64) - y_m) / y_s
        if c == 0:
            print(f"Each full dataset has {y_tensor.numel()} datapoints.")
            c = 1
        datasets.append((X_tensor, y_tensor.unsqueeze(-1)))

    full_test_sets = datasets[-16:]
    test_sets = [ctxt_trgt_split(*dataset, ctxt_proportion_range=[0.025, 0.25]) for dataset in full_test_sets]
    swissless_test_sets = [swissless_ctxt_trgt_split(*dataset, X_m, X_s, ctxt_proportion_range=[0.025, 0.25]) for dataset in full_test_sets]
    train_sets = datasets[:-16]
    
    torch.save(test_sets, PATH + "/data/test_sets.pt")
    torch.save(swissless_test_sets, PATH + "/data/swissless_test_sets.pt")
    torch.save(train_sets, PATH + "/data/train_sets.pt")


if __name__ == "__main__":
    main()