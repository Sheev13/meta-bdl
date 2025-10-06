import os
import subprocess
import pandas as pd
import numpy as np
import torch
import scipy.signal as sps
from tqdm import tqdm
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]  # two levels up
sys.path.insert(0, str(root_dir))

torch.set_default_dtype(torch.float64)
PATH = str(Path(__file__).resolve().parent)
Path(PATH + "/ecg_data").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
BASE_URL = "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
DATA_DIR = "./ecg_arrhythmia_data"
LEAD_NAME = "II"
ORIGINAL_FS = 500
TARGET_FS = 50
TARGET_DURATION = 10.0
MAX_RECORDS = None  # set e.g. 100 for quick tests
# -------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. Download dataset using wget (no manual intervention)
# -------------------------------------------------------------------
if not any(fname.endswith(".csv") for _, _, files in os.walk(DATA_DIR) for fname in files):
    print("⬇️ Downloading dataset recursively with wget...")
    subprocess.run(
        [
            "wget",
            "-r", "-N", "-c", "-np",
            "-P", DATA_DIR,
            BASE_URL,
        ],
        check=True,
    )
    print("✅ Download complete.")

# The files end up in DATA_DIR/physionet.org/files/ecg-arrhythmia/1.0.0/
DOWNLOAD_DIR = os.path.join(DATA_DIR, "physionet.org/files/ecg-arrhythmia/1.0.0")

# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------
def load_csv_signal(path, lead_name):
    df = pd.read_csv(path)
    if lead_name not in df.columns:
        raise ValueError(f"Lead {lead_name} not found in {path}.")
    return df[lead_name].to_numpy(dtype=np.float32)

def downsample(signal, orig_fs, target_fs):
    n_target = int(len(signal) * target_fs / orig_fs)
    return sps.resample(signal, n_target)

def make_task(signal, fs):
    n = len(signal)
    t = np.linspace(0, n / fs, n, endpoint=False)
    X = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(signal, dtype=torch.float32).unsqueeze(1)
    return (X, Y)

# -------------------------------------------------------------------
# 3. Collect all CSV files
# -------------------------------------------------------------------
csv_files = []
for root, _, files in os.walk(DOWNLOAD_DIR):
    for f in files:
        if f.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, f))
csv_files.sort()
if MAX_RECORDS:
    csv_files = csv_files[:MAX_RECORDS]

print(f"Found {len(csv_files)} CSV files.")

# -------------------------------------------------------------------
# 4. Process into meta-dataset
# -------------------------------------------------------------------
tasks = []
target_len = int(TARGET_FS * TARGET_DURATION)

for path in tqdm(csv_files, desc="Processing ECG CSVs"):
    try:
        sig = load_csv_signal(path, LEAD_NAME)
        sig_ds = downsample(sig, ORIGINAL_FS, TARGET_FS)
        if len(sig_ds) < target_len:
            continue
        sig_ds = sig_ds[:target_len]
        sig_ds = (sig_ds - np.mean(sig_ds)) / (np.std(sig_ds) + 1e-8)
        X, Y = make_task(sig_ds, TARGET_FS)
        tasks.append((X, Y))
    except Exception as e:
        print(f"Skipping {path}: {e}")


torch.save(tasks, PATH + "/ecg_data/ecg_tasks.pt")