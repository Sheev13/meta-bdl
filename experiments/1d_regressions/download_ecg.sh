#!/bin/bash

#SBATCH --job-name=ecg_dwnld
#SBATCH --output=experiments/1d_regressions/batch_logs/%j.out
#SBATCH --error=experiments/1d_regressions/batch_logs/%j.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --nice=1000

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/1d_regressions/download_ecg.py