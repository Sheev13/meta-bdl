#!/bin/bash

#SBATCH --job-name=era5_dwnld
#SBATCH --output=experiments/era5_prior_transfer/slurm_logs/%j.out
#SBATCH --error=experiments/era5_prior_transfer/slurm_logs/%j.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --nice=1000

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/era5_prior_transfer/get_era5_data.py