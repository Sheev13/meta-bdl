#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --job-name=ecg
#SBATCH --output=experiments/1d_regressions/batch_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/1d_regressions/batch_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/1d_regressions/ecg_regressions.py --codename xavier