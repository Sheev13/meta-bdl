#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --job-name=learn_era5
#SBATCH --output=experiments/era5_prior_transfer/slurm_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/era5_prior_transfer/slurm_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/era5_prior_transfer/learn_prior.py \
    --codename firefly \
    --architecture 64 64 64 \
    --nonlinearity relu \
    --training_steps 250_000 \
    --learning_rate 1e-3 \
    --final_learning_rate 1e-5 \
    --use_gpu