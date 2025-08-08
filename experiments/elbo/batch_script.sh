#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint="[a100_80gb|h100_80gb]"
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --job-name=elbo_experiment
#SBATCH --output=experiments/elbo/batch_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/elbo/batch_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/elbo/elbo.py --codename albert --model_name mfvi --dataset bnn --scale_prior --use_gpu