#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --job-name=prior_transfer
#SBATCH --output=experiments/prior_transfer/slurm_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/prior_transfer/slurm_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/prior_transfer/transfer_prior.py \
    --prior bnn \
    --model_name hmc \
    --function_type sawtooth \
    --use_gpu
