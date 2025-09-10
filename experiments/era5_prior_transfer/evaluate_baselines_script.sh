#!/bin/bash

#SBATCH --job-name=prior_transfer_array
#SBATCH --output=experiments/prior_transfer/slurm_logs/%A_%a.out
#SBATCH --error=experiments/prior_transfer/slurm_logs/%A_%a.err
#SBATCH --array=0-11

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

models=("mfvi" "givi" "bdnp" "lmc" "hmc" "swag")
priors=("bnn" "aardvark")

num_models=${#models[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID % num_models ))
prior_idx=$(( SLURM_ARRAY_TASK_ID / num_models ))

model=${models[$model_idx]}
prior=${priors[$prior_idx]}

# Run your script with arguments
python experiments/era5_prior_transfer/transfer_prior.py \
    --prior $prior \
    --model_name $model \
    --use_gpu \