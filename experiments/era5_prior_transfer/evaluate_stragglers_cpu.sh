#!/bin/bash

#SBATCH --job-name=era5_transfer
#SBATCH --output=experiments/era5_prior_transfer/slurm_logs/%A_%a.out
#SBATCH --error=experiments/era5_prior_transfer/slurm_logs/%A_%a.err
#SBATCH --array=3,9

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=512G
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --nice=100

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

models=("mfvi" "givi" "bdnp" "lmc" "hmc" "swag")
priors=("bnn" "echidna")

num_models=${#models[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID % num_models ))
prior_idx=$(( SLURM_ARRAY_TASK_ID / num_models ))

model=${models[$model_idx]}
prior=${priors[$prior_idx]}

# Run your script with arguments
python experiments/era5_prior_transfer/transfer_prior.py \
    --prior $prior \
    --model_name $model \