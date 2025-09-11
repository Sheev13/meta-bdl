#!/bin/bash

#SBATCH --job-name=prior_transfer_array
#SBATCH --output=experiments/prior_transfer/slurm_logs/%A_%a.out
#SBATCH --error=experiments/prior_transfer/slurm_logs/%A_%a.err
#SBATCH --array=0,6,12,18,24,30

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=1000

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

function_types=("gp" "sawtooth" "heaviside")
models=("mfvi" "givi" "bdnp" "lmc" "hmc" "swag")
priors=("bnn" "pretrained")

num_models=${#models[@]}
num_functions=${#function_types[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID % num_models ))
function_idx=$(( (SLURM_ARRAY_TASK_ID / num_models) % num_functions ))
pretrained_prior=$(( SLURM_ARRAY_TASK_ID / (num_models * num_functions) ))

model=${models[$model_idx]}
function=${function_types[$function_idx]}

if [[ $pretrained_prior -eq 0 ]]; then
    prior="bnn"
else
    prior="${function}_48_48"
fi

# Run your script with arguments
python experiments/prior_transfer/transfer_prior.py \
    --prior $prior \
    --model_name $model \
    --function_type $function \
    --use_gpu \
    --use_shared_test_sets