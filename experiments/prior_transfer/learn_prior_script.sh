#!/bin/bash

#SBATCH --job-name=prior_learning
#SBATCH --output=experiments/prior_transfer/slurm_logs/%A_%a.out
#SBATCH --error=experiments/prior_transfer/slurm_logs/%A_%a.err
#SBATCH --array=0

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

function_types=("gp" "sawtooth" "heaviside")
function=${function_types[$SLURM_ARRAY_TASK_ID]}

# Run your script with arguments
python experiments/prior_transfer/learn_prior.py \
    --function_type $function \
    --use_gpu