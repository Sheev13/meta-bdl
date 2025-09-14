#!/bin/bash

#SBATCH --job-name=goldilocks
#SBATCH --output=experiments/goldilocks/slurm_logs/%A_%a.out
#SBATCH --error=experiments/goldilocks/slurm_logs/%A_%a.err
#SBATCH --array=0-19

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

models=("mfvi" "givi" "np" "bnp" "ar-tnp")
seeds=("21" "42" "69" "420")

num_models=${#models[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID % num_models ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / num_models ))

model=${models[$model_idx]}
seed=${seeds[$seed_idx]}

# Run your script with arguments
python experiments/goldilocks/goldilocks.py \
    --model $model \
    --seed $seed \
    --use_gpu \