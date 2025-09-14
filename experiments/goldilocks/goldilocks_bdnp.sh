#!/bin/bash

#SBATCH --job-name=goldilocks
#SBATCH --output=experiments/goldilocks/slurm_logs/%A_%a.out
#SBATCH --error=experiments/goldilocks/slurm_logs/%A_%a.err
#SBATCH --array=0-43

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment






prior_settings=11
seeds=("21" "42" "69" "420")
num_seeds=${#seeds[@]}

params=($(awk -v start=0.0 -v end=1.0 -v n=$prior_settings \
  'BEGIN { for (i=0; i<n; i++) printf "%.6f ", start + i*(end-start)/(n-1) }'))

task_id=$SLURM_ARRAY_TASK_ID
param_index=$((task_id % prior_settings))
seed_index=$((task_id / prior_settings))

param=${params[$param_index]}
seed=${seeds[$seed_index]}

# Run your script with arguments
python experiments/goldilocks/goldilocks.py \
    --model 'bdnp' \
    --prior_trainability $param \
    --seed $seed \
    --use_gpu \