#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb|h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nice=100

#SBATCH --job-name=image_completions
#SBATCH --output=experiments/image_completions/slurm_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/image_completions/slurm_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/image_completions/image_completions.py \
    --codename haiti \
    --architecture 96 96 96 96 \
    --training_steps 30_000 \
    --num_samples 16 \
    --ctxt_proportion_range 0.01 0.4 \
    --use_gpu