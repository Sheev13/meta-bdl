#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=64G
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --nice=1000

#SBATCH --job-name=image_completions
#SBATCH --output=experiments/image_completions/slurm_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/image_completions/slurm_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/image_completions/image_completions.py \
    --codename cold_avi \
    --architecture 64 64 64 64 \
    --training_steps 250_000 \
    --num_samples 32 \
    --ctxt_proportion_range 0.01 0.99 \
    --loss_function avi \
    --learning_rate 1e-4 \
    --final_learning_rate 1e-6 \
    --within_task_batch_size 128 \
    --use_gpu