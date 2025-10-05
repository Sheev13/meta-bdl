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

#SBATCH --job-name=1d_regressions_
#SBATCH --output=experiments/1d_regressions/batch_logs/%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/1d_regressions/batch_logs/%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/1d_regressions/1d_regressions.py \
    --codename thomas \
    --function_type heaviside \
    --trainable_likelihood_noise \
    --init_likelihood_noise 0.1 \
    --nonlinearity tanh \
    --architecture 48 48 \
    --training_steps 30_000 \
    --loss_function pp-avi \
    --num_samples 32 \
    --release_prior_at_step 0 \
    --learning_rate 5e-3 \
    --final_learning_rate 1e-5 \
    --train_new_model \
    --use_gpu