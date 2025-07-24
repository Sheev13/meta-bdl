#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --mem=64G
#SBATCH --constraint="[a100_80gb|h100_80gb]"
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --job-name=1d_sawtooth_regressions_
#SBATCH --output=experiments/1d_regressions/batch_logs/sawtooth_%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/1d_regressions/batch_logs/sawtooth_%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

# Run your script with arguments
python experiments/1d_regressions/1d_regressions.py --architecture 500 500 500 --learning_rate 1e-4 --final_learning_rate 1e-5 --train_new_model --use_gpu