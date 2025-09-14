#!/bin/bash

#SBATCH --job-name=np-debug
#SBATCH --output=notebooks/slurm_logs/%j.out
#SBATCH --error=notebooks/slurm_logs/%j.err

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

# Run your script with arguments
python notebooks/debug_np.py --model bnp --use_gpu