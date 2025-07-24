#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --job-name=cuda_tests
#SBATCH --output=experiments/1d_regressions/batch_logs/test_%j.out    # STDOUT (%j = Job ID)
#SBATCH --error=experiments/1d_regressions/batch_logs/test_%j.err     # STDERR

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-sandbox

# Run your script with arguments
python experiments/1d_regressions/test_py.py 