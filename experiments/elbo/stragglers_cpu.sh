#!/bin/bash

#SBATCH --job-name=elbo_experiment_job_array
#SBATCH --output=experiments/elbo/slurm_logs/%A_%a.out
#SBATCH --error=experiments/elbo/slurm_logs/%A_%a.err
#SBATCH --array=4,12,20,28

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --nice=1000

#SBATCH --mail-user=thomas.rochussen@helmholtz-munich.de
#SBATCH --mail-type=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdnp-environment

codenames=("mfvi_final" "ucvi_final" "lcvi_final" "fcvi_final" "givi_final" "bdnp_final" "meta_bdnp_final" "mc_final")
models=("mfvi" "ucvi" "lcvi" "fcvi" "givi" "bdnp" "meta_bdnp" "mc")
seeds=("21" "42" "69" "420")

num_models=${#models[@]}
model_idx=$(( SLURM_ARRAY_TASK_ID % num_models ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / num_models ))

codename=${codenames[$model_idx]}
model=${models[$model_idx]}
seed=${seeds[$seed_idx]}

# Run your script with arguments
python experiments/elbo/elbo.py \
    --codename $codename \
    --model_name $model \
    --seed $seed \
    --scale_prior \
    --learning_rate 1e-2 \
    --final_learning_rate 5e-4 \
    --use_shared_dataset