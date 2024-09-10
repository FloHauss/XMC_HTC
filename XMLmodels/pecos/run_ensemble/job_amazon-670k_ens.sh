#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --time=10:00:00
#SBATCH --mem=150000
#SBATCH --job-name=amazon-670k
#SBATCH --gres=gpu:2
#SBATCH --dependency=singleton

# [X] sbatch params optimized

DATASET="amazon-670k"
PATH_TO_DATASET="xmc-base"

NOW=$(date "+%Y-%m-%d %H:%M:%S")

source /home/ul/ul_student/ul_ruw26/.bashrc
conda activate xr_transformer_env

cd /home/ul/ul_student/ul_ruw26/XMC_HTC/XMLmodels/pecos/run_ensemble

echo "------- Ensemble run at $NOW for $DATASET ----------"

bash run.sh ${DATASET} ${PATH_TO_DATASET}