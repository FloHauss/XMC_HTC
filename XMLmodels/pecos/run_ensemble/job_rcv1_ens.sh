#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --time=12:00:00
#SBATCH --mem=80000
#SBATCH --job-name=rcv1
#SBATCH --gres=gpu:2
#SBATCH --dependency=singleton

# [x] sbatch params optimized

DATASET="rcv1"
PATH_TO_DATASET="htc-base"

NOW=$(date "+%Y-%m-%d %H:%M:%S")

source /home/ul/ul_student/ul_ruw26/.bashrc
conda activate xr_transformer_env

cd /home/ul/ul_student/ul_ruw26/XMC_HTC/XMLmodels/pecos/run_ensemble

echo "------- Ensemble run at $NOW for $DATASET ----------"

bash run.sh ${DATASET} ${PATH_TO_DATASET}