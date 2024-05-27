#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --time=24:00:00
#SBATCH --mem=150000
#SBATCH --job-name=amazoncat-13k
#SBATCH --gres=gpu:2
#SBATCH --dependency=singleton

source /home/ul/ul_student/ul_ruw26/.bashrc
conda activate xr_transformer_env
rm trained-models/xr_model_amazoncat-13k

NOW=$(date "+%Y-%m-%d %H:%M:%S")
UID=$(date "+%Y-%m-%d-%H:%M")

# --- train ---
python3 -m pecos.xmc.xtransformer.train -t ./xmc-base/amazoncat-13k/X.trn.txt -x ./xmc-base/amazoncat-13k/tfidf-attnxml/X.trn.npz -y ./xmc-base/amazoncat-13k/Y.trn.npz -m ./trained-models/xr_model_amazoncat-13k

# --- predict ---
python3 -m pecos.xmc.xtransformer.predict -t ./xmc-base/amazoncat-13k/X.tst.txt -x ./xmc-base/amazoncat-13k/tfidf-attnxml/X.tst.npz -m ./trained-models/xr_model_amazoncat-13k -o ./predictions/amazoncat-13k/$UID

# --- evaluate ---
NOW=$(date "+%Y-%m-%d %H:%M:%S")
echo -e "\n\n*** Run at $NOW ***\n" >> ./results/xr_result_amazoncat-13k
python3 -m pecos.xmc.xlinear.evaluate -y ./xmc-base/amazoncat-13k/Y.tst.npz -p ./predictions/amazoncat-13k/$UID -k 10 >> ./results/xr_result_amazoncat-13k