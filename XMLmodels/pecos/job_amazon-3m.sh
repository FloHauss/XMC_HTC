#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=24:00:00
#SBATCH --mem=80000
#SBATCH --job-name=amazon-3m
#SBATCH --gres=gpu:4
#SBATCH --dependency=singleton

source /home/ul/ul_student/ul_ruw26/.bashrc
conda activate xr_transformer_env

python3 -m pecos.xmc.xtransformer.train -t ./xmc-base/amazon-3m/X.trn.txt -x ./xmc-base/amazon-3m/tfidf-attnxml/X.trn.npz -y ./xmc-base/amazon-3m/Y.trn.npz -m ./trained-models/xr_model_amazon-3m

python3 -m pecos.xmc.xtransformer.predict -t ./xmc-base/amazon-3m/X.tst.txt -x ./xmc-base/amazon-3m/tfidf-attnxml/X.tst.npz -m ./trained-models/xr_model_amazon-3m -o ./predictions/xr_prediction_amazon-3m