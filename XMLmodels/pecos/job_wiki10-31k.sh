#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --time=3:00:00
#SBATCH --mem=40000
#SBATCH --job-name=wiki10-31k
#SBATCH --gres=gpu:1
#SBATCH --dependency=singleton

# script params:
dataset="wiki10-31k"
# ens_models="bert roberta xlnet"
ens_models="bert"


# ---- no changes needed below ----

NOW=$(date "+%Y-%m-%d %H:%M:%S")
UUID=$(date "+%Y-%m-%d-%H-%M-%S")

source /home/ul/ul_student/ul_ruw26/.bashrc
conda activate xr_transformer_env

rm -rf trained-models/$dataset
mkdir -p ./predictions/$dataset/$UUID
mkdir -p ./results/$dataset

echo "dataset is $dataset"
echo "UUID is $UUID"

echo -e "*** Run at $NOW for ensemble of $ens_models ***\n" >>./results/$dataset/$UUID

for model in $ens_models; do
    # --- train ----
    echo "--- start training of $model ---"
    python3 -m pecos.xmc.xtransformer.train -t ./xmc-base/$dataset/X.trn.txt -x ./xmc-base/$dataset/tfidf-attnxml/X.trn.npz -y ./xmc-base/$dataset/Y.trn.npz -m ./trained-models/$dataset/$model --params-path ./params/$dataset/$model/params.json

    # --- predict ---
    echo "--- start prediction of $model ---"
    python3 -m pecos.xmc.xtransformer.predict -t ./xmc-base/$dataset/X.tst.txt -x ./xmc-base/$dataset/tfidf-attnxml/X.tst.npz -m ./trained-models/$dataset/$model -o ./predictions/$dataset/$UUID/$model

    # --- evaluate ---
    echo "--- start evaluation of $model ---"
    echo -e "\n*** results from $model ***\n" >>./results/$dataset/$UUID
    python3 -m pecos.xmc.xlinear.evaluate -y ./xmc-base/$dataset/Y.tst.npz -p ./predictions/$dataset/$UUID/$model -k 10 >>./results/$dataset/$UUID
done