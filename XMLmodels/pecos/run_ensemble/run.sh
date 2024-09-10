#!/bin/bash
data=$1
path=$2
data_dir="./../${path}/${data}/"

if [ ${data} == "eurlex-4k" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
elif [ ${data} == "wiki10-31k" ]; then
	models=(bert roberta xlnet)
	ens_method=rank_average
elif [ ${data} == "amazoncat-13k" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
elif [ ${data} == "wiki-500k" ]; then
	models=(bert1 bert2 bert3)
	ens_method=sigmoid_average
elif [ ${data} == "amazon-670k" ]; then
	models=(bert1 bert2 bert3)
	ens_method=softmax_average
elif [ ${data} == "amazon-3m" ]; then
	models=(bert1 bert2 bert3)
	ens_method=rank_average
elif [ ${data} == "nyt" ] || [ ${data} == "nyt_leaves" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
elif [ ${data} == "wos" ] || [ ${data} == "wos_leaves" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
elif [ ${data} == "rcv1" ] || [ ${data} == "rcv1_leaves" ]; then
	models=(bert roberta xlnet)
	ens_method=softmax_average
else
	echo Unknown dataset $1!
	exit
fi

Preds=""
Tags=""

rm -r "models/${data}" # remove existing models to ensure completely new model generation

for mm in "${models[@]}"; do
	bash ./train_and_predict.sh ${data} ${mm} ${data_dir}
	Preds="${Preds} models/${data}/${mm}/Pt.npz"
	Tags="${Tags} ${mm}"
done

Y_tst=${data_dir}/Y.tst.npz # test label matrix

python3 ./ensemble_evaluate.py \
	-y ${Y_tst} \
	-p ${Preds} \
	--tags ${Tags} \
	--ens-method ${ens_method} \
	--dataset ${data_dir}
2>&1 | tee models/${data}/ensemble.log
# |& tee models/${data}/ensemble.log
