
# For Table 2 (data_set=nq320k),
# model_name = [nq-distilbert-base-v1, multi-qa-mpnet-base-dot-v1, sentence-t5-base, gtr-t5-base]
# For Table 3 (data_set=trivia),
# model_name = [multi-qa-distilbert-dot-v1, multi-qa-mpnet-base-dot-v1, sentence-t5-base, gtr-t5-base]
data_set=$1  # "nq320k"
model_name=$2 # "gtr-t5-base"
if [ -z ${data_set} ] || [ -z ${model_name} ]; then
    echo "bash run_pefa_xl.sh [data_set] [model_name]"
    exit
fi

if [ ${data_set} == "nq320k" ]; then
    knn_type="v0"
    topk_q=16
    lambda_erm=0.1
elif [ ${data_set} == "trivia" ]; then
    knn_type="v1"
    topk_q=64
    if [ ${model_name} == "multi-qa-distilbert-dot-v1" ]; then
        lambda_erm=0.5
    elif [ ${model_name} == "multi-qa-mpnet-base-dot-v1" ]; then
        lambda_erm=0.9
    elif [ ${model_name} == "sentence-t5-base" ]; then
        lambda_erm=0.7
    elif [ ${model_name} == "gtr-t5-base" ]; then
        lambda_erm=0.5
    else
        echo "For trivia, model_name=${model_name} is not support!"
    fi
else
    echo "can not set knn_type due to unknown data_set!"
fi
input_xmc_dir=./data/xmc/${data_set}
input_emb_dir=./data/embeddings/${data_set}/${model_name}
python -u pefa_xl.py ${input_xmc_dir} ${input_emb_dir} ${lambda_erm} ${topk_q} ${knn_type}
