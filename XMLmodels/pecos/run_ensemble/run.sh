#!/usr/bin/env bash
set -euo pipefail

data=${1:?usage: run.sh DATASET DATA_ROOT}
path=${2:?usage: run.sh DATASET DATA_ROOT}
data_dir="../${path}/${data}"

case "$data" in
  eurlex-4k|amazoncat-13k|nyt|nyt_leaves|wos|wos_leaves|rcv1|rcv1_leaves)
    models=(bert roberta xlnet)
    ens_method=softmax_average
    ;;
  wiki10-31k)
    models=(bert roberta xlnet)
    ens_method=rank_average
    ;;
  wiki-500k)
    models=(bert1 bert2 bert3)
    ens_method=sigmoid_average
    ;;
  amazon-670k)
    models=(bert1 bert2 bert3)
    ens_method=softmax_average
    ;;
  amazon-3m)
    models=(bert1 bert2 bert3)
    ens_method=rank_average
    ;;
  *)
    echo "Unknown dataset: $data" >&2
    exit 2
    ;;
esac

if [[ -e "models/${data}" ]]; then
  echo "Output already exists: models/${data}. Archive it or choose a clean checkout." >&2
  exit 1
fi

predictions=()
for model in "${models[@]}"; do
  bash ./train_and_predict.sh "$data" "$model" "$data_dir"
  predictions+=("models/${data}/${model}/Pt.npz")
done

python3 ./ensemble_evaluate.py \
  -y "${data_dir}/Y.tst.npz" \
  -p "${predictions[@]}" \
  --tags "${models[@]}" \
  --ens-method "$ens_method" \
  2>&1 | tee "models/${data}/ensemble.log"
