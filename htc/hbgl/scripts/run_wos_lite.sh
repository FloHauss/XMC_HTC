#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=wos_lite
fi

if [ ! -f  ../data/wos_lite/wos_lite_train.json ] || [ ! -f  ../data/wos_lite/wos_lite_val.json ] || [ ! -f  ../data/wos_lite/wos_lite_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=42
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/wos_lite/wos_lite_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi


if [ ! -f $TRAIN_FILE ]; then
  python3 ../preprocess.py wos_lite 2
fi

python3 ../run.py\
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --max_source_seq_length 509 --max_target_seq_length 2\
    --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1\
    --valid_file ../data/wos_lite/wos_lite_val_generated.json \
    --test_file ../data/wos_lite/wos_lite_test_generated.json \
    --add_vocab_file ../data/wos_lite/wos_lite_label_map.pkl \
    --label_smoothing 0 \
    --wandb \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 96000 --cache_dir ${CACHE_DIR}\
    --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
    --self_attention \
    --label_cpt ../data/wos_lite/wos_lite.taxonomy --label_cpt_not_incr_mask_ratio --label_cpt_steps 300 --label_cpt_use_bce 
    # --random_label_init \

## num_warmup_steps 500
## num_training_steps 96000
## label_cpt_steps 300