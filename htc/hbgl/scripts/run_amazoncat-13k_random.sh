#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=amazoncat-13k
fi

if [ ! -f  ../data/amazoncat-13k/amazoncat-13k_train.json ] || [ ! -f  ../data/amazoncat-13k/amazoncat-13k_val.json ] || [ ! -f  ../data/amazoncat-13k/amazoncat-13k_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=42
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/amazoncat-13k/amazoncat-13k_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi


if [ ! -f $TRAIN_FILE ]; then
  python3 ../preprocess.py amazoncat-13k
fi

python3 ../run.py\
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --max_source_seq_length 500 --max_target_seq_length 4\
    --per_gpu_train_batch_size 12 --gradient_accumulation_steps 1\
    --valid_file ../data/amazoncat-13k/amazoncat-13k_val_generated.json \
    --test_file ../data/amazoncat-13k/amazoncat-13k_test_generated.json \
    --add_vocab_file ../data/amazoncat-13k/amazoncat-13k_label_map.pkl \
    --label_smoothing 0\
    --wandb \
    --save_steps 3000 \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 96000 --cache_dir ${CACHE_DIR}\
    --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
    --self_attention \
    --ignore_meta_label \
    --label_cpt ../data/amazoncat-13k/amazoncat-13k.taxonomy --label_cpt_steps 100 --label_cpt_use_bce
    #--random_label_init

## num_warmup_steps 500
## num_training_steps 96000
## label_cpt_steps 300
