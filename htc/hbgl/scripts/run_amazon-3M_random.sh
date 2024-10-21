#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=amazon-3M
fi

if [ ! -f  ../data/amazon-3M/amazon-3M_train.json ] || [ ! -f  ../data/amazon-3M/amazon-3M_val.json ] || [ ! -f  ../data/amazon-3M/amazon-3M_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=42
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/amazon-3M/amazon-3M_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi


if [ ! -f $TRAIN_FILE ]; then
  python3 ../preprocess.py amazon-3M 5
fi

python3 ../run.py\
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --max_source_seq_length 256 --max_target_seq_length 5\
    --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1\
    --valid_file ../data/amazon-3M/amazon-3M_val_generated.json \
    --test_file ../data/amazon-3M/amazon-3M_test_generated.json \
    --add_vocab_file ../data/amazon-3M/amazon-3M_label_map.pkl \
    --label_smoothing 0 \
    --save_steps 4000 \
    --wandb \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 64000 --cache_dir ${CACHE_DIR}\
    --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
    --self_attention \
    --random_label_init \
    #--label_cpt_not_incr_mask_ratio --label_cpt_steps 300 --label_cpt_use_bce 

## num_warmup_steps 500
## num_training_steps 96000
## label_cpt_steps 300
