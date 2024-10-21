#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=wiki10
fi

if [ ! -f  ../data/wiki10/wiki10_train.json ] || [ ! -f  ../data/wiki10/wiki10_val.json ] || [ ! -f  ../data/wiki10/wiki10_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=42
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/wiki10/wiki10_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi


if [ ! -f $TRAIN_FILE ]; then
  python3 ../preprocess.py wiki10
fi

python3 ../run.py\
	--train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
	--model_type bert --model_name_or_path bert-base-uncased \
 	--do_lower_case --max_source_seq_length 507 --max_target_seq_length 5\
	--per_gpu_train_batch_size 1 --gradient_accumulation_steps 1\
  --valid_file ../data/wiki10/wiki10_val_generated.json \
  --test_file ../data/wiki10/wiki10_test_generated.json \
  --add_vocab_file ../data/wiki10/wiki10_label_map.pkl \
  --label_smoothing 0 \
  --wandb \
  --learning_rate 3e-5 --num_warmup_steps 1 --num_training_steps 16 --cache_dir ${CACHE_DIR}\
  --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
  --label_cpt_not_incr_mask_ratio --label_cpt_steps 3 --label_cpt_lr 1e-4 \
  --self_attention \
  --label_cpt ../data/wiki10/wiki10.taxonomy --label_cpt_use_bce \
  #--random_label_init \

## num_warmup_steps 500
## num_training_steps 96000
## label_cpt_steps 300
