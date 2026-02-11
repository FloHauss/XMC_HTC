#!/usr/bin/env bash

# 1. Activate the Virtual Environment
if [ -d "../.venv" ]; then
  source ../.venv/bin/activate
  echo "Virtual environment activated."
else
  echo "Error: .venv not found in parent directory."
  exit 1
fi

# 2. WandB Offline Mode & Environment Setup
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# 3. Handle Run Name & Logging Path
# Defaults to amazon-670k if no argument provided
RUN_NAME=${1:-amazon-670k} 
LOG_FILE="../logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ../logs

# Redirect all subsequent output (stdout and stderr) to the log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting session. Logging to: $LOG_FILE"

# 4. Setup Paths
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/amazon-670k/amazon-670k_train_generated_tl.json

# 5. Check for existence of raw data
if [ ! -f  ../data/amazon-670k/amazon-670k_train.json ] || [ ! -f  ../data/amazon-670k/amazon-670k_val.json ] || [ ! -f  ../data/amazon-670k/amazon-670k_test.json ] ; then
  echo "Please preprocess raw dataset first"
  exit 0
fi

# 6. Run Preprocessing if necessary
# Note: Using depth 5 as per original script
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Generating training file..."
  python3 ../preprocess.py amazon-670k 5
fi

# 7. Stacked Run Loop
# Adjusted seeds to include the original 42 and subsequent seeds
SEEDS=(42 1 2 3 4)

for i in "${!SEEDS[@]}"
do
  current_seed=${SEEDS[$i]}
  run_num=$((i+1))
  JOB_ID="run_${run_num}_s${current_seed}_$(date +%H%M%S)"
  
  echo "-------------------------------------------------------"
  echo " Starting Stacked Run ${run_num}/${#SEEDS[@]} | Seed: ${current_seed}"
  echo " Date: $(date)"
  echo "-------------------------------------------------------"

  # Clear output directory for each run to avoid conflicts (as per target format)
  if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
  fi
  mkdir -p "$OUTPUT_DIR"

  # 8. Execute Training
  python3 ../run.py \
      --train_file "${TRAIN_FILE}" \
      --output_dir "${OUTPUT_DIR}" \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --do_lower_case \
      --max_source_seq_length 500 \
      --max_target_seq_length 5 \
      --per_gpu_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --valid_file ../data/amazon-670k/amazon-670k_val_generated.json \
      --test_file ../data/amazon-670k/amazon-670k_test_generated.json \
      --add_vocab_file ../data/amazon-670k/amazon-670k_label_map.pkl \
      --label_smoothing 0 \
      --wandb \
      --save_steps 3000 \
      --learning_rate 3e-5 \
      --num_warmup_steps 500 \
      --num_training_steps 96000 \
      --cache_dir "${CACHE_DIR}" \
      --random_prob 0 \
      --keep_prob 0 \
      --soft_label \
      --seed "${current_seed}" \
      --job_id "${JOB_ID}" \
      --self_attention \
      --random_label_init \
      #--label_cpt_not_incr_mask_ratio \
      #--label_cpt_steps 300 \
      #--label_cpt_use_bce

  echo "Run ${run_num} finished. Results saved in ${OUTPUT_DIR}"
done

echo "All runs completed successfully."