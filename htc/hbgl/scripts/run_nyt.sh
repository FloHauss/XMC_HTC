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
# This prevents the script from hanging on network calls or VS Code disconnects
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# 3. Handle Run Name & Logging Path
# Default to 'nyt' if no argument provided
RUN_NAME=${1:-nyt}
LOG_FILE="../logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ../logs

# Redirect all subsequent output (stdout and stderr) to the log file
# while still printing to the console so you can see it if you're attached.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting session. Logging to: $LOG_FILE"

# 4. Setup Paths
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/nyt/nyt_train_generated_tl.json

# 5. Check for existence of raw data
if [ ! -f  ../data/nyt/nyt_train.json ] || [ ! -f  ../data/nyt/nyt_val.json ] || [ ! -f  ../data/nyt/nyt_test.json ] ; then
  echo "Please preprocess raw dataset first"
  exit 0
fi

# 6. Run Preprocessing if necessary
# Note: NYT uses depth 9 (from your original script)
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Generating training file..."
  python3 ../preprocess.py nyt 9
fi

# 7. Stacked Run Loop
SEEDS=(42 1 2 3 4)

for i in "${!SEEDS[@]}"
do
  current_seed=${SEEDS[$i]}
  run_num=$((i+1))
  JOB_ID="run_${run_num}_s${current_seed}"
  
  echo "-------------------------------------------------------"
  echo " Starting Stacked Run ${run_num}/5 | Seed: ${current_seed}"
  echo " Date: $(date)"
  echo "-------------------------------------------------------"

  # Clean the directory inside the loop so the model starts fresh
  # Note: The original script exited if dir existed; this version overwrites for the loop.
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
      --max_source_seq_length 472 \
      --max_target_seq_length 9 \
      --per_gpu_train_batch_size 12 \
      --gradient_accumulation_steps 1 \
      --valid_file ../data/nyt/nyt_val_generated.json \
      --test_file ../data/nyt/nyt_test_generated.json \
      --add_vocab_file ../data/nyt/nyt_label_map.pkl \
      --label_smoothing 0 \
      --wandb \
      --learning_rate 3e-5 \
      --num_warmup_steps 500 \
      --num_training_steps 96000 \
      --cache_dir "${CACHE_DIR}" \
      --random_prob 0 \
      --keep_prob 0 \
      --soft_label \
      --seed "${current_seed}" \
      --label_cpt ../data/nyt/nyt.taxonomy \
      --label_cpt_use_bce \
      --self_attention \
      --nyt_only_last_label_init \
      --label_cpt_not_incr_mask_ratio \
      --label_cpt_steps 1000 \
      --label_cpt_lr 1e-4 \
      --job_id "${JOB_ID}"

  echo "Run ${run_num} finished. Results saved in ${OUTPUT_DIR}"
done

echo "All 5 runs completed successfully."