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
RUN_NAME=${1:-rcv1}
LOG_FILE="../logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p ../logs

# Redirect all subsequent output (stdout and stderr) to the log file
# while still printing to the console so you can see it if you're attached.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting session. Logging to: $LOG_FILE"

# 4. Setup Paths
OUTPUT_DIR=../models/$RUN_NAME
CACHE_DIR=./cache
TRAIN_FILE=../data/rcv1/rcv1_train_generated_tl.json

# 5. Check for existence of raw data
if [ ! -f  ../data/rcv1/rcv1_train.json ] || [ ! -f  ../data/rcv1/rcv1_val.json ] || [ ! -f  ../data/rcv1/rcv1_test.json ] ; then
  echo "Please preprocess raw dataset first"
  exit 0
fi

# 6. Run Preprocessing if necessary
# Note: RCV1 uses depth 5 (unlike WOS which used 3)
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Generating training file..."
  python3 ../preprocess.py rcv1 5
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
      --max_source_seq_length 492 \
      --max_target_seq_length 5 \
      --per_gpu_train_batch_size 12 \
      --gradient_accumulation_steps 1 \
      --valid_file ../data/rcv1/rcv1_val_generated.json \
      --test_file ../data/rcv1/rcv1_test_generated.json \
      --add_vocab_file ../data/rcv1/rcv1_label_map.pkl \
      --label_smoothing 0 \
      --wandb \
      --learning_rate 3e-5 \
      --num_warmup_steps 500 \
      --num_training_steps 96000 \
      --save_steps 3000 \
      --cache_dir "${CACHE_DIR}" \
      --random_prob 0 \
      --keep_prob 0 \
      --soft_label \
      --seed "${current_seed}" \
      --random_label_init \
      --label_cpt ../data/rcv1/rcv1.taxonomy \
      --label_cpt_steps 100 \
      --rcv1_expand ../data/rcv1/rcv1.topics.hier.expanded \
      --label_cpt_use_bce \
      --job_id "${JOB_ID}"

  echo "Run ${run_num} finished. Results saved in ${OUTPUT_DIR}"
done

echo "All 5 runs completed successfully."