#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=$1
DATA_ROOT=$2

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-camerawarp"
    --model_type "camerawarp"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "$OUTPUT_DIR"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root ${DATA_ROOT}
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    --gradient_checkpointing
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 500 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_steps 500  # should be multiple of checkpointing_steps
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config_a100x4.yaml train.py  \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    --i_log 50 \
    --i_print 50
