#!/bin/bash

# Define constants
# Define constants
OUTPUT_DIR="saves/tmp"
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32
NUM_EPOCHS=9
LEARNING_RATE=2e-5

# Input
activation=$1
a=${2:-1}
b=${3:-1}
c=${4:-1}
d=${5:-1}
MODEL=${6:-"bert-base-uncased"}  # Accept model name from command line, default to bert-base-uncased

export a
export b
export c
export d

export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE
export NUM_EPOCHS
export LEARNING_RATE

# Run the script for the specified model, each task, and seed
for TASK in rte cola stsb mrpc sst2 qnli; do
    export MODEL TASK activation
    parallel -j 1 -u 'conda activate double_env_6;echo "Running for model: $MODEL, task: $TASK, with seed: {}"; \
        python3 ./source/run_glue.py \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL \
        --seed {} \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --job_id glue \
        --split_train n \
        --just_download n \
        --overwrite_saves n \
        --activation $activation \
        --store_best_result y \
        --a $a \
        --b $b \
        --c $c \
        --d $d \
        --task_name $TASK' ::: 41 42 43 44 45
done
