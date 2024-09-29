#!/bin/bash

# Define constants
OUTPUT_DIR="saves/tmp"
MODELS=("distilbert/distilbert-base-cased" "albert/albert-base-v1") #"albert/albert-base-v1") #("albert/albert-base-v1" "squeezebert/squeezebert-uncased" "facebook/bart-base" "bert-base-uncased" "google-t5/t5-base")  # Add your model names here
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32
NUM_EPOCHS=9
LEARNING_RATE=2e-5

#Input
activation=$1
a=${2:-1}  # Use first argument if provided, otherwise default to 1
b=${3:-1}  # Use second argument if provided, otherwise default to 1
c=${4:-1}  # Use second argument if provided, otherwise default to 1
d=${5:-1}  # Use second argument if provided, otherwise default to 1

export a
export b
export c
export d

export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE
export NUM_EPOCHS
export LEARNING_RATE
# Run the script for each model, task, and seed
for MODEL in "${MODELS[@]}"; do
    for TASK in rte cola stsb mrpc sst2 qnli qqp mnli; do


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
done
