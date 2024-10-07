#!/bin/bash

mapfile -t activations < ./metadata/activations.txt
a=${1:-1}  # Use first argument if provided, otherwise default to 1
b=${2:-1}  # Use second argument if provided, otherwise default to 1
c=${3:-1}  # Use second argument if provided, otherwise default to 1
d=${4:-1}  # Use second argument if provided, otherwise default to 1

export a
export b
export c
export d

for seed in 41 42 43 44 45; do
  parallel -j 1 -u "conda activate double_env_6;python source/run_cifar10.py --task cifar10 --model resnet50 --epochs 10 --batch-size 64 --lr 0.001 --seed {2} --activation {1} --a $a --b $b --c $c --d $d" ::: "${activations[@]}" ::: $seed
done
