#!/bin/bash

mapfile -t activations < ./metadata/activations.txt
a=${1:-1}  # Use first argument if provided, otherwise default to 1
b=${2:-1}  # Use second argument if provided, otherwise default to 1

export a
export b
for seed in 41 42 43 44 45; do
  parallel -j 1 -u "python source/run_svhn.py --task svhn --model resnet18 --epochs 50 --batch-size 128 --lr 0.001 --seed {2} --activation {1} --a $a --b $b" ::: "${activations[@]}" ::: $seed
done
