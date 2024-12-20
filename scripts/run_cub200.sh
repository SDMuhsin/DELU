#!/bin/bash
mapfile -t activations < ./metadata/activations.txt

a=${1:-1}  # Use first argument if provided, otherwise default to 1
b=${2:-1}  # Use second argument if provided, otherwise default to 1

export a
export b

echo $a $b

for seed in 41 42 43 44 45; do
  parallel -j 1 -u "conda activate double_env_6;conda activate double_env_6;python source/run_cub200.py --task cub200 --model resnet18 --epochs 10 --batch-size 128 --lr 0.001 --seed {2} --activation {1} --a $a --b $b" ::: "${activations[@]}" ::: $seed
done
