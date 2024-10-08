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

echo $a $b

for seed in 41; do
  parallel -j 1 -u "conda activate double_env_6;conda activate double_env_6;python source/inspect_kmnist_fadelu_variation.py --task kmnist --model resnet18 --epochs 10 --batch-size 128 --lr 0.001 --seed {2} --activation {1} --a $a --b $b --c $c --d $d" ::: FADELU ::: $seed
done
