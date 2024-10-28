#!/bin/bash
mapfile -t activations < ./metadata/approx_activations.txt

a=${1:-1}  # Use first argument if provided, otherwise default to 1
b=${2:-1}  # Use second argument if provided, otherwise default to 1
c=${3:-1}  # Use second argument if provided, otherwise default to 1
d=${4:-1}  # Use second argument if provided, otherwise default to 1

export a
export b
export c
export d

echo $a $b

for segments in 4 8 16; do
	parallel -j 1 -u "conda activate double_env_6;conda activate double_env_6;python source/approx_run_kmnist.py --task kmnist --model resnet18 --epochs 20 --progressive_epochs 8 --batch-size 128 --lr 0.001 --seed 41 --pwl_segments {2} --activation {1} --a $a --b $b --c $c --d $d" ::: "${activations[@]}" ::: $segments
done
