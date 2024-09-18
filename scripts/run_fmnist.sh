#!/bin/bash

mapfile -t activations < ./metadata/activations.txt

for seed in 41 42 43 44 45; do
  parallel -j 3 -u "python source/run_fmnist.py --model resnet18 --epochs 10 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done