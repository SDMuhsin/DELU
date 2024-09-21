#!/bin/bash

mapfile -t activations < ./metadata/activations.txt

for seed in 41; do
  #parallel -j 1 -u "python source/run_cinic10.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
  python source/run_cinic10.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.001 --seed 41 --activation ReLU
done
