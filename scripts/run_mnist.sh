#!/bin/bash

mapfile -t activations < ./metadata/activations.txt

for seed in 41; do
  parallel -j 1 -u "python source/run_mnist.py --model resnet18 --epochs 5 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
