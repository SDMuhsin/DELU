#!/bin/bash

mapfile -t activations < ./metadata/activations.txt

for seed in 41 42 43 44 45; do
  parallel -j 1 -u "python source/run_cifar10.py --task cifar10 --model resnet18 --epochs 10 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
