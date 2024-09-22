#!/bin/bash

mapfile -t activations < ./metadata/activations.txt

for seed in 41; do
  parallel -j 1 -u "python source/run_cifar10.py --task cifar10 --model resnet18 --epochs 1 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
