#!/bin/bash
mapfile -t activations < ./metadata/activations.txt

for seed in 41; do
  parallel -j 1 -u "python source/run_cifar100.py --model resnet18 --epochs 3 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
