#!/bin/bash
mapfile -t activations < ./metadata/activations.txt

for seed in 41; do
  parallel -j 2 -u "python source/run_cifar100.py --model resnet18 --epochs 10 --batch-size 128 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
