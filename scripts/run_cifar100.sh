#!/bin/bash
mapfile -t activations < ./metadata/activations.txt

for seed in 41 42 43 44 45; do
  parallel -j 1 -u "python source/run_cifar100.py --task cifar100 --model resnet18 --epochs 50 --batch-size 128 --lr 0.001 --seed {2} --activation {1}" ::: "${activations[@]}" ::: $seed
done
