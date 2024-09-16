#!/bin/bash

for seed in 41; do
  parallel -j 3 -u "python source/run_cifar100.py --model resnet18 --epochs 20 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: DELU ReLU LeakyReLU ::: $seed
done
