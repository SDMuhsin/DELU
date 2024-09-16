#!/bin/bash

for seed in 41 42 43 44 45; do
  parallel -j 3 -u "python source/run_svhn.py --model resnet18 --epochs 10 --batch-size 64 --lr 0.001 --seed {2} --activation {1}" ::: DELU ReLU LeakyReLU ::: $seed
done
