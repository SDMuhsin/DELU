#!/bin/bash

# Iterate a and b from 0.1 to 2.0 in steps of 0.1
for a in $(seq 0.1 0.1 2.0)
do
  for b in $(seq 0.1 0.1 2.0)
  do
    echo "Running with a=$a, b=$b"
    python source/run_cifar100.py --model resnet18 --epochs 10 --batch-size 128 --lr 0.001 --seed 41 --activation DELU --a "$a" --b "$b"
  done
done


