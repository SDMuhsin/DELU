#!/bin/bash
activations=(
    "ReLU"
    "LeakyReLU"
    "ELU"
    "SELU"
    "GELU"
    "Tanh"
    "Sigmoid"
    "Hardswish"
    "Mish"
    "SiLU"
    "Softplus"
    "Softsign"
    "Hardshrink"
    "Softshrink"
    "Tanhshrink"
    "PReLU"
    "RReLU"
    "CELU"
    "Hardtanh"
)

# Loop through each activation function
for activation in "${activations[@]}"
do
    echo "Running experiment with activation function: $activation"
    python source/fmnist_resnet_baseline.py --activation "$activation"
    echo "Finished experiment with activation function: $activation"
    echo "----------------------------------------"
done

echo "All experiments completed."
