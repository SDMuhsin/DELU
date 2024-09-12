#!/bin/bash
activations=(
    "SEL"
    "HCR"
    "LogSig"
)

# Loop through each activation function
for activation in "${activations[@]}"
do
    echo "Running experiment with activation function: $activation"
    python source/fmnist_resnet_new.py --activation "$activation"
    echo "Finished experiment with activation function: $activation"
    echo "----------------------------------------"
done

echo "All experiments completed."
