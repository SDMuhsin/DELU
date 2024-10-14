# DELUs: Dampened Exponential Linear Units

This repository contains the implementation and experimental code for the paper "DELUs: Dampened Exponential Linear Units". We introduce DELU (Dampened Exponential Linear Unit) and FADELU (Flexible and Adaptive DELU), novel activation functions for deep neural networks that demonstrate competitive performance across various image classification tasks.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)

## Installation

To set up the environment and install the required dependencies, run:

```bash
./install.sh
```

This script consists of a series of pip install commands.

## Project Structure

The repository is organized as follows:

- `/source/`: Contains the main source code
  - `common/DELU.py`: Implementation of the DELU activation function
  - Various `run_*.py` files for different datasets
- `/scripts/`: Bash scripts to run experiments
- `/data/`: Directory for storing datasets
- `/saves/`: Stores results in CSV format
- `/source/consolidate/cnn.py`: Script for tabulating results


## Usage

To run experiments on a specific dataset, use the corresponding script in the `/scripts/` directory. For example, to run experiments on KMNIST:

```bash
./scripts/run_kmnist.sh
```
The activation functions are picked from `./metadata/activations.txt`
You can modify the activation function parameters (either contants or initial values for learnable parameters) by passing arguments:

```bash
bash scripts/run_kmnist.sh 0.5 1.2 1 1
```

These are all set to 1 by default.

## Datasets

The paper evaluates the proposed activation functions on 8 image classification datasets:

1. MNIST
2. Fashion MNIST (FMNIST)
3. CIFAR-10
4. CIFAR-100
5. STL-10
6. SVHN (Street View House Numbers)
7. EMNIST (Extended MNIST)
8. KMNIST (Kuzushiji-MNIST)

Dataset files should be placed in the `/data/` directory. 
There are experimental files for other datasets but those need not work.

## Results

After running experiments, results are stored as CSV files in the `/saves/` directory. To consolidate and tabulate the results, use:

```bash
python source/consolidate/cnn.py
```





