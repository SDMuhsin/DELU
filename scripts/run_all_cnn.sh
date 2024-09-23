#!/bin/bash

# List of tasks
tasks=(cinic10 tinyimagenet)  #(mnist fmnist cifar10 cifar100 svhn stl10 tinyimagenet cinic10)

# List of (a,b) pairs
a_b_pairs=("1 1" "0.5 1.2" "0.7 1.1")

# Loop through each task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    
    # Loop through each (a,b) pair
    for pair in "${a_b_pairs[@]}"; do
        # Split the pair into a and b
        a=$(echo $pair | cut -d' ' -f1)
        b=$(echo $pair | cut -d' ' -f2)
        
        echo "  With a=$a and b=$b"
        
        # Run the task-specific script with the current a and b values
        ./scripts/run_${task}.sh $a $b
    done
    
    echo "Completed task: $task"
    echo "------------------------"
done

echo "All tasks completed."
