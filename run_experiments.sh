#!/bin/bash

DATA_DIR="flower_photos"  # Path to data directory
OUTPUT_FILE="results.txt"  # File for results

if [ -f "$OUTPUT_FILE" ]; then
    # If file exists
    echo "################## - next experiment - ##################" >> "$OUTPUT_FILE"
fi

# Loop through parameter configurations
for TR in 4 #12 in total
do
    for LR in 0.0001 0.0005 # Select two parameters for Learning rate: 0.0001, 0.0005
    do
        for BATCH_SIZE in 3 5 # Select two parameters for batch size: 8, 16
        do
            echo "Running with LR=$LR, Batch Size=$BATCH_SIZE"
            python flower_classification_tuning.py --data_dir $DATA_DIR --lr $LR --batch_size $BATCH_SIZE --num_epochs 3 --test_size 0.2 --output_file $OUTPUT_FILE
        done
    done
done

echo "Experiments finished. Results saved to $OUTPUT_FILE"