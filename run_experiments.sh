#!/bin/bash

OUTPUT_FILE="results.txt"  # File for results

if [ -f "$OUTPUT_FILE" ]; then
    # If file exists
    echo "################## - next experiment - ##################" >> "$OUTPUT_FILE"
fi

# Loop through parameter configurations
for TR in 1 2 3 4 5 6 7 8 9 10 # One run per number
do
    for DROPOUT in 0.0 0.2 # Select two parameters for dropout: 0.0, 0.2
    do
        for BATCH_SIZE in 32 64 # Select two parameters for batch size: 8, 16
        do
            for LR in 0.00001 0.00005 0.0001 # Select two parameters for Learning rate: 0.0001, 0.0005
            do 
                echo "Running with LR=$LR, Batch Size=$BATCH_SIZE, Dropout=$DROPOUT"
                python flower_classification_tuning_largeDataSet_small.py --lr $LR --batch_size $BATCH_SIZE --num_epochs 5 --test_size 0.2 --dropout $DROPOUT --output_file $OUTPUT_FILE
            done
        done
    done
done

echo "Experiments finished. Results saved to $OUTPUT_FILE"
