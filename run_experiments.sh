#!/bin/bash

DATA_DIR="flower_photos"  # Pfad zu deinem Dataset
OUTPUT_FILE="results.txt"  # Datei für die Ergebnisse

# Falls die Datei existiert, alte Ergebnisse löschen
rm -f $OUTPUT_FILE

# Parameter-Kombinationen durchlaufen
for TR in 4 #12 in total
do
    for LR in 0.0001 0.00001
    do
        for BATCH_SIZE in 1 2 # 8, 16
        do
            echo "Running with LR=$LR, Batch Size=$BATCH_SIZE"
            python flower_classification_tuning.py --data_dir $DATA_DIR --lr $LR --batch_size $BATCH_SIZE --num_epochs 1 --test_size 0.99 --output_file $OUTPUT_FILE
        done
    done
done

echo "Experiments finished. Results saved to $OUTPUT_FILE"