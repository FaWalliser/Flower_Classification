# Flower Classification Model Training

This project is used to train a model on 102 flower categories for classifying flower images.

# How to start the program
There are multiple commands to either run the program with predefined or specific parameters or to even have an automated run with all combinations stated in our Milestone 2.

The program can be run with predefined parameters:
```python flower_classification_tuning_largeDataSet_small.py```

The program can be run with specific parameters (if you don't mention parameters, the predefinitions will be used):
python flower_classification_tuning_largeDataSet_small.py --data_dir <training data set path> --mat_file_dir <image label matrix path> --lr <learning rate value> --batch_size <batch size value> --num_epochs <number of epochs> --test_size <test size (percentage)> --dropout <dropout value> --output_file <name of the file for the results>

If you want to run exactly the configurations we used for training the model, use the following command:
bash run_experiments.sh
