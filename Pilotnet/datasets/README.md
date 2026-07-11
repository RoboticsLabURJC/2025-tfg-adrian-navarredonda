## datasets

This directory contains the datasets ready to be used for training of PilotNet.

This directory contains the **train**, **validation**, and **test** directories, each one with the following use:

    - train: Contains the 70% of the data, and will be use for the training of the models
    - validation: Contains the 15% of the data, and will be use for the validation of the training of the models
    - test: Contains the 15% of the data, and will be use for testing the models once they are train.

This directories will contain several different datasets, so the name of the csvs will be used to distinguish what train file goes 
whith its validation and test file.
The name of the files use for each directory uses the following logic:

    - train:        track_x_train_xxx.csv
    - validation:   track_x_validation_xxx.csv
    - test:         track_x_test_xxx.csv
Being the x, the number of track or number of dataset of that particular track