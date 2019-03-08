# Spooky Author Identification

The problem we occur is the old Kaggle compatition to predict the author of excerpts from horror stories by Edgar Allan
Poe, Mary Shelley, and HP Lovecraft. 

We have implemented different models and adapted to the data. Methods used:
1. Pure Deep Neural Network
2. CNN
3. LSTM with POS Tagging
4. Term Frequency Based Comparision

The first method gives the best accuracy (around 86%). LSTM with POS tagging method is inspired and taken by seminar
tutorial and adapted into the above problem. The result is around 65%. CNN and Term Frequency based approaches do not
behave well with the data.

## Running the models

We have designed simple command line interface (CLI) to easily generate and run all models. The models can be 
retrained with different data using our custom CLI.

You just need to run `cli.py`:

``python cli.py run``

The command will randomly divide data in "data/data.csv" to training and testing before running the models. 
It will generate two files: _training_data.csv_ and _testing_data.csv_. By default, 20% of the data is used as testing 
data, and 80% as training data. If you prefer to have a different ratio (i.e 40% testing data), you can just run 
the below command:

``python cli.py run --test_ratio 0.4``


Running the above command will generate testing and training accuracy for each method. 



## Playground

CLI allow a user to test certain method using training models or reading different data. Or methods even can be used for 
different text classification problem if the data is in the required format.

#### Test single model

You can test single model. METHOD_NAME must be one of the trained methods below:
_**dnn**_ for deep neural network; _**rnn**_ for recurrence neural network; _**cnn**_ for convolutional neural network; 
predByFreq for term frequency based approach
``python cli.py test_method --method METHOD_NAME``

You can use different testing data:
``python cli.py test_method --method METHOD_NAME --test_data  "PATH_TO_TEST_DATA"``

The command will print out accuracy for the given method and testing data

#### Train a model

You can re-train a model for provided methodology with original data.
model_path flag is the path to store the generated model


``python cli.py train_method --method METHOD_NAME --model_path MODEL_PATH``

You can train with different data as well:

``python cli.py train_method --method METHOD_NAME --train_data "PATH_TO_TRAIN_DATA" --model_path MODEL_PATH``

#### Predict a sentence

It is possible to predict a sentence using CLI

``python cli.py predict_sentence --method METHOD_NAME --sentence "This is scary sentence to predict" ``