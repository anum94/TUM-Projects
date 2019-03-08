# Spooky Author Identification

## Description of the Problem and Approachs

Our challenge is the old Kaggle competition that we need to predict an author of excerpts from horror stories 
by Edgar Allan Poe, Mary Shelley, and HP Lovecraft. 

We have addressed to the problem with four different approach.

1. CNN
    - We used 4 layer convolutional neural network with filter size 2x2 with softmax activation function.
    - Before training, we preprocessed the data similar to those provided in the seminar.
    - Using our custom hyperparameter optimizer, we run our method by changing the below parameters and analyzed the
        results. Total training with each different parameter values take 36 hours.
        Parameters we optimized: learning rate, optimization algorithm, batch size, activation function, number of hidden layer, 
        number of hidden units of each layer.
    - The methodology does not give satisfiable accuracy. Mean accuracy: ~45%
2. LSTM with POS Tagging
    - For comparision, we also tried LSTM with POS Tagging that has been explained in the seminar. We refactored the
    solution provided and adapted into the given problem.
    - Mean accuracy: ~65%
3. Pure Deep Neural Network (DNN)
    - After trying and not satisfying with results of CNN and LSTM, we decided to focus more on preprocessing.
    - To preprocess the data, we used Bag of Words approach followed by stemming. And for 
    predication we used Pure Deep Neural Network with three fully connected layers and softmax activation for the output.
    - We realized with efficient preprocessing of the data, we were able to outperform the above complex methods.
    - Mean accuracy: ~86%
4. Term Frequency Based Comparision
    - After working on DNN, we decided to develop very basic approach that is based grouping term frequency of words for
    each author. By counting the term frequency, we predict test data.
    - Mean accuracy: ~60%
    
#### Summary:
We realized that texts of the given three authors are very similar each other. Therefore, hyperparameter optimization
causes either over- or under fitting. Neural network stuggles to differentiate the three types of data. As a comparision,
the examples (carroll alice, melville, austen sense) in the seminars have more obvious differerences among data. After spending
substantial amount of time on CNN optimization, we decided to focus more on postprocessing the data. As a result Even basic term 
frequency based approach gave better result than CNN based approach. Merging good preprocessing with simple neural network resulted
with satisfiable accuracy.


## Running the models

We have designed simple command line interface (CLI) to easily generate and run all models. The models can be 
retrained with different data using our custom CLI.

You just need to run `cli.py`:

``python cli.py run``

The command will run DNN and PredByFreq methods using pre-trained models and test_data.csv. And print the latest
testing results for LSTM and CNN.


#### Run single model
You can run each method individually using CLI command:

``python cli.py run --method lstm``

``python cli.py run --method cnn``

``python cli.py run --method dnn``

``python cli.py run --method freq``

You can also set the test ratio of the data for CNN method (i.e. use 40% of data for testing and 60% training):

``python cli.py run --method cnn --test_ratio 0.4``


#### Train single model

You can also train your own model using provided or different data:

``python cli.py train --method --model_path "path/to/dir/" --train_data_path "path/to/train/data.csv"``


## Data

There are 3 csv files provided for testing and training:

- data.csv: whole data that is used by CNN and LSTM based methods. Those methods divide the data testing and training
as part of the process.
- train.csv: 80% of original data. Used for training of DNN and Prediction By Frequency methods
- test.csv: The rest 20% of original data: Used for testing of DNN and Prediction By Frequency methods