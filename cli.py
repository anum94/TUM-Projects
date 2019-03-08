"""Command Line Interface to run and compare all models or test certain model"""

import fire
from spooky_author_identification import prediction_by_frequency
from spooky_author_identification import dataset_seperation
from spooky_author_identification import cnn
from spooky_author_identification import lstm_pos

_PRED_BY_FREQ_MODEL_PATH = "data/pred_by_freq.pkl"
_TEST_DATA_PATH = "data/test_labeled.csv"
_TRAIN_DATA_PATH = "data/train.csv"

_PRED_BY_FREQ = "predByFreq"
_CNN = "cnn"
_LSTM = "lstm"


def run(data_path="data/train.csv", train_path="data/training_data.csv", test_path="data/testing_data.csv", method=None):

    """
    Run all models using pre-trained models
    :param test_ratio: ratio of data to use for testing
    :param data_path: path to whole data to divide testing and training
    :param train_path: path to store training data
    :param test_path: path to store testing data
    :return: void
    """

    if method == None:
        # todo run using DNN model
        print("!!! PREDICTION BY DEEP NEURAL NETWORK USING TENSORFLOW !!!")
        print("tflearn is used for training and optimizing the model.")
        prediction_by_frequency.testing(_PRED_BY_FREQ_MODEL_PATH, test_path)
    if method == _LSTM:
        lstm_pos.train_and_test(data_path)
    if method == _CNN:
        cnn.train_and_test(data_path)

    # separate data into two sets: training and testing
    # dataset_seperation.separate_data(test_ratio, data_path, train_path, test_path)

    # prediction by term frequency
    print("!!! PREDICTION BY FREQUENCY !!!")
    prediction_by_frequency.testing(_PRED_BY_FREQ_MODEL_PATH, test_path)
    print("PREDICTION BY FREQUENCY IS COMPLETED!")

    print("!!! PREDICTION BY LSTM WITH POS TAGGING")
    print("accuracy: ~65%")

    print("!!! PREDICTION BY CNN")
    print("accuracy: ~50%")

    print("Please note that accuracy for LSTM and CNN are based on experimental runs.")
    print("If you want to train and test LSTM or CNN models please run respective command: ")
    print("python cli.py run --method CNN")
    print("pyton cli.py run --method LSTM")




def test_method(method, path_to_model=_PRED_BY_FREQ_MODEL_PATH, path_to_test_data=_TEST_DATA_PATH):

    """
    Test data and output accuracy for a certain model
    :param method: provide one of the trained models: cnn, rnn, predByFreq, dnn
    :param path_to_model: path to stored model for chosen method
    :param path_to_test_data: path to test data
    :return: void
    """

    if method == _PRED_BY_FREQ:
        prediction_by_frequency.testing(path_to_model, path_to_test_data)


def train_model(method, path_to_model, path_to_train_data="data/training_data.csv"):
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.create_model(path_to_train_data, path_to_model)
    pass


def predict_sentence(sentence, method=_PRED_BY_FREQ, path_to_model=_PRED_BY_FREQ_MODEL_PATH):
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.predict_sentence(path_to_model, sentence)
    pass


if __name__ == '__main__':
    fire.Fire()
