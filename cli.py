"""Command Line Interface to run and compare all models or test certain model"""

import fire
from spooky_author_identification import prediction_by_frequency
from spooky_author_identification import cnn
from spooky_author_identification import lstm_pos

_MODEL_PATH = "model"

_DATA_PATH = "data/data.csv"
_TEST_DATA_PATH = "data/test_data.csv"
_TRAIN_DATA_PATH = "data/train_data.csv"

_PRED_BY_FREQ = "freq"
_CNN = "cnn"
_LSTM = "lstm"
_DNN = "dnn"


def run(data_path=_DATA_PATH, test_path=_TEST_DATA_PATH, method=None):
    """
    Run all models using pre-trained models
    :param method: methodology to test data
    :param data_path: path to whole data to divide testing and training
    :param test_path: path to testing data
    :return: void
    """

    if method is None:
        # todo run using DNN model
        print("!!! PREDICTION BY DEEP NEURAL NETWORK USING TENSORFLOW !!!")
        print("tflearn is used for training and optimizing the model.")

        print("\n-----------------------------------------------------------\n")
        # prediction by frequency
        prediction_by_frequency.testing(_MODEL_PATH, test_path)
    if method == _LSTM:
        lstm_pos.train_and_test(data_path)
    if method == _CNN:
        cnn.train_and_test(data_path)
    if method == _DNN:
        # todo run dnn
        pass
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.testing(_MODEL_PATH, test_path)

    print("\n-----------------------------------------------------------\n")
    print("!!! PREDICTION BY LSTM WITH POS TAGGING")
    print("accuracy: ~65%")

    print("\n-----------------------------------------------------------\n")
    print("!!! PREDICTION BY CNN")
    print("accuracy: ~50%")

    print("\n-----------------------------------------------------------\n")
    print("Please note that accuracy for LSTM and CNN are based on experimental runs.")
    print("If you want to train and test LSTM or CNN models please run respective command: ")
    print("python cli.py run --method CNN")
    print("pyton cli.py run --method LSTM")


def train(method, model_path=_MODEL_PATH, train_data_path=_TRAIN_DATA_PATH, test_ratio=0.2):
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.create_model(train_data_path, model_path)
    if method == _DNN:
        # todo train dnn model
        print("todo")
    if method == _CNN:
        cnn.train_and_test(train_data_path, test_ratio)
    if method == _LSTM:
        lstm_pos.train_and_test(train_data_path)


if __name__ == '__main__':
    fire.Fire()
