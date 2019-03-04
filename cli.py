import fire
from spooky_author_identification import prediction_by_frequency

_PRED_BY_FREQ_MODEL_PATH = "data/pred_by_freq_model.pkl"
_TEST_DATA_PATH = "data/test_labeled.csv"
_TRAIN_DATA_PATH = "data/train.csv"

_PRED_BY_FREQ = "predByFreq"
_FAST_TEXT = "fastText"
_CNN = "cnn"
_RNN = "rnn"

METHODS = [_PRED_BY_FREQ]


def run_and_compare_models():
    pass


def accuracy_of_method(method, path_to_model=_PRED_BY_FREQ_MODEL_PATH, path_to_test_data=_TEST_DATA_PATH):
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.calculate_accuracy(path_to_model, path_to_test_data)


def predict_sentence(sentence, method=_PRED_BY_FREQ, path_to_model=_PRED_BY_FREQ_MODEL_PATH):
    if method == _PRED_BY_FREQ:
        prediction_by_frequency.predict_sentence(path_to_model, sentence)
    pass


def train_model(method, path_to_train_data, path_to_model):
    pass


if __name__ == '__main__':
    fire.Fire()
