import nltk
import pandas as pd
import tensorflow as tf

class TextClass(object):
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



def read_data (data_file):
    return pd.read_csv(data_file)


def create_word_dict(texts):
    word_set = set()
    counter = 0

    for sentence in texts["text"]:
        sentence_tok = nltk.word_tokenize(sentence)
        for word in sentence_tok:
            word_set.add(word)

    word_index_dict = dict()
    word_dict = dict()
    for word in word_set:
        word_dict[counter] = word
        word_index_dict[word] = counter
        counter += 1

    return word_dict, word_index_dict


def tokenize_pad_sentences(data, index_dict):
    df = pd.DataFrame(columns=['author', 'indexed_text'])
    max_sentence_length = 0
    for _, row in data.iterrows():
        word_tokens = nltk.word_tokenize(row["text"])
        word_indexes = [index_dict[word] for word in word_tokens]
        if len(word_indexes) > max_sentence_length:
            max_sentence_length = len(word_indexes)
        df = df.append({'author': row["author"], 'indexed_text': word_indexes}, ignore_index=True)

    for i, row in enumerate(df["indexed_text"]):
        if len(row) < max_sentence_length:
            pads = [0] * (max_sentence_length - len(row))
            df["indexed_text"][i] = row + pads
    return df

def one_hot_output(data):
    for i, row in enumerate(data["author"]):
        one_hot_encoding = [0] * 3
        if row == "EAP":
            one_hot_encoding[0] = 1
        elif row == "HPL":
            one_hot_encoding[1] = 1
        else:
            one_hot_encoding[2] = 1
        data["author"][i] = one_hot_encoding
    return data

data = read_data("../data/train_sml.csv")
index_word_dict, word_index_dict = create_word_dict(data)
tokenize_data = tokenize_pad_sentences(data, word_index_dict)
tokenize_data = one_hot_output(tokenize_data)
