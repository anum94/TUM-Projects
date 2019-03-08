import numpy as np
import pandas as pd
import nltk as nl
from numpy import array
import tensorflow as tf
import tflearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
stemmer = nl.stem.lancaster.LancasterStemmer()

TRAIN_DATA_PATH = '../data/train_sml.csv'

def read_file(file_path):
    return pd.read_csv(file_path)

def preprocess_data(train):
    train['tokens'] = [nl.word_tokenize(sentences) for sentences in train.text]

    words = []
    for item in train.tokens:
        words.extend(item)



    words = [stemmer.stem(word) for word in words]
    words = set(words)

    training = []
    for index, item in train.iterrows():
        onehot = []
        token_words = [stemmer.stem(word) for word in item['tokens']]
        for w in words:
            onehot.append(1) if w in token_words else onehot.append(0)

        training.append([onehot, item['author']])

    training_new = np.array(training)
    return training_new, words

def one_hot_label(y):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


training_data = read_file(TRAIN_DATA_PATH)
training_data.head()

processed_training_data, words = preprocess_data(training_data)

train_x = list(processed_training_data[:,0])
train_y = one_hot_label(processed_training_data[:,1])


# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
 
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=10, batch_size=8, show_metric=True)
model.save('model.tflearn')


test = pd.read_csv('../data/test_data.csv')
test.head()



test['tokens'] = [nl.word_tokenize(sentences) for sentences in test.text]


testing = []
for index,item in test.iterrows():
    onehot = []
    token_words = [stemmer.stem(word) for word in item['tokens']]
    for w in words:
        onehot.append(1) if w in token_words else onehot.append(0)
    
    testing.append(onehot)


testing = list(np.array(testing))


predicted = model.predict(X=testing)



result_val = round(pd.DataFrame(predicted),6)
result_val.columns = ["EAP","HPL","MWS"]



result = pd.DataFrame(columns=['id'])
result['id'] = test['id']



result['EAP'] = result_val['EAP']
result['HPL'] = result_val['HPL']
result['MWS'] = result_val['MWS']


result.head()


def calculate_model_accuracy(result):

    final_results = dict()
    real_result = dict()
    for index, row in result.iterrows():
        author = ""
        row = list(row)
        id = row.pop(0)
        index = row.index(max(row))
        if index == 0:
            author = "EAP"
        elif index == 1:
            author = "HPL"
        else:
            author = "MWS"
        final_results[id] = author

    for index, row in test.iterrows():

        row = list(row)
        id = row.pop(0)
        author = row.pop(1)

        real_result[id] = author

    total_test_samples = 0
    correct_samples = 0

    for id, author in final_results.items():
        total_test_samples += 1
        if real_result[id] == author:
            correct_samples += 1
    accuracy = correct_samples/total_test_samples
    return accuracy

acc = calculate_model_accuracy(result)
print ("Accuracy: ", acc)

