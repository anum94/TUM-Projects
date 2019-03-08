import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
PADWORD = 'PAD'
n_inputs = 0
n_classes = 3
n_features = 0
VOCAB_SIZE = 0


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
    n_features = max_sentence_length
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
def convert_output_to_number(out):
    for index,author in enumerate(out):
        if author == "EAP":
            out[index] = 0
        elif author == "HPL":
            out[index] = 1
        else:
            out[index] = 2
    return out
class Cnn_Magic:
    def _init_(self):
        pass
    def experiment(self, train_X, train_y, test_X, test_y, num_train_sample, iterations,lr,bs):
        self.n_iterations = iterations
        self.learning_rate = lr
        self.batch_size = bs

        w0 = tf.get_variable('W0', shape=(3, 3, 1, 16), initializer=tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable('W1', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable('W2', shape=(n_features * 32, 64),initializer=tf.contrib.layers.xavier_initializer())
        w3 = tf.get_variable('W3', shape=(64, n_classes), initializer=tf.contrib.layers.xavier_initializer())


        b0 = tf.get_variable('B0', shape=(16), initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('B3', shape=(3), initializer=tf.contrib.layers.xavier_initializer())

        weights = {
            'wc1': w0,
            'wc2': w1,
            'wd1': w2,
            'out': w3,
            }

        biases = {
            'bc1': b0,
            'bc2': b1,
            'bd1': b2,
            'out': b3,
            }
        beta = 0.1
        # define place holder for both input and output
        x = tf.placeholder("float", [None, n_features, 1, 1])
        y = tf.placeholder("float", [None, 3])
        pred = self.conv_net(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        regularizer  = tf.nn.l2_loss(weights['wc1'] + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1'])
                                     + tf.nn.l2_loss(weights['out']))

        cost = tf.reduce_mean(cost + beta * regularizer)



        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # calculate accuracy across all the given images and average them out.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
           # Initializing the variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            train_loss = []
            test_loss = []
            train_accuracy = []
            test_accuracy = []
            loss = 0
            acc = 0
            summary_writer = tf.summary.FileWriter('./Output', sess.graph)
            for i in range(self.n_iterations):
                for batch in range(num_train_sample // self.batch_size):
                    batch_x = np.array(train_X[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])
                    batch_y = np.array(train_y[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])

                    # Run optimization op (backprop).
                    # Calculate batch loss and accuracy
                    opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                         y: batch_y})
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                      y: batch_y})
                print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))


            #Calculate accuracy for all test samples
            test_batch_size = int(len(test_X)/5)
            for batch_number in range(5):
                test_batch_x = np.array(
                    test_X[batch_number * test_batch_size:min((batch + 1) * test_batch_size, len(test_data))])
                test_batch_y = np.array(
                    test_y[batch_number * test_batch_size:min((batch + 1) * test_batch_size, len(test_data))])

                test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_batch_x, y: test_batch_y})
                train_loss.append(loss)
                test_loss.append(valid_loss)
                train_accuracy.append(acc)
                test_accuracy.append(test_acc)
                print("Testing Accuracy:", "{:.5f}".format(test_acc))
            summary_writer.close()


        return loss, acc, valid_loss
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.sigmoid(x)


    def conv_net(self, x, weights, biases):
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.

        #compute embedding

        '''
        embeds = tf.contrib.layers.embed_sequence(x, vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_SIZE)

        WINDOW_SIZE = EMBEDDING_SIZE
        STRIDE = int(WINDOW_SIZE / 2)
        conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE,
                                        stride=STRIDE, padding='SAME')  # (?, 4, 1)
        conv = tf.nn.relu(conv)  # (?, 4, 1)
        words = tf.squeeze(conv, [2])  # (?, 4)
        print (words.shape)
        '''

        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
# Reading training data
train_data = read_data("../data/train.csv")
index_word_dict, word_index_dict = create_word_dict(train_data)
tokenize_data = tokenize_pad_sentences(train_data, word_index_dict)
tokenize_data = one_hot_output(tokenize_data)
#output_labels = tf.convert_to_tensor(convert_output_to_number(list(train_data['author'])))
data = list(tokenize_data['indexed_text'])
num_test_sample = 90
data_labels = np.array(list(tokenize_data['author']))
N = len (train_data)
VOCAB_SIZE = len(data[1])
n_features = len(data[1])
EMBEDDING_SIZE = 10


data = np.array(data)
data = data.reshape(N, n_features, 1, 1)
indices = list(np.random.permutation(N))
test_idx, training_idx = indices[:num_test_sample], indices[num_test_sample:-1]
train_data, test_data = data[training_idx], data[test_idx]
train_labels , test_labels = data_labels[training_idx], data_labels[test_idx]


training_iterations = 1000
learning_rate = 0.01
batch_size = 128

cnn_model = Cnn_Magic()
loss, acc, valid_loss = cnn_model.experiment(train_X=train_data ,train_y=train_labels,test_X=test_data,test_y=test_labels, num_train_sample=len(train_labels),
                                                         iterations=training_iterations,lr=learning_rate,bs=batch_size)
print ("For learning rate " , learning_rate, " and batch size " , batch_size ," , the validation/test accuracy is " , valid_loss,  ".")
print ("Finish")
