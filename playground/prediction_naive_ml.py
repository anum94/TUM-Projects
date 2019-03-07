import nltk
import numpy as np
import pandas as pd
import tensorflow as tf

PADWORD = 'PAD'
n_inputs = 0
n_classes = 3
n_features = 0


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
        beta = 0.1
        weights = {
            'wc1': tf.get_variable('W0', shape=(3, 3, 1, 16), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 16, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W2', shape=(n_features * 64, 128),
                                   initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W3', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B3', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
        }

        # define place holder for both input and output
        x = tf.placeholder("float", [None, n_features, 1, 1])
        y = tf.placeholder("float", [None, 3])



        pred = self.conv_net(x, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        # Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # calculate accuracy across all the given images and average them out.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        regularizers = tf.reduce_mean(
            beta * (tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['out'])))

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            train_loss = []
            test_loss = []
            train_accuracy = []
            test_accuracy = []


            summary_writer = tf.summary.FileWriter('./Output', sess.graph)
            for i in range(self.n_iterations):
                for batch in range(num_train_sample // self.batch_size):
                    batch_x = np.array(train_X[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])
                    batch_y = np.array(train_y[batch * self.batch_size:min((batch + 1) * self.batch_size, num_train_sample)])

                    # Run optimization op (backprop).

                    opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                         y: batch_y})

                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([cost,accuracy], feed_dict={x: batch_x,
                                                                      y: batch_y})
                    #reg = sess.run(regularizers)
                    #loss += reg



                #print ("check point")
                #print("Iter " + str(i) + ", Loss= " + \
                #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #      "{:.5f}".format(acc))
                #print("Optimization Finished!")

                # Calculate accuracy for all test samples

                test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})
                train_loss.append(loss)
                test_loss.append(valid_loss)
                train_accuracy.append(acc)
                test_accuracy.append(test_acc)
                #print("Testing Accuracy:", "{:.5f}".format(test_acc))
            summary_writer.close()
        return loss, acc, valid_loss

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def conv_net(self, x, weights, biases):
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
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
train_data = read_data("../data/train_sml.csv")
index_word_dict, word_index_dict = create_word_dict(train_data)
tokenize_data = tokenize_pad_sentences(train_data, word_index_dict)
tokenize_data = one_hot_output(tokenize_data)

#output_labels = tf.convert_to_tensor(convert_output_to_number(list(train_data['author'])))
data = list(tokenize_data['indexed_text'])

num_test_sample = 20
data_labels = np.array(list(tokenize_data['author']))
N = len (train_data)
n_features = len(data[1])
## converting data to 4D matrix which is the desired dimensions for tensor
data = np.array(data)
data = data.reshape(N, n_features, 1, 1)

indices = list(np.random.permutation(N))
test_idx, training_idx = indices[:num_test_sample], indices[num_test_sample:-1]
train_data, test_data = data[training_idx], data[test_idx]
train_labels , test_labels = data_labels[training_idx], data_labels[test_idx]



#embeds = tf.contrib.layers.embed_sequence(train_data, vocab_size=len (index_word_dict), embed_dim=EMBEDDING_SIZE)
#print('words_embed={}'.format(embeds))

training_iterations = [200]
learning_rate = [0.1, 0.2, 0.4,0.02,0.05,0.5, 0.001]
batch_size = [2,4,8,16,32,64,128]


best_validation_loss = 1000
best_lr = 0
best_batch_size = 0

for iterations in training_iterations:
    for lr in learning_rate:
        for bs in batch_size:

            cnn_model = Cnn_Magic()

            loss, acc, valid_loss = cnn_model.experiment(train_X=train_data ,train_y=train_labels,test_X=test_data,test_y=test_labels, num_train_sample=len(train_labels),
                                                         iterations=iterations,lr=lr,bs=bs)
            print ("For learning rate " , lr, " and batch size " , bs ," , the validation/test accuracy is " , valid_loss,  ".")
            if valid_loss < best_validation_loss:
                best_validation_loss =  valid_loss
                best_batch_size = bs
                best_lr = lr

print ("Best learning rate " , best_lr, " and best batch size " , best_batch_size ," , which gives validation/test accuracy of " , best_validation_loss,  ".")
print ("Finish")