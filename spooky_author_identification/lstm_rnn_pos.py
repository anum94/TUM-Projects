import numpy as np
import tensorflow as tf
import nltk
import pandas as pd


class PredictionByLSTM:

    def __init__(self):
        self.batch_size = 100
        self.word2index_map_tags = {}
        self.word2index_map = {}
        self.num_classes = 3

    def __get_sentence_batch(self, batch_size, data_x, data_y, data_seqlens, data_x_tags):
        instance_indices = list(range(len(data_x)))
        np.random.shuffle(instance_indices)
        batch = instance_indices[:batch_size]
        x = [[self.word2index_map[word] for word in data_x[i]]
             for i in batch]
        x2 = [[self.word2index_map_tags[word] for word in data_x_tags[i]]
              for i in batch]
        y = [data_y[i] for i in batch]
        seqlens = [data_seqlens[i] for i in batch]
        return x, y, seqlens, x2

    def generate_data(self, raw_data):
        default_st = nltk.sent_tokenize
        default_wt = nltk.word_tokenize

        EAP = str()
        MWS = str()
        HPL = str()
        train_text = pd.read_csv(raw_data)
        for _, row in train_text.iterrows():
            if row["author"] == "EAP":
                EAP += " " + str(row["text"])
            if row["author"] == "MWS":
                MWS += " " + str(row["text"])
            if row["author"] == "HPL":
                HPL += " " + str(row["text"])

        eap_sentences = default_st(text=EAP)
        eap_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in eap_sentences]
        eap_words = [[word[0] for word in sentence] for sentence in eap_tuples]
        eap_tags = [[word[1] for word in sentence] for sentence in eap_tuples]
        eap_len = len(eap_sentences)

        mws_sentences = default_st(MWS)
        mws_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in mws_sentences]
        mws_words = [[word[0] for word in sentence] for sentence in mws_tuples]
        mws_tags = [[word[1] for word in sentence] for sentence in mws_tuples]
        mws_len = len(mws_sentences)

        hpl_sentences = default_st(HPL)
        hpl_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in hpl_sentences]
        hpl_words = [[word[0] for word in sentence] for sentence in hpl_tuples]
        hpl_tags = [[word[1] for word in sentence] for sentence in hpl_tuples]
        hpl_len = len(hpl_sentences)

        data = np.array(eap_words + mws_words + hpl_words)
        data_tags = np.array(eap_tags + mws_tags + hpl_tags)
        data_len = len(data)

        max_len = 876
        # for sent in data:
        #     if len(sent) > max_len:
        #         max_len = len(sent)

        seqlens = []

        for sentence_id in range(data_len):
            seqlens.append(len(data[sentence_id]))

            if len(data[sentence_id]) < max_len:
                pads = ['PAD'] * (max_len - len(data[sentence_id]))
                data[sentence_id] = data[sentence_id] + pads

        ### tags ###
        for sentence_id in range(data_len):
            if len(data_tags[sentence_id]) < max_len:
                pads = ['PAD'] * (max_len - len(data_tags[sentence_id]))
                data_tags[sentence_id] = data_tags[sentence_id] + pads

        ####
        labels = [2] * eap_len + [1] * mws_len + [0] * hpl_len

        for i in range(len(labels)):
            label = labels[i]
            one_hot_encoding = [0] * self.num_classes
            one_hot_encoding[label] = 1
            labels[i] = one_hot_encoding

        index = 0
        for sent in data:
            for word in sent:
                if word not in self.word2index_map:
                    self.word2index_map[word] = index
                    index += 1

        #### tags ####
        index = 0
        for sent in data_tags:
            for tag in sent:
                if tag not in self.word2index_map_tags:
                    self.word2index_map_tags[tag] = index
                    index += 1

        return data, labels, seqlens, data_tags, max_len

    def training(self, model_path, training_data):
        epochs = 10

        data, labels, seqlens, data_tags, max_len = self.generate_data(training_data)
        data_len = len(data)

        ###
        train_size = int(data_len)  # has to be integer for slicing array
        data_indices = list(range(len(data)))
        np.random.shuffle(data_indices)

        data = np.array(data)[data_indices]
        labels = np.array(labels)[data_indices]
        seqlens = np.array(seqlens)[data_indices]
        train_x = data[:train_size]  # added dimension of array
        train_y = labels[:train_size]
        train_seqlens = seqlens[:train_size]

        #### tags ###
        data_tags = np.array(data_tags)[data_indices]
        train_x_tags = data_tags[:train_size]

        final_output, _labels, _inputs, _seqlens, _inputs_tags, accuracy = self.tf_scopes(max_len)

        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=_labels)
        cross_entropy = tf.reduce_mean(softmax)

        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(epochs):
                x_batch, y_batch, seqlen_batch, x2_batch = self.__get_sentence_batch(self.batch_size, train_x, train_y,
                                                                                     train_seqlens, train_x_tags)
                sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                                _seqlens: seqlen_batch, _inputs_tags: x2_batch})

                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                        _labels: y_batch,
                                                        _seqlens: seqlen_batch, _inputs_tags: x2_batch})
                    print("Accuracy at %d: %.5f" % (step, acc))

            tf.train.Saver().save(sess, model_path, write_meta_graph=True)

    def tf_scopes(self, max_len):
        embedding_dimension = 64
        embedding_dimension_tags = 32
        hidden_layer_size = 128
        num_LSTM_layers = 4
        hidden_layer_size_tags = 64
        num_LSTM_layers_tags = 2

        index2word_map = {index: word for word, index in self.word2index_map.items()}
        vocabulary_size = len(index2word_map)
        index2word_map_tags = {index: tag for tag, index in self.word2index_map_tags.items()}
        vocabulary_size_tags = len(index2word_map_tags)

        _inputs = tf.placeholder(tf.int32, shape=[self.batch_size, max_len], name='Input')
        _labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_classes], name='Labels')
        _seqlens = tf.placeholder(tf.int32, shape=[self.batch_size], name='Seqlens')
        _inputs_tags = tf.placeholder(tf.int32, shape=[self.batch_size, max_len], name='Input_tags')

        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size,
                                   embedding_dimension],
                                  -1.0, 1.0), name='embedding')
            embed = tf.nn.embedding_lookup(embeddings, _inputs)

        with tf.variable_scope("lstm"):
            # Define a function that gives the output in the right shape
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)

            cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers)],
                                               state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                                sequence_length=_seqlens,
                                                dtype=tf.float32)

        #### tags ####
        with tf.name_scope("embeddings_tags"):
            embeddings_tags = tf.Variable(
                tf.random_uniform([vocabulary_size_tags,
                                   embedding_dimension_tags],
                                  -1.0, 1.0), name='embedding_tags')
            embed_tags = tf.nn.embedding_lookup(embeddings_tags, _inputs_tags)

        with tf.variable_scope("lstm_tags"):
            # Define a function that gives the output in the right shape
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size_tags, forget_bias=1.0)

            cell_tags = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers_tags)],
                                                    state_is_tuple=True)
            outputs_tags, states_tags = tf.nn.dynamic_rnn(cell_tags, embed_tags,
                                                          sequence_length=_seqlens,
                                                          dtype=tf.float32)

        weights = {
            'linear_layer': tf.Variable(
                tf.truncated_normal([hidden_layer_size + hidden_layer_size_tags, self.num_classes],
                                    mean=0, stddev=.01))
        }
        biases = {
            'linear_layer': tf.Variable(tf.truncated_normal([self.num_classes], mean=0, stddev=.01))
        }
        # extract the last relevant output and use in a linear layer
        lstm_states = tf.concat([states[num_LSTM_layers - 1][1], states_tags[num_LSTM_layers_tags - 1][1]], 1)
        final_output = tf.matmul(lstm_states, weights["linear_layer"]) + biases["linear_layer"]

        correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

        return final_output, _labels, _inputs, _seqlens, _inputs_tags, accuracy

    def testing(self, model_path, testing_data):
        data, labels, seqlens, data_tags, max_len = self.generate_data(testing_data)

        final_output, _labels, _inputs, _seqlens, _inputs_tags, accuracy = self.tf_scopes(max_len)

        train_size = int(len(data))

        test_x = data[:train_size]
        test_y = labels[:train_size]
        test_seqlens = seqlens[:train_size]
        test_x_tags = data_tags[:train_size]
        test_y = labels[:train_size]
        test_seqlens = seqlens[:train_size]

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('model/'))

            graph = tf.get_default_graph()
            final_output, _labels, _inputs, _seqlens, _inputs_tags, accuracy = self.tf_scopes(max_len)

            w1 = graph.get_tensor_by_name("w1:0")
            w2 = graph.get_tensor_by_name("w2:0")
            feed_dict = {w1: 13.0, w2: 17.0}

            # Now, access the op that you want to run.
            op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

            print
            sess.run(op_to_restore, feed_dict)
            # tf.train.Saver().restore(sess, model_path)

            mean_acc = 0
            for test_batch in range(5):
                x_test, y_test, seqlen_test, x2_test = self.__get_sentence_batch(self.batch_size,
                                                                                 test_x, test_y,
                                                                                 test_seqlens, test_x_tags)
                batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                                 feed_dict={_inputs: x_test,
                                                            _labels: y_test,
                                                            _seqlens: seqlen_test, _inputs_tags: x2_test})
                print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
                mean_acc = mean_acc + batch_acc

            print("Mean test accuracy: %.5f" % (mean_acc / 5))


predByLSTM = PredictionByLSTM()
# predByLSTM.training("model/model.ckpt", "../data/train_test_removed.csv")
predByLSTM.testing("model/model.ckpt", "../data/test_labeled.csv")

