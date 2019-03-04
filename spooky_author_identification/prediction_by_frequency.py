import nltk
import pandas as pd
import pickle


class PredictionByFreq:
    AUTHOR = "author"
    WORD = "word"
    PROBABILITY = "probability"
    JOINT_PROBABILITY = "joint_probability"
    TEXT = "text"

    def __init__(self):
        pass

    def create_model(self, training_data, model_path):
        training_texts = pd.read_csv(training_data)
        by_author = training_texts.groupby(self.AUTHOR)
        word_freq_by_author = nltk.probability.ConditionalFreqDist()

        for name, group in by_author:
            sentences = group[self.TEXT].str.cat(sep=' ').lower()
            tokens = nltk.tokenize.word_tokenize(sentences)
            frequency = nltk.FreqDist(tokens)
            word_freq_by_author[name] = frequency

        f = open(model_path, "wb")
        pickle.dump(word_freq_by_author, f)
        f.close()

    def predict_sentence(self, sentence, word_freq_by_author_model):
        preprocessed_test_sentence = nltk.tokenize.word_tokenize(sentence.lower())
        test_probabilities = pd.DataFrame(columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

        for i in word_freq_by_author_model.keys():
            for j in preprocessed_test_sentence:
                word_freq = word_freq_by_author_model[i].freq(j) + 0.000001
                output = pd.DataFrame([[i, j, word_freq]], columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

                test_probabilities = test_probabilities.append(output, ignore_index=True)

        test_probabilities_by_author = pd.DataFrame(columns=[self.AUTHOR, self.JOINT_PROBABILITY])

        for i in word_freq_by_author_model.keys():
            one_author = test_probabilities.query(self.AUTHOR + ' == "' + i + '"')
            joint_probability = one_author.product(numeric_only=True)[0]

            output = pd.DataFrame([[i, joint_probability]], columns=[self.AUTHOR, self.JOINT_PROBABILITY])
            test_probabilities_by_author = test_probabilities_by_author.append(output, ignore_index=True, sort=True)

        return test_probabilities_by_author.loc[test_probabilities_by_author[self.JOINT_PROBABILITY].idxmax(),
                                                self.AUTHOR]

    def calculate_accuracy(self, path_to_model, test_data):
        f = open(path_to_model, "rb")
        word_freq_by_author_model = pickle.load(f)
        f.close()

        t = 0
        f = 0
        test_texts = pd.read_csv(test_data)
        counter = 0
        for sentence, author in zip(test_texts[self.TEXT], test_texts[self.AUTHOR]):
            counter += 1
            predicted_author = self.predict_sentence(sentence, word_freq_by_author_model)
            if predicted_author == author:
                t += 1
            else:
                f += 1

            if counter % 100 == 0:
                print("accuracy of %s sentences is %s percentage " % (str(counter), str((t/(t+f))*100)))
            if counter == 400:
                break
        return t, f


def calculate_accuracy(path_to_model, path_to_test_data):
    pred_by_freq = PredictionByFreq()
    t, f = pred_by_freq.calculate_accuracy(path_to_model, path_to_test_data)
    print("accuracy of model is %s " % (str((t/(t+f))*100)))


def predict_sentence(path_to_model, sentence):
    pred_by_freq = PredictionByFreq()

    f = open(path_to_model, "rb")
    word_freq_by_author_model = pickle.load(f)
    f.close()

    predicted_author = pred_by_freq.predict_sentence(sentence, word_freq_by_author_model)
    print("\"%s\" is most likely said by %s" % (sentence, predicted_author))


def create_model(path_to_train_data, path_to_model):
    pred_by_freq = PredictionByFreq()
    pred_by_freq.create_model(path_to_train_data, path_to_model)
    pass

