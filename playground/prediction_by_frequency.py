import nltk
import pandas as pd


class PredictionByFreq:
    AUTHOR = "author"
    WORD = "word"
    PROBABILITY = "probability"
    JOINT_PROBABILITY = "joint_probability"
    TEXT = "text"

    def __init__(self, data_to_read):
        self.texts = pd.read_csv(data_to_read)
        self.by_author = self.texts.groupby(self.AUTHOR)
        self.word_freq_by_author = nltk.probability.ConditionalFreqDist()

    def calculate_freq_by_element(self):
        for name, group in self.by_author:
            sentences = group[self.TEXT].str.cat(sep=' ').lower()
            tokens = nltk.tokenize.word_tokenize(sentences)
            frequency = nltk.FreqDist(tokens)
            self.word_freq_by_author[name] = frequency

    def find_word_freq_by_element(self, word):
        for author in self.word_freq_by_author.keys():
            print ("%s: %s" % (word, author))
            print(self.word_freq_by_author[author].freq(word))

    def predict_sentence(self, sentence):
        preprocessed_test_sentence = nltk.tokenize.word_tokenize(sentence.lower())
        test_probabilities = pd.DataFrame(columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

        for i in self.word_freq_by_author.keys():
            for j in preprocessed_test_sentence:
                word_freq = self.word_freq_by_author[i].freq(j) + 0.000001
                output = pd.DataFrame([[i, j, word_freq]], columns=[self.AUTHOR, self.WORD, self.PROBABILITY])

                test_probabilities = test_probabilities.append(output, ignore_index=True)

        test_probabilities_by_author = pd.DataFrame(columns=[self.AUTHOR, self.JOINT_PROBABILITY])

        for i in self.word_freq_by_author.keys():
            one_author = test_probabilities.query(self.AUTHOR + ' == "' + i + '"')
            joint_probability = one_author.product(numeric_only=True)[0]

            output = pd.DataFrame([[i, joint_probability]], columns=[self.AUTHOR, self.JOINT_PROBABILITY])
            test_probabilities_by_author = test_probabilities_by_author.append(output, ignore_index=True, sort=True)

        print ("-----------------------------------------------------------------------------------------------------")
        print ("\"%s\" is most likely said by %s" % (sentence, test_probabilities_by_author.loc[
            test_probabilities_by_author[self.JOINT_PROBABILITY].idxmax(), self.AUTHOR]))


def pred_by_freq_playground():
    pred_by_freq = PredictionByFreq("../data/train.csv")
    pred_by_freq.calculate_freq_by_element()

    pred_by_freq.find_word_freq_by_element("blood")
    pred_by_freq.find_word_freq_by_element("gun")
    pred_by_freq.find_word_freq_by_element("dead")

    pred_by_freq.predict_sentence("gun is bloody hell")


pred_by_freq_playground()
