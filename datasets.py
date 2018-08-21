import keras
import numpy as np
from keras.datasets import imdb
from utils import load_data
from keras.preprocessing import sequence

NUM_WORDS=10000 # only use top 10000 words
INDEX_FROM=3   # word index offset
MAXLEN=200

class Imdb():
    """
    Convert the tokens from imdb format to local format.
    """
    def __init__(self):
        self.ptb = load_data()
        # train_data, valid_data, total_words, reversed_dictionary, dictionary, wordlist = load_data()

        # (train_x, train_y), (test_x, test_y) = imdb.load_data(path="imdb.npz",
        self.imdb = imdb.load_data(path="imdb.npz",
                                  num_words=NUM_WORDS,
                                  skip_top=0,
                                  maxlen=None,
                                  seed=113,
                                  start_char=1,
                                  oov_char=2,
                                  index_from=INDEX_FROM)

        self.train_x = sequence.pad_sequences(self.imdb[0][0], maxlen=MAXLEN+1,
                                              padding='post', truncating='post')
        self.train_x = self.train_x[:, 1:]
        self.test_x = sequence.pad_sequences(self.imdb[1][0], maxlen=MAXLEN+1,
                                             padding='post', truncating='post')
        self.test_x = self.test_x[:, 1:]

        self.word_to_id = keras.datasets.imdb.get_word_index()
        self.word_to_id = {k:(v+INDEX_FROM) for k,v in self.word_to_id.items()}
        self.word_to_id["<eos>"] = 0
        self.word_to_id["<start>"] = 1
        self.word_to_id["<unk>"] = 2

        self.id_to_word = {value:key for key,value in self.word_to_id.items()}

    def id_convert(self, data_in, id_to_word, dictionary):
        data_out = []
        for data in data_in:
            sentence = [id_to_word[id] for id in data]
            # print(' '.join(sentence))
            sentence_id = [dictionary[word] if word in dictionary else dictionary['<unk>']
                           for word in sentence]
            data_out.append(sentence_id)
            # print("train_x", data[:10])
            # print("sentence_id", sentence_id[:10])
        return np.array(data_out)

    def load_data(self):
        train_x = self.id_convert(self.train_x, self.id_to_word, self.ptb[4])
        test_x = self.id_convert(self.test_x, self.id_to_word, self.ptb[4])
        return (train_x, self.imdb[0][1]), (test_x, self.imdb[1][1])

if __name__ == '__main__':
    imdb = Imdb()
    (train_x, train_y), (test_x, test_y) = imdb.load_data()
    print("train shape:", train_x.shape)
