import numpy as np
#import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences

def tokenize(texts, n_words=1000):
    tokenizer = Tokenizer(num_words=n_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, texts, labels, tokenizer, batch_size=32, max_len=100,
                 n_classes=2, n_words=1000, shuffle=True):
        'Initialization'
        self.max_len = max_len
        self.batch_size = batch_size
        self.texts = texts
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.tokenizer = tokenizer
        self.steps_per_epoch = int(np.floor(self.texts.size / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        texts = np.array([self.texts[k] for k in indexes])
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = np.array([self.labels[k] for k in indexes])

        return X, y

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.texts.size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
