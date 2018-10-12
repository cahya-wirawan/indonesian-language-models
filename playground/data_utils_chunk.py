import numpy as np
#import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, reader, batch_size=32, max_len=100,
                 n_classes=2, n_words=1000, shuffle=True):
        'Initialization'
        self.max_len = max_len
        self.batch_size = batch_size
        self.reader = reader
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.tokenizer = Tokenizer(num_words=n_words)
        for df in reader:
            texts = df.values[:, 1]
            self.tokenizer.fit_on_texts(texts)
        self.steps_per_epoch = int(np.floor(self.tokenizer.document_count / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        df_temp = [self.reader.iloc[k].values for k in indexes]

        # Generate data
        X, y = self.__data_generation(df_temp)

        return X, y

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.tokenizer.document_count)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.max_len))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(df_temp):
            # Store sample
            X[i,] = self.tokenizer.texts_to_sequences(ID[1].values)
            # Store class
            y[i] = ID[0]

        return X, to_categorical(y, num_classes=self.n_classes)