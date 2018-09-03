import tensorflow.keras.backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import (LSTM, Activation, Dense, Dropout,
                                     Embedding, TimeDistributed)
from tensorflow.keras.models import Sequential
import numpy as np

#import word_embedding

EMBEDDING_FILENAME = 'data/input/word_embeddings/glove.6B.100d.txt'

def perplexity(y_true, y_pred):
    """
    Perplexity
    """
    ce = K.categorical_crossentropy(y_true, y_pred)
    #pp = K.exp(cross_entropy)
    perplexity = np.power(2, ce)
    #cross_entropy = K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
    #perplexity = K.exp(cross_entropy)
    return perplexity

def crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def create_model(total_words, hidden_size, num_steps, optimizer='adam', wordlist=None):
    model = Sequential()
    # embedding_layer = word_embedding.embedding_layer(embedding_filename=EMBEDDING_FILENAME, wordlist=wordlist)
    embedding_layer = Embedding(total_words, hidden_size, input_length=num_steps)
    model.add(embedding_layer)
    model.add(LSTM(units=hidden_size, return_sequences=True))
    model.add(LSTM(units=hidden_size, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(total_words)))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(total_words))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=[metrics.categorical_accuracy, crossentropy, perplexity])
    return model
