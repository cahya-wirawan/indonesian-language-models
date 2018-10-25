import os

import numpy as np
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import (LSTM, Activation, Dense, Dropout, Flatten,
                                     Embedding, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from datasets import Imdb

import CONFIG
from utils import BatchGenerator, load_data, beamsearch

_, _, _, indexToString, stringToIndex, _ = load_data()


num_class = 2
batch_size = 256
maxlen = 100

model = load_model(os.path.join(os.getcwd(), 'model', 'model-100.h5'))

for i, layer in enumerate(model.layers):
    print(i, layer, layer.trainable)

model.pop()
model.add(Activation('relu', name='activation_new_01'))
model.add(Dropout(0.6, name='dropout_new_01'))
model.add(Dense(100, name='dense_new_01'))
model.add(Activation('relu', name='activation_new_02'))
model.add(Dropout(0.6, name='dropout_new_02'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid', name='dense_new_02'))

#for layer in model.layers[:8]:
#    layer.trainable = False

for i, layer in enumerate(model.layers):
    print("After:", i, layer, layer.trainable)

model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
print(model.summary())

imdb = Imdb()
(train_x, train_y), (test_x, test_y) = imdb.load_data()
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                      test_size = 0.1, random_state = 2)

print('x_train shape:', train_x.shape)
print('x_test shape:', test_x.shape)

print('Train...')
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=100,
          verbose=2,
          validation_data=(valid_x, valid_y))
score, acc = model.evaluate(test_x, test_y,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)