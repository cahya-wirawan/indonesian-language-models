import numpy as np
import pandas as pd
#from keras.utils import Sequence
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from pathlib import Path
from utils import BatchGenerator
from data_utils import DataGenerator, tokenize
from keras_model import create_model
from sklearn import model_selection, metrics
from keras.preprocessing.sequence import pad_sequences

DATA_TRAIN = Path("lmdata/imdb_clas/train.csv")
DATA_TEST = Path("lmdata/imdb_clas/test.csv")

def create_cnn(total_words=1000, embedded_dimension=300,
               embedding_matrix=None, input_length=100, optimizer='adam'):
    # Add an Input Layer
    input_layer = layers.Input((input_length, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(total_words + 1, embedded_dimension,
                                       weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.5)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.6)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

# Parameters
params = {'batch_size': 256,
          'n_classes': 2,
          'max_len': 200,
          'n_words': 20000,
          'shuffle': True}

learning_rate = 0.5


df_train = pd.read_csv(DATA_TRAIN, names=['label','text'])
df_test = pd.read_csv(DATA_TEST, names=['label','text'])

tokenizer = tokenize(df_train['text'].values, n_words=params['n_words'])
train_x, valid_x, train_y, valid_y = \
    model_selection.train_test_split(df_train['text'].values, df_train['label'].values)


# Generators

training_generator = DataGenerator(train_x, train_y, tokenizer, **params)
valid_generator = DataGenerator(valid_x, valid_y, tokenizer,  **params)
#test_generator = DataGenerator(df_test, **params)


# load the pre-trained word-embedding vectors
embeddings_index = {}
for i, line in enumerate(open('lmdata/wiki-news-300d-1M.vec', encoding='utf8')):
    if i%10000 == 0:
        print(i)
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create token-embedding mapping
word_index = training_generator.tokenizer.word_index
embedding_matrix = np.zeros((params['n_words'] + 1, 300))
for word, i in word_index.items():
    if i>=params['n_words']:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Design model
model = create_cnn(total_words=params['n_words'], embedded_dimension=300,
                   embedding_matrix=embedding_matrix, input_length=params['max_len'], optimizer='adam')

print(model.summary())
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=valid_generator,
                    use_multiprocessing=True,
                    workers=6, epochs=20)


sequences = tokenizer.texts_to_sequences(df_test['text'].values)
X = pad_sequences(sequences, maxlen=params['max_len'])

predictions = model.predict(X)
predictions = np.round(predictions).astype(int).reshape(-1)
score = metrics.accuracy_score(predictions, df_test['label'].values)
print("Score: ", score)
