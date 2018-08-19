import os
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100

# first, build index mapping words in the embeddings set
# to their embedding vector


def embedding_layer(embedding_filename='glove.6B.100d.txt', type='glove', wordlist=None):
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(embedding_filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(wordlist) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for i, word in enumerate(wordlist):
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    return embedding_layer
