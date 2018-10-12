import numpy as np
import pandas as pd
#from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
from pathlib import Path
from utils import BatchGenerator
from data_utils import DataGenerator
from keras_model import create_model

DATA_TRAIN = Path("lmdata/imdb_clas/train_small.csv")
DATA_TEST = Path("lmdata/imdb_clas/test_small.csv")

# Parameters
params = {'batch_size': 64,
          'n_classes': 2,
          'max_len': 100,
          'n_words': 1000,
          'shuffle': False}

learning_rate = 0.5

df_train = pd.read_csv(DATA_TRAIN, chunksize=100)
df_valid = pd.read_csv(DATA_TEST, chunksize=100)

# Generators

training_generator = DataGenerator(df_train, **params)
validation_generator = DataGenerator(df_valid, **params)

print("training_generator", isinstance(training_generator, Sequence))
print("validation_generator", isinstance(validation_generator, Sequence))
# Design model
model = create_model(total_words=1000, hidden_size=128,
                     num_steps=100, optimizer='adam')

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
