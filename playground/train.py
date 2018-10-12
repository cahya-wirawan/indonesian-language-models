import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import CONFIG
from keras_model import create_model
from utils import BatchGenerator, load_data, save_json

train_data, valid_data, total_words, reversed_dictionary, dictionary, wordlist = load_data()

train_data_generator = BatchGenerator(
    train_data, CONFIG.num_steps, CONFIG.batch_size, total_words, skip_step=CONFIG.num_steps)
valid_data_generator = BatchGenerator(
    valid_data, CONFIG.num_steps, CONFIG.batch_size, total_words, skip_step=CONFIG.num_steps)

optimizer = Adam(lr=CONFIG.learning_rate, decay=CONFIG.learning_rate_decay)

model = create_model(total_words=total_words, hidden_size=CONFIG.hidden_size,
                     num_steps=CONFIG.num_steps, optimizer='adam', wordlist=wordlist)

print(model.summary())

checkpointer = ModelCheckpoint(filepath=os.path.join(
    os.getcwd(), 'model', 'checkpoint', 'model-{epoch:02d}.h5'), verbose=1)

save_json(dictionary, os.path.join(
    os.getcwd(), 'web', 'web_model', 'dictionary.json'))

save_json(reversed_dictionary, os.path.join(
    os.getcwd(), 'web', 'web_model', 'reversed-dictionary.json'))

model.fit_generator(
    generator=train_data_generator.generate(),
    steps_per_epoch=len(train_data)//(CONFIG.batch_size*CONFIG.num_steps),
    epochs=CONFIG.num_epochs,
    validation_data=valid_data_generator.generate(),
    validation_steps=len(valid_data)//(CONFIG.batch_size*CONFIG.num_steps),
    callbacks=[checkpointer],
)

model.save(os.path.join(os.getcwd(), 'model', "model-{}.h5".format(CONFIG.num_steps)))
