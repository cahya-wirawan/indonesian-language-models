import os

from numpy import argmax
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from keras_model import perplexity

import CONFIG
from utils import BatchGenerator, load_data, beamsearch

_, _, _, indexToString, stringToIndex, _ = load_data()


model = load_model(os.path.join(os.getcwd(), 'model', 'model-100.h5'),
                   custom_objects={'perplexity': perplexity})

for layer in model.layers:
    print(layer, layer.trainable)

def predict_next_word(string, verbose=True, NUMBER_OF_PREDICTIONS=10):
  ques_bool = False
  idx, ques_bool = string_to_indexes(string.split(), ques_bool)
  eos = stringToIndex['<eos>']
  idx = [eos]*(CONFIG.num_steps - len(idx)) + idx

  if len(idx) >= 3:
    if verbose:
      print('\nindexes of last 3 words\t:', idx[-CONFIG.num_steps:])

    prediction = model.predict([[idx[-CONFIG.num_steps:]]])

    best_predictions = []

    for _ in range(NUMBER_OF_PREDICTIONS):
      argmax_idx = argmax(prediction[:, CONFIG.num_steps - 1, :])
      best_predictions.append(argmax_idx)
      prediction[:, CONFIG.num_steps - 1, argmax_idx] = 0.0

    if verbose:
      print('\nprediction indexes\t:', best_predictions)
    converted_string = indexes_to_string(best_predictions, ques_bool)
    sentences = []

    for word in converted_string:
      sentences.append(string + ' ' + word)
    return sentences
  else:
    print('\n\nPlease enter atleast 3 words.\n')



def get_next_word(string, verbose=False, NUMBER_OF_PREDICTIONS=10):
    ques_bool = False
    idx, ques_bool = string_to_indexes(string, ques_bool)
    # idx = [stringToIndex[s] for s in string]
    eos = stringToIndex['<eos>']
    idx = [eos]*(CONFIG.num_steps - len(idx)) + idx

    if len(idx) >= 3:
        if verbose:
            print('\nindexes of last 3 words\t:', idx[-CONFIG.num_steps:])

        prediction = model.predict([[idx[-CONFIG.num_steps:]]])
        best_predictions = []
        for _ in range(NUMBER_OF_PREDICTIONS):
            argmax_idx = argmax(prediction[:, CONFIG.num_steps - 1, :])
            word = indexToString[argmax_idx]
            probability = prediction[:, CONFIG.num_steps - 1, argmax_idx][0]
            if word == '<unk>':
                probability = 1e-10
            best_predictions.append((probability, word))
            prediction[:, CONFIG.num_steps - 1, argmax_idx] = 0.0

        return best_predictions
    else:
        print('\n\nPlease enter atleast 3 words.\n')
        return None

def string_to_indexes(array_of_string, ques_bool):
  array_of_indexes = []

  for word in array_of_string:
    if word == '<rare word>':
      word = '<unk>'
    if word == '.' or word == '?':
      word = '<eos>'
    if word == 'what' or word == 'why' or word == 'who' or word == 'how' or word == 'whose' or word == 'when' or word == 'which' or word == 'where':
      ques_bool = True

    try:
      array_of_indexes.append(stringToIndex[word])
    except:
      print('Word ', word, ' does not exist')
      word = '<unk>'
      array_of_indexes.append(stringToIndex[word])
      pass
  return array_of_indexes, ques_bool


def indexes_to_string(array_of_indexes, ques_bool):
  array_of_strings = []

  for index in array_of_indexes:
    word = indexToString[index]
    if word == '<eos>':
      if ques_bool == True:
        word = '?'
      else:
        word = '.'
    if word == 'N':
      #TODO
      pass
    array_of_strings.append(word)
  return array_of_strings


while True:
  string=input('\n\nEnter atleast 3 words: \n')
  (sentences, probability) = beamsearch(get_next_word, string.split(" "))
  print()
  print("prob: {}, sentence: {}".format(probability, sentences))
