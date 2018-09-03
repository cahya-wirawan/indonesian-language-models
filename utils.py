import collections
import json
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

data_path = os.path.join(os.getcwd(), 'data/LM/simple-examples/data/')


def read_words(filename):
  with tf.gfile.GFile(filename, 'r') as f:
    return f.read().replace('\n', '<eos>').split()


def build_vocab(filename):
  data = read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id, words


def file_to_word_ids(filename, word_to_id):
  data = read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
  train_path = os.path.join(data_path, 'ptb.train.txt')
  valid_path = os.path.join(data_path, 'ptb.valid.txt')

  word_to_id, wordlist = build_vocab(train_path)
  train_data = file_to_word_ids(train_path, word_to_id)
  valid_data = file_to_word_ids(valid_path, word_to_id)
  total_words = len(word_to_id)
  reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
  dictionary = {value: key for key, value in reversed_dictionary.items()}

  print('\ntotalwords : ', total_words, '\n')
  return train_data, valid_data, total_words, reversed_dictionary, dictionary, wordlist


def save_json(dictionary, filename):
  with open(filename, 'w') as fp:
    json.dump(dictionary, fp)


class BatchGenerator(object):

  def __init__(self, data, num_steps, batch_size, total_words, skip_step=5):
    self.data = data
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.total_words = total_words
    self.current_idx = 0
    self.skip_step = skip_step

  def generate(self):
      x = np.zeros((self.batch_size, self.num_steps))
      y = np.zeros((self.batch_size, self.num_steps, self.total_words))
      while True:
        for i in range(self.batch_size):
          if self.current_idx + self.num_steps >= len(self.data):
              self.current_idx = 0
          x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
          temp_y = self.data[self.current_idx +
                             1:self.current_idx + self.num_steps + 1]
          y[i, :, :] = to_categorical(
              temp_y, num_classes=self.total_words)
          self.current_idx += self.skip_step
        yield x, y

import heapq

class Beam(object):
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an
    # incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)

def beamsearch(probabilities_function, string=None, beam_width=5, clip_len=8):
    string_len = len(string)
    prev_beam = Beam(beam_width)
    prev_beam.add(0.0, False, string)
    depth = 0
    while True:
        curr_beam = Beam(beam_width)
        depth = depth + 1
        #Add complete sentences that do not yet have the best probability to the current beam,
        #the rest prepare to add more words to them.
        for (prefix_prob, complete, prefix) in prev_beam:
            if complete == True:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                #Get probability of each possible next word for the incomplete prefix.
                result = probabilities_function(prefix)
                print("prefix: {}".format(prefix))
                # print("result: {}".format(result))
                for (next_prob, next_word) in result:
                    if next_word == '<eos>':
                        #if next word is the end token then mark prefix as complete and leave out the end token
                        curr_beam.add(prefix_prob + math.log10(next_prob), True, prefix)
                    else: #if next word is a non-end token then mark prefix as incomplete
                        curr_beam.add(prefix_prob + math.log10(next_prob), False, prefix+[next_word])

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        print("dept: {}, length: {}, clip_len: {}".format(depth, len(best_prefix), clip_len))

        #if best_complete == True or len(best_prefix)-1 == clip_len:
        if ((depth >= 6) and best_complete and (len(best_prefix) != string_len)) \
                or depth == clip_len:
            # if most probable prefix is a complete sentence or has a length that
            # exceeds the clip length (ignoring the start token) then return it
            print("depth: {}".format(depth))
            return (best_prefix[0:], best_prob)
            #return best sentence without the start token and together with its probability

        prev_beam = curr_beam