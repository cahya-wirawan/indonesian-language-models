import json
import pathlib

from fastai.text import *

import numpy as np
import pandas as pd
import utils

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH = pathlib.Path("lm-data/wiki_extr/id")
LM_PATH=Path('lm-data/id/lm/')

# Truncating our vocab to ignore the rare words
max_vocab = 60000
min_freq = 5

# Loading the indexed representation of our dataset from disk
# we also load the index-word mapping to to help us convert the indexes to word datasets, if need be.
trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))

# creating a index-key dictionary for our vocabulary
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

# checking vocabulary size
vs=len(itos)
print(vs,len(trn_lm))

em_sz,nh,nl = 400,1150,3

PRE_PATH = LM_PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7 # if you're overfitting, increase this. Underfitting? decrease this.

learner= md.get_model(opt_fn, em_sz, nh, nl,
                      dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.load("lm_indonesia_v2_3")

m = learner.model
m.eval()
m[0].bs = 1

def get_next_word(string, predictions_number=10):
    idxs = np.array([[stoi[p] for p in string]])

    prediction = m(VV(idxs))
    probabilities = F.softmax(prediction[0][-1])
    top = torch.topk(probabilities, predictions_number)
    best_predictions = []
    for i in range(predictions_number):
        probability = top[0].data[i]
        word = itos[top[1].data[i]]
        if word == '_unk_':
            probability = 1e-10
        best_predictions.append((probability, word))
    return best_predictions


while True:
    string=input('\n\nEnter atleast 3 words: \n')
    (sentences, probability) = utils.beamsearch(get_next_word, string.strip().split(" "))
    print()
    print("prob: {}, sentence: {}".format(probability, " ".join(sentences)))
