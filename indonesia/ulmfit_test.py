from fastai.text import *

import numpy as np
from utils import beamsearch

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

VERSION = '0.2'
LANG = 'id'
LM_PATH = Path(f'lmdata_{VERSION}/{LANG}/')
LM_PATH_MODEL = LM_PATH/'models/wiki_id_lm.h5'
LM_PATH_ITOS = LM_PATH/'wiki_id_itos.pkl'

# Loading the index-word mapping to to help us convert the indexes to word datasets, if need be.
itos = pickle.load(open(LM_PATH_ITOS, 'rb'))

# creating a index-key dictionary for our vocabulary
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

# checking vocabulary size
vs=len(itos)

em_sz,nh,nl = 400,1150,3

seq_rnn = get_language_model(vs, em_sz, nh, nl, 1)
lm = LanguageModel(to_gpu(seq_rnn))
load_model(seq_rnn, LM_PATH_MODEL)

seq_rnn.eval()
seq_rnn[0].bs = 1


def get_next_word(string, predictions_number=10):
    idxs = np.array([[stoi[p] for p in string]])
    # print("<------- " + " ".join(string) + " ------->")
    seq_rnn.reset()
    t = LongTensor(idxs).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    pred,*_ = seq_rnn(t)
    probabilities = F.softmax(pred[-1])

    #prediction = seq_rnn(VV(idxs))
    #probabilities = F.softmax(prediction[0][-1])

    top = torch.topk(probabilities, predictions_number)
    best_predictions = []
    for i in range(predictions_number):
        probability = top[0].data[i]
        word = itos[top[1].data[i]]
        if word == '_unk_':
            probability = 1e-10
        best_predictions.append((probability, word))
    return best_predictions

class LongTensor(torch.LongTensor):
    def __init__(self, *args, **kwargs):
        pass

def gen_sentences(ss,nb_words):
    result = []
    s = ss.strip().split(" ")
    t = LongTensor([stoi[i] for i in s]).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    seq_rnn.reset()
    pred,*_ = seq_rnn(t)
    for i in range(nb_words):
        pred_i = pred[-1].topk(2)[1]
        pred_i = pred_i[1] if pred_i.data[0] < 2 else pred_i[0]
        result.append(itos[pred_i.data[0]])
        pred,*_ = seq_rnn(pred_i[0].unsqueeze(0))
    return(result)

def gen_text(ss,topk):
    #s = word_tokenize(ss,engine='newmm')
    s = ss.strip().split(" ")
    t = LongTensor([stoi[i] for i in s]).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    seq_rnn.reset()
    pred,*_ = seq_rnn(t)
    pred_i = torch.topk(pred[-1], topk)[1]
    return [itos[o] for o in to_np(pred_i)]

def generate_sentence(ss, nb_words):
    result = []
    s = ss.strip().split(" ")
    t = LongTensor([stoi[i] for i in s]).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    seq_rnn.reset()
    pred,*_ = seq_rnn(t)
    for i in range(nb_words):
        pred_i = pred[-1].topk(2)[1]
        pred_i = pred_i[1] if pred_i.data[0] < 2 else pred_i[0]
        word = itos[pred_i.data[0]]
        if word != 'xbos':
            result.append(word)
        else:
            break
        pred,*_ = seq_rnn(pred_i[0].unsqueeze(0))

    result = re.sub('\s+([.,])', r'\1', "{} {}".format(ss, " ".join(result).rstrip()))
    return(result)

BS = False

if BS:
    while True:
        string=input('\n\nEnter at least 2 words: \n')
        seq_rnn.reset()
        result = beamsearch(get_next_word, string=string.strip().split(" "),
                            beam_width=10, length_min=10, length_max=100)

        #print("prob: {}, sentence: {}".format(probability, " ".join(sentences)))
        for sentence in result.best_sentences(complete=True):
            print("{}".format(sentence))
else:
    while True:
        string=input('\n\nEnter at least 3 words: \n')
        #print(gen_text(string, 10))
        print(generate_sentence(string, 70))
