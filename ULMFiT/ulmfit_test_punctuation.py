from fastai.text import *

import numpy as np
from utils import beamsearch, beamsearch_punctuation

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

# VERSION = '0.2'
LANG = 'id'
LM_PATH = Path(f'lmdata_0.2/{LANG}/')
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

punctuations = ['.', ',', '?']

def get_next_words(string, next_word):
    idxs = np.array([[stoi[p] for p in string]])
    # print("<------- " + " ".join(string) + " ------->")
    seq_rnn.reset()
    t = LongTensor(idxs).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    pred,*_ = seq_rnn(t)
    probabilities = F.softmax(pred[-1])
    print(probabilities.shape)
    if next_word != '':
        print("next word:{} - {:.4f}".format(next_word, probabilities.data[stoi[next_word]]))
    for i in punctuations:
        print("punct.:{} - {:.4f}".format(i, probabilities.data[stoi[i]]))

    return np.log(probabilities.data[stoi[next_word]])

probability_threshold = 0.01

def get_punctuation(string, next_word, punctuations):
    idxs = np.array([[stoi[p.lower()] for p in string]])
    seq_rnn.reset()
    t = LongTensor(idxs).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    pred,*_ = seq_rnn(t)
    probabilities = F.softmax(pred[-1])

    best_predictions = []
    if next_word is not None and (string[-1] not in punctuations or next_word not in punctuations):
        probability = probabilities.data[stoi[next_word]]
        #print("next word:{} - {:.4f}".format(next_word, probability))
        best_predictions.append((probability, next_word))
    if string[-1] not in punctuations and next_word not in punctuations:
        for p in punctuations:
            probability = probabilities.data[stoi[p]]
            if probability >= probability_threshold:
                best_predictions.append((probability, p))

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
        if word != 'xbos' and word != 'xfld':
            result.append(word)
        else:
            break
        pred,*_ = seq_rnn(pred_i[0].unsqueeze(0))

    result = re.sub('\s+([.,])', r'\1', "{} {}".format(ss, " ".join(result).rstrip()))
    return(result)

BS = True


def tokenize(string, punctuations):
    for p in punctuations:
        s = string.split(p)
        string = f' {p} '.join(s)
    tokens = string.split()
    return tokens

if BS:
    while True:
        string=input('\n\nEnter at least 2 words: \n')
        seq_rnn.reset()
        string=tokenize(string, punctuations)
        if len(string) < 3:
            continue
        for word in string:
            if stoi[word.lower()] == 0:
                print(f'OOV {word.lower()}')
        result = beamsearch_punctuation(get_punctuation, punctuations, string=string, beam_width=5)
        for i, sentence in enumerate(result.best_sentences(complete=False)):
            print(f'{i+1}. {sentence}')
else:
    while True:
        string = input('\n\nEnter at least 3 words: \n')
        string_list = tokenize(string, punctuations)
        prob = 0.0
        for i in range(len(string_list)-1):
            prob += get_next_words(string_list[:i+1], string_list[i+1])
            print(prob)
        print(get_next_words(string_list, ''))
        #print(gen_text(string, 10))
        #print(generate_sentence(string, 70))
