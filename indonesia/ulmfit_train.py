import json
import pathlib

from fastai.text import *

import numpy as np
import pandas as pd

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH = pathlib.Path("lm-data/wiki_extr/id")
LM_PATH=Path('lm-data/id/lm/')
LM_PATH.mkdir(parents=True, exist_ok=True)

LANG_FILENAMES = [str(f) for f in PATH.rglob("*/*")]
print(len(LANG_FILENAMES))
print(LANG_FILENAMES[0:5])

LANG_TEXT = []
for i in LANG_FILENAMES:
    for line in open(i, encoding="utf-8"):
        LANG_TEXT.append(json.loads(line))

LANG_TEXT = pd.DataFrame(LANG_TEXT)
LANG_TEXT.to_csv("{}/Wiki_Indonesia_Corpus.csv".format(LM_PATH), index=False)
LANG_TEXT = pd.read_csv("{}/Wiki_Indonesia_Corpus.csv".format(LM_PATH))

(LANG_TEXT.assign(labels = 0).pipe(lambda x: x[['labels', 'text']])
 .to_csv("{}/Wiki_Indonesia_Corpus2.csv".format(LM_PATH), header=None, index=False))

# Some statistics of Indonesia Wikipedia
### Getting rid of the title name in the text field
def split_title_from_text(text):
    words = text.split("\n\n")
    if len(words) >= 2:
        return ''.join(words[1:])
    else:
        return ''.join(words)

LANG_TEXT['text'] = LANG_TEXT['text'].apply(lambda x: split_title_from_text(x))

### Number of documents
print(LANG_TEXT['text'][:5])
print(LANG_TEXT.shape)

### Number of words in all the documents
print(LANG_TEXT['text'].apply(lambda x: len(x.split(" "))).sum())

# ### Number of unique tokens across documents
print(len(set(''.join(LANG_TEXT['text'].values).split(" "))))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = '\n{} {} 1 '.format(BOS, FLD) + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += ' {} {} '.format(FLD, i-n_lbls) + df[i].astype(str)
    #texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts)) # splits the list into sublists for processing by each core
    # Lower and upper case is inside the tokenizer
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        #pdb.set_trace()
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

LANG_TEXT = pd.read_csv("{}/Wiki_Indonesia_Corpus2.csv".format(LM_PATH), header=None)#, chunksize=5000)


print(LANG_TEXT.head())
print(LANG_TEXT.shape)

trn_texts,val_texts = sklearn.model_selection.train_test_split(
    LANG_TEXT, test_size=0.1) # split the data into train and validation sets

np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts)) # generate a random ordering
val_idx = np.random.permutation(len(val_texts))

df_trn = trn_texts.iloc[trn_idx,:] # sort things randomly
df_val = val_texts.iloc[val_idx,:] # sort things randomly

df_trn.columns = ['labels', 'text']
df_val.columns = ['labels', 'text']

df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH/'test.csv', header=False, index=False) # saving the data in our new format to disk


chunksize = 10000
df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

# create a tmp directory to store the upcoming numpy arrays
(LM_PATH/'tmp').mkdir(exist_ok=True)

# save the train and validation tokens in the tmp directories
np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)

print("Trn:", tok_trn[:2], "\n")
print("Val:", tok_val[:2])

tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')

# Identify the most common tokens and numericalizing the text
freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

# Truncating our vocab to ignore the rare words
max_vocab = 60000
min_freq = 5

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq] # getting rid of the rare words
itos.insert(0, '_pad_') #
itos.insert(0, '_unk_') # itos is the list of all the strings in the vocab

# creating a index-key dictionary for our vocabulary
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)

# creating a index representation for our train and validation dataset
trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

# saving our indexed representation of our dataset to disk
# we also save the index-word mapping to retrieve the complete text representation from these numpy arrays
np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))

# Loading the indexed representation of our dataset from disk
# we also load the index-word mapping to to help us convert the indexes to word datasets, if need be.
trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))

# checking vocabulary size
vs=len(itos)
vs,len(trn_lm)


# # Model Setup
# ! wget -nH -r -np http://files.fast.ai/models/wt103/
# mv models/ {LM_PATH}

em_sz,nh,nl = 400,1150,3

PRE_PATH = LM_PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

# itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb')) # mapping the itos from wiki to our own mapping
# stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# In[9]:


# we train from scratch so these are unused
# wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

# enc_wgts = to_np(wgts['0.encoder.weight'])
# row_m = enc_wgts.mean(0)

# wgts['0.encoder.weight'] = T(new_w)
# wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
# wgts['1.decoder.weight'] = T(np.copy(new_w))


# # Language Model

# In[17]:


wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


# In[18]:


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)


# In[19]:


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7 # if you're overfitting, increase this. Underfitting? decrease this.


# In[20]:


learner= md.get_model(opt_fn, em_sz, nh, nl,
                      dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.clip = 0.2
learner.unfreeze()


# In[244]:


lr=1e-3
lrs = lr


# In[55]:


learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1) # last layer is the embedding weights


# In[56]:


learner.save('lm_indonesia_v2')


# In[22]:


learner.load('lm_indonesia_v2')


# In[58]:


learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)


# In[ ]:


learner.sched.plot()


# In[ ]:


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=1)


# In[ ]:


learner.save('lm_indonesia_v2_2')


# In[ ]:


learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)


# In[ ]:


learner.save('lm_indonesia_v2_3')


# In[ ]:


learner.save_encoder('lm_indonesia_v2_3_enc')


# ### Generate text

# In[40]:


learner.load("lm_indonesia_v2_3")


# In[41]:


m = learner.model
m.eval()
m[0].bs = 1


# In[173]:


print("Training is done")


# ## Inference

# In[174]:


sen = """saya ucapkan terima"""


# In[169]:


idxs = np.array([[stoi[p] for p in sen.strip().split(" ")]])
idxs


# In[170]:


VV(idxs)


# In[46]:


probs = learner.model(VV(idxs))


# In[172]:


type(probs), len(probs)
print(probs)


# In[49]:


learner.model


# In[50]:


probs[0].shape, [x.shape for x in probs[1]], [x.shape for x in probs[2]]


# In[30]:


# probs[0] is most likely the output vector


# ### Arvind's answer

# In[74]:


def get_next_0(inp):
    #     m[0].bs = 1 ## why?
    idxs = np.array([[stoi[p] for p in inp.strip().split(" ")]])
    p = m(VV(idxs))
    #pdb.set_trace()
    i = np.argmax(to_np(p)[0], 1)[0]
    try:
        r = itos[i]
    except:
        r = "oor"
    return r

def get_next_1(inp):
    idxs = np.array([[stoi[p] for p in inp.strip().split(" ")]])
    p = m(VV(idxs))
    #i = np.argmax(to_np(p)[0], 1)[0]
    i = torch.topk(p[0][-1], 1)[1].data[0]
    try:
        r = itos[i]
    except:
        r = "oor"
    return r

def get_next_2(inp):
    m[0].bs =1
    #print(inp)
    idxs = np.array([[stoi[p] for p in inp.strip().split(" ")]])
    probs = m(VV(idxs))
    encc = probs[-1][-1][-1][-1].squeeze()
    pred = to_np(learner.model[1].decoder(encc).exp()).argmax()
    try:
        r = itos[pred]
    except:
        r = "oor"
    return r



# In[249]:


def get_next_4(inp, pos):
    idxs = np.array([[stoi[p] for p in inp.strip().split(" ")]])
    p = m(VV(idxs))
    # torch.topk(p[-1], 3)
    top = torch.topk(p[0][-1], 10)
    #print(top)
    #print(to_np(top))
    #print(to_np(nn.Softmax()(p[0][-1])).shape)
    i = top[1].data[pos]
    #print(to_np(p[0][-1][i]))
    #print("pos:", pos)
    try:
        r = itos[i]
    except:
        r = "oor"
    return r


# In[251]:


def get_next_n(inp, n, pos):
    res = inp
    for i in range(n):
        c = get_next_4(inp, pos)
        # res += c # ???
        res = res + " " + c
        #print(res)
        inp = inp.strip().split(" ") + [c]
        #         inp = ' '.join(inp[1:])
        inp = ' '.join(inp)

    return res



m = learner.model
m.eval()
m[0].bs = 1


# In[ ]:


string = "ibu saya masak sayur"


# In[ ]:


(sentences, probability) = beamsearch(get_next_word, string.split(" "))
print()
print("prob: {}, sentence: {}".format(probability, sentences))


# In[250]:


sen = """tikus adalah"""
for i in range(10):
    print(get_next_n(sen, 10, i))


# In[176]:


sen = """kemarin ibu saya masak sayur"""
for i in range(10):
    print(get_next_n(sen, 10, i))


# In[55]:


sen = """సౌకర్యం కూడా"""
get_next_n(sen, 10)


# # Classifier Tokens

# In[56]:


CLAS_PATH = Path("lm/telugu/telugu_clas/")
LM_PATH.mkdir(exist_ok=True)


# In[57]:


df_clas_data = pd.read_csv(CLAS_PATH/"ACTSA_telugu_polarity_annotated_UTF.txt", sep="\t", header=None)
df_clas_data[1] = df_clas_data[0].str[2:]
df_clas_data[0] = df_clas_data[0].str[0:2]

# Cleaning the target
df_clas_data[0] = df_clas_data[0].str.strip()
df_clas_data = df_clas_data[df_clas_data[0] != '+'].reset_index(drop=True)
df_clas_data[0] = df_clas_data[0].astype(np.float32)

df_clas_data.to_csv(CLAS_PATH/"Telugu_Sentiment_Data.csv", index=False)

# Ignoring neutral class for this exercise
df_clas_data = df_clas_data[df_clas_data[0] != 0].reset_index(drop=True)

# Creating train and validation sets
np.random.seed(42)
trn_keep = np.random.rand(len(df_clas_data))>0.1
df_trn = df_clas_data[trn_keep]
df_val = df_clas_data[~trn_keep]

# Saving train and validation sets to disk
df_trn.to_csv(CLAS_PATH/"Telugu_Sentiment_Data_Train.csv", header=None, index=False)
df_val.to_csv(CLAS_PATH/"Telugu_Sentiment_Data_Test.csv", header=None, index=False)

len(df_trn),len(df_val)


# In[58]:


chunksize = 10000
df_trn = pd.read_csv(CLAS_PATH/"Telugu_Sentiment_Data_Train.csv", header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/"Telugu_Sentiment_Data_Test.csv", header=None, chunksize=chunksize)


# In[59]:


tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)


# In[60]:


(CLAS_PATH/'tmp').mkdir(exist_ok=True)

np.save(CLAS_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(CLAS_PATH/'tmp'/'tok_val.npy', tok_val)

np.save(CLAS_PATH/'tmp'/'trn_labels.npy', trn_labels)
np.save(CLAS_PATH/'tmp'/'val_labels.npy', val_labels)


# In[61]:


tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')


# In[62]:


itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


# In[63]:


trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])


# In[64]:


np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)


# # Classifier

# In[65]:


trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')


# In[66]:


trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))


# In[67]:


Counter(trn_labels)


# In[68]:


bptt,em_sz,nh,nl = 70,400,1150,3
vs = len(itos)
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
bs = 48


# In[69]:


min_lbl = trn_labels.min()
trn_labels -= min_lbl
val_labels -= min_lbl
c=int(trn_labels.max())+1


# In[70]:


c


# In[71]:


trn_ds = TextDataset(trn_clas, trn_labels)
val_ds = TextDataset(val_clas, val_labels)
trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
md = ModelData(PATH, trn_dl, val_dl)


# In[72]:


dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])


# In[73]:


m = get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                      layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
                      dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])


# In[74]:


opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# In[75]:


m


# In[ ]:


# learner= md.get_model(opt_fn, em_sz, nh, nl,
#     dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

# learner.metrics = [accuracy]
# learner.unfreeze()


# In[76]:


learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learn.clip=25.
learn.metrics = [accuracy]


# In[77]:


lr=3e-3
lrm = 2.6
lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])


# In[78]:


learner


# In[79]:


wd = 1e-7
wd = 0
learn.load_encoder('lm_telugu_v2_3_enc')


# In[80]:


learn.freeze_to(-1)


# In[81]:


learn.lr_find(lrs/1000)


# In[82]:


learn.sched.plot()


# In[83]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))


# In[84]:


learn.save('clas_0')


# In[85]:


learn.load('clas_0')


# In[86]:


learn.freeze_to(-2)


# In[87]:


learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8, 3))


# In[88]:


learn.unfreeze()


# In[89]:


learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32, 10))


# In[90]:


learn.save('clas_1')