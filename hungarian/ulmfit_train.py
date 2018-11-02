import html
import re
from fastai.text import *

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

LANG = 'hu'

PATH_ROOT = Path(f'lmdata/{LANG}/')
PATH_DATA = PATH_ROOT/'raw'
PATH_LM = PATH_ROOT
PATH_TMP = PATH_ROOT/'tmp'
PATH_TMP.mkdir(parents=True, exist_ok=True)

TRAIN_FILENAME = PATH_DATA/'valid'
VALID_FILENAME = PATH_DATA/'valid1000'

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)):
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
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
        tok += tok_
        labels += labels_
    return tok, labels

def data_gen(filename, name):
    LANG_TEXT = []
    for line in open(filename, encoding="utf-8"):
        LANG_TEXT.append(line)

    LANG_TEXT = pd.DataFrame(LANG_TEXT)
    LANG_TEXT.columns = ['text']
    LANG_TEXT.insert(loc=0, column='labels', value=0)

    LANG_TEXT = LANG_TEXT.assign(length = 0)
    LANG_TEXT = LANG_TEXT.assign(labels = 0).pipe(lambda x: x[['labels', 'text', 'length']])
    LANG_TEXT['length'] = LANG_TEXT['text'].str.len()
    LANG_TEXT = LANG_TEXT.sort_values(by=['length'], ascending=False)

    LANG_TEXT.columns = ['labels', 'text', 'length']
    LANG_TEXT = LANG_TEXT[LANG_TEXT['length'] > 10]
    print(LANG_TEXT.head())
    print(len(LANG_TEXT))

    return LANG_TEXT

trn_texts = data_gen(TRAIN_FILENAME, 'train')
val_texts = data_gen(VALID_FILENAME, 'valid')
trn_texts.to_csv(f"{PATH_ROOT}/emlam_{LANG}_train.csv", header=False, index=False)
val_texts.to_csv(f"{PATH_ROOT}/emlam_{LANG}_valid.csv", header=False, index=False)

chunksize = 10000
trn_texts = pd.read_csv(f"{PATH_ROOT}/emlam_{LANG}_train.csv", header=None, chunksize=chunksize)
val_texts = pd.read_csv(f"{PATH_ROOT}/emlam_{LANG}_valid.csv", header=None, chunksize=chunksize)

emlam_tok_trn, emlam_trn_labels = get_all(trn_texts, 1)
emlam_tok_val, emlam_val_labels = get_all(val_texts, 1)

# save the train and validation tokens in the tmp directories
np.save(PATH_TMP/'emlam_tok_train.npy', emlam_tok_trn)
np.save(PATH_TMP/'emlam_tok_valid.npy', emlam_tok_val)

emlam_tok_trn = np.load(PATH_TMP/'emlam_tok_train.npy')
emlam_tok_val = np.load(PATH_TMP/'emlam_tok_valid.npy')

# Get the Counter object from all the splitted files.
# Identify the most common tokens
freq = Counter(p for o in emlam_tok_trn for p in o)
freqs = pd.DataFrame.from_dict(freq, orient="index")
freqs.sort_values(0, ascending=False).head(25)

# Sanity check
print(len([p for o in emlam_tok_trn for p in o]))

cnt = []
for i in range(49):
    row_cnt = freqs[freqs[0]>=i+1].shape[0]
    cnt.append(row_cnt)

# Truncating our vocab to ignore the rare words
max_vocab = 60000
min_freq = 5

emlam_itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq] # getting rid of the rare words
emlam_itos.insert(0, '_pad_') #
emlam_itos.insert(0, '_unk_') # wiki_itos is the list of all the strings in the vocab

# creating a index-key dictionary for our vocabulary
emlam_stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(emlam_itos)})
len(emlam_itos)

# creating a index representation for our train and validation dataset
emlam_trn_lm = np.array([[emlam_stoi[o] for o in p] for p in emlam_tok_trn])
emlam_val_lm = np.array([[emlam_stoi[o] for o in p] for p in emlam_tok_val])

# saving our indexed representation of our dataset to disk
# we also save the index-word mapping to retrieve the complete text representation from these numpy arrays
np.save(PATH_TMP/'emlam_trn_ids.npy', emlam_trn_lm)
np.save(PATH_TMP/'emlam_val_ids.npy', emlam_val_lm)
pickle.dump(emlam_itos, open(f'{PATH_ROOT}/emlam_{LANG}_itos.pkl', 'wb'))

# Loading the indexed representation of our dataset from disk
# we also load the index-word mapping to to help us convert the indexes to word datasets, if need be.
trn_lm = np.load(PATH_TMP/'emlam_trn_ids.npy')
val_lm = np.load(PATH_TMP/'emlam_val_ids.npy')
emlam_itos = pickle.load(open(f'{PATH_ROOT}/emlam_{LANG}_itos.pkl', 'rb'))

# checking vocabulary size
vs=len(emlam_itos)
word_count = 0
for i in trn_lm:
    word_count += len(trn_lm[i])

print(f'vocabulary size: {vs}, #articles: {len(trn_lm)}, #words: {word_count}')

em_sz,nh,nl = 400,1150,3

wd=1e-7
bptt=70
#bs=52
bs=40
#opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
opt_fn = partial(optim.SGD, momentum=0.9)
weight_factor = 0.3

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*weight_factor

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH_ROOT, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

learner= md.get_model(opt_fn, em_sz, nh, nl,
                      dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.clip = 0.2
learner.unfreeze()

#print(learner.summary)
#learner.lr_find2(num_it=1000)
#learner.sched.plot()
lr = 8
learner.fit(lr, 1, wds=wd, cycle_len=15, use_clr=(10,33,0.95,0.85), best_save_name=f'emlam_{LANG}_1cycle_best')
learner.save(f'emlam_{LANG}_lm')
learner.save_encoder(f'emlam_{LANG}_lm_enc')

