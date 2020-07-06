# Transformer based Language Models for Indonesian

The transformer-based language model has been very popular since 2018, and there are already pre-trained models 
in many different languages. Unfortunately there is still no transfomer indonesian language model available for public 
(May 2020), maybe due to huge resources needed  for the training from scratch. There are already some Malaysian 
language models, but Malaysian and Indonesian still differ, even though they both belong to the same Austronesian 
language. For this reason, I try to train some language models myself and make them available to the public in 
the hope that they can be useful for Indonesian NLP research.

A documentation and a better GPT2 Language Model trained with more indonesian datasets (indonesian corpus from 
[OSCAR dataset](https://oscar-corpus.com/) is already in the plan) will follow. If you have dataset or just 
any suggestion which datasets I could use to train, just let me know and I will try to compile all datasets 
and to find hardware resources to train it. Thanks.

## GPT-2

### 1. GPT-2 small with indonesian Wikipedia
This is my first attempt to create a small indonesian GPT2 Language Model. GPT-2 small has 122M parameters. It was 
trained only with dataset from indonesian Wikipedia (around 522MB) for 5 hours. It is hosted at huggingface:
[gpt2-small-indonesian-522M](https://huggingface.co/cahya/gpt2-small-indonesian-522M).

### 2. GPT-2 small with indonesian Wikipedia and OSCAR Corpus
The OSCAR Corpus for the deduplicated indonesian dataset is about 16GB. The training is on the plan.

### 3. GPT-2 XL with indonesian Wikipedia and OSCAR Corpus
GPT-2 XL has 1.5B parameters. It's just too huge to train it on my few years old single NVidia 1080 GPU, so *ayo patungan 
buat nyewa TPU di Google Cloud buat training nya ;-)*

## BERT
### 1. BERT-base with indonesian Wikipedia
[bert-base-indonesian-522M](https://huggingface.co/cahya/bert-base-indonesian-522M).

## RoBERTa
### 1. RoBERTa-base with indonesian Wikipedia
[roberta-base-indonesian-522M](https://huggingface.co/cahya/roberta-base-indonesian-522M).

## Reformer

## BART

## LONGFORMER

## T5


## XLM-RoBERTa

## ELECTRA

## Linformer

## Usage

### Language Generation
The Jupyter notebook: [gpt2-indonesian.ipynb](https://github.com/cahya-wirawan/language-modeling/blob/master/Transformers/GPT2/gpt2-indonesian.ipynb)
```
import torch
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("cahya/gpt2-small-indonesian-522M", pad_token_id=tokenizer.eos_token_id)

input_sentences = [
    'Alkisah pada jaman dahulu kala seekor babi tengah melintas di sebuah hutan belantara. Babi hutan itu sedang merasa kehausan di tengah panasnya terik matahari',
    'Cirebon adalah sebuah kota kecil di Jawa Barat dengan keindahannya yang melebihi kota Bandung',
    'Sriwijaya adalah salah satu kemaharajaan bahari yang pernah berdiri di pulau Sumatra dan banyak memberi pengaruh di Nusantara dengan daerah kekuasaan yang luas',
    'Pantai berpasir putih ini cukup populer akhir-akhir ini karena menawarkan pemandangan yang begitu eksotis, indah dan mempesona',  
]

# We set the seed manualy for reproducible result
for i, sentence in enumerate(input_sentences):
    torch.manual_seed(1)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    sample_output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=150, 
        top_k=50, 
        top_p=0.95
    )
    print("Output:\n" + 100 * '-')
    print("{}: {}".format(i, tokenizer.decode(sample_output[0], skip_special_tokens=True)))
```
Following ist the result

```
Output:
----------------------------------------------------------------------------------------------------
0: Alkisah pada jaman dahulu kala seekor babi tengah melintas di sebuah hutan belantara. Babi hutan itu sedang 
merasa kehausan di tengah panasnya terik matahari, karena itu ia pergi ke tengah hutan untuk mencari makanan. 
Kemudian ia kembali ke istana dan bertemu dengan seekor musang bernama Sang Hyang (Himala). Sang Hyang pun 
berjanji akan membiarkannya hidup sendirian. Sang Hyang pun pergi ke hutan untuk mencari makanannya. 
Setelah sampai di tengah hutan, Sang Hyang bertanya kepada istrinya, “ Wahai para dewa, aku akan menemukan 
makanan dan air yang dapat memenuhi kebutuhanmu. ” Sang Hyang kemudian menjelaskan mengapa sang dewi merasa 
haus darah dan kembali tertidur. Sang dewi marah dan tak sadarkan diri saat terbangun, tetapi Sang Hyang 
merasa ketakutan dan tertawa. Sang Hyang berteriak, ” "Bagaimana jika ia berada pada

Output:
----------------------------------------------------------------------------------------------------
1: Cirebon adalah sebuah kota kecil di Jawa Barat dengan keindahannya yang melebihi kota Bandung, seperti 
kota-kota di Jawa Barat yang lainnya. Di Kota Cirebon, kota ini menjadi pusat perdagangan antaretnis 
antarabangsa Asia-Arab maupun Arab-Indonesia. Selain itu Kota Cirebon dikenal sebagai kota industri.
Kota Cirebon memiliki banyak pelabuhan di sepanjang pantai utara Jawa dan Selat Sunda. Pelabuhan ini 
menghubungkan kota Cirebon dengan kota-kota lain di pulau Jawa. Kota Cirebon juga memiliki pelabuhan 
kapal laut di Selat Sunda, seperti Pelabuhan Banten dan Pelabuhan Lemahir di Laut Jawa. Pelabuhan Cirebon 
terdapat di kota Cirebon yang menghubungkan Pelabuhan Cirebon dengan Pelabuhan Pelabuhan Kalibagor 
(Pelabuhan Merak). Pelabuhan ini juga menjadi penghubung kapal terbesar untuk rute lintas samudera 
(kapal barang di Jawa. Pelabuhan Cirebon dikenal akan tiga pelabuhan

Output:
----------------------------------------------------------------------------------------------------
2: Sriwijaya adalah salah satu kemaharajaan bahari yang pernah berdiri di pulau Sumatra dan banyak 
memberi pengaruh di Nusantara dengan daerah kekuasaan yang luas, seperti kerajaan Melayu di pulau 
Sumatra dan Kesultanan Malaka di pulau Sumatra.
Pada awal abad ke-14 Sriwijaya membangun benteng pertamanya di pulau Bintan yang disebut " "Bataram Lama" ", 
yang disebut " "bataram", yang kemudian digantikan oleh pemerintahan "Mataram Lama" ". "Bataram Lama" 
menjadi ibukota " "tujuh belas kerajaan yang meliputi wilayah laut dan laut di seluruh Nusantara.
Dari catatan-catatan China yang diketahui dengan pasti Sriwijaya disebut di Prasasti Batu Gelang 
(atau "Batu Tulis"), Prasasti Padang Roco (nama lain dari Prasasti Batu Tulis), Prasasti Cindulungagung 
dan Prasasti Batu Barabai ), dan Prasasti

Output:
----------------------------------------------------------------------------------------------------
3: Pantai berpasir putih ini cukup populer akhir-akhir ini karena menawarkan pemandangan yang begitu 
eksotis, indah dan mempesona, dengan pantai yang indah, seperti Pantai Pasir Putih, Pantai Pasir Putih, 
dan Pantai Tanjung Kelabit. Keunikan Pantai berpasir putih ini adalah pantainya yang alami dan indah yang 
ditumbuhi lumut-semut putih.
Berada di pantai ini tak pernah terlepas dari aktivitas para nelayan karena letaknya yang strategis 
di pinggir jalan raya Pantura. Namun di pantai ini tidak pernah sepi dan terdapat banyak pedagang yang 
menjual hasil laut seperti kepiting, udang dll. Salah satunya adalah lobster air tawar "(kerang)" atau 
udang. Banyak pengunjung pantai berpasir putih yang menjual makanan laut ini dan menjualnya ke penjual 
ikan hias.

Pantai pasir putih juga cukup potensial dikembangkan di sisi utara atau utara desa Pantai

```
We hope that the result could be improved a lot if we have much more dataset. As comparison, The OpenAI 
GPT-2 small model was pre-trained with 40GB of data (our indonesian Wikipedia dataset is only about 1.25% 
of it).

### Text Classification
We can use the MLM language models such as BERT or RoBERTa for text classification. We use the dataset 
[Word Bahasa Indonesia Corpus and Parallel English Translation](https://github.com/cahya-wirawan/language-modeling/tree/master/data). 
This is the same dataset we used for [text classification using ULMFiT](https://github.com/cahya-wirawan/language-modeling/tree/master/indonesia).

- A detail [Text Classification's Notebook](https://github.com/cahya-wirawan/language-modeling/blob/master/Transformers/BERT/bert-indonesian-text-classification.ipynb)
using BERT. The first test achieved an accuracy of 0.9429 which is little bit lower than the accuracy I get from ULMFiT (0.9563). 
This shows that ULMFiT is an excellent language model for text classification despite it's lower number of parameters 
(40M) comparing to BERT-base (110M).
- A [Simple Text Classification's Notebook](https://github.com/cahya-wirawan/language-modeling/blob/master/Transformers/BERT/bert-indonesian-text-classification-simple.ipynb)
using BERT or RoBERTa and [Simple Transformers](https://simpletransformers.ai/)
- A [Simple Text Classification's Notebook](https://github.com/cahya-wirawan/language-modeling/blob/master/Transformers/BERT/bert-indonesian-text-classification-simple-news.ipynb)
using BERT or RoBERTa and [Simple Transformers](https://simpletransformers.ai/) but using [indonesian news dataset](https://github.com/andreaschandra/indonesian-news/) 
 
### Other Downstream Tasks
