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

### 1. GPT2-small with indonesian Wikipedia
This is my first attempt to create a small indonesian GPT2 Language Model. It was trained only with dataset from 
indonesian Wikipedia (around 522MB) for 5 hours. It is hosted at huggingface:
[gpt2-small-indonesian-522M](https://huggingface.co/cahya/gpt2-small-indonesian-522M).

### 2. GPT2-small with indonesian Wikipedia and OSCAR Corpus
The OSCAR Corpus for the deduplicated indonesian dataset is about 16GB. The training is on the plan.

### Usage
#### Language Generation
The Jupyter notebook: [gpt2-indonesian.ipynb](https://github.com/cahya-wirawan/language-modeling/blob/master/Transformers/GPT2/gpt2-indonesian.ipynb)
```
import torch
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("cahya/gpt2-small-indonesian-522M", pad_token_id=tokenizer.eos_token_id)

input_sentences = [
    'Cirebon adalah sebuah kota kecil di Jawa Barat dengan keindahannya yang melebihi kota Bandung',
    'Sriwijaya adalah salah satu kemaharajaan bahari yang pernah berdiri di pulau Sumatra dan banyak memberi pengaruh di Nusantara dengan daerah kekuasaan yang luas',
    'Pantai berpasir putih ini cukup populer akhir-akhir ini karena menawarkan pemandangan yang begitu eksotis dan mempesona.',
    'Perbukitan yang hijau dipenuhi dengan pepohonan tropis yang lengkap dengan area persawahan dan lembah'    
]

# We set the seed manualy for reproducible result
torch.manual_seed(3)
for i, sentence in enumerate(input_sentences):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    sample_output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=100, 
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
0: Cirebon adalah sebuah kota kecil di Jawa Barat dengan keindahannya yang melebihi kota Bandung, 
Jawa Barat, Indonesia.  Kota ini didirikan pada tahun 1946 dengan sebutan "Calabo" (bahasa Indonesia: 
"Balabo") (Bahasa Indonesia: "Badiwa") yang artinya kota ini memiliki sejarah yang panjang dan sedikit 
yang lebih luas. Kota ini hanya terdapat di sekitar kota Bandung, tetapi di beberapa kota besar, 
melainkan banyak kota di kota kecil saja seperti kota Bandung, dan beberapa kota di luar pulau

Output:
----------------------------------------------------------------------------------------------------
1: Sriwijaya adalah salah satu kemaharajaan bahari yang pernah berdiri di pulau Sumatra dan banyak 
memberi pengaruh di Nusantara dengan daerah kekuasaan yang luas pada tahun 1685 – 1177. Kerajaan ini
pertama kali dijadikan salah satu kota yang sangat populer dan dibangun di Sulawesi Selatan.  Kerajaan 
kerajaan ini dibentuk oleh Sultan Agung Agung, yaitu penguasa Kerajaan Sriwijaya tahun 1387 M 
(tahun 1063 M), dan merupakan salah satu dari Kerajaan Sriwijaya yang memerintah selama Perang Dunia I. 
Raja ini sendiri berasal dari Kerajaan Sumbawa dan kemudian menetap di Sumatra

Output:
----------------------------------------------------------------------------------------------------
2: Pantai berpasir putih ini cukup populer akhir-akhir ini karena menawarkan pemandangan yang begitu 
eksotis dan mempesona. Pantai ini bisa kita bisa berenang dan bisa merasakan rasa dingin di udara kering.
Pantai ini letaknya di bagian timur pantai. Pantai ini sangat indah. Pantai ini letaknya sangat sangat 
sejuk dan sejuk dan sejuk. Pantai ini memiliki pemandangan yang indah. Pantai ini sangat sejuk dan baik,
dapat diakses dengan baik dari kota kota-kota kecil maupun tempat-tempat tertentu seperti Pantai Gading.
Pantai ini biasanya banyak ditemui di daerah kota lain

Output:
----------------------------------------------------------------------------------------------------
3: Perbukitan yang hijau dipenuhi dengan pepohonan tropis yang lengkap dengan area persawahan dan lembah 
yang indah, serta pegunungan tropis, di sebelah utara, sungai dengan puncak pegunungan. Pada umumnya, 
kawasan hutan ini memiliki iklim tropis yang subur dan tropis. Rata-rata suhu yang rendah mencapai 1,2 %.
Suhu rata-rata rata-rata 25,4 mm / tahun dengan suhu rata-rata 23,3 mm / tahun. Suhu rata-rata antara 
5-4 hari dan rata-rata tahunan adalah 23-3 mm

```
We hope that the result could be improved a lot if we have much more dataset. As comparison, The OpenAI 
GPT-2 small model was pre-trained with 40GB of data (our indonesian Wikipedia dataset is only about 1.25% 
of it).

#### Text Classification

#### Other Downstream Tasks

## T5

## BERT

## RoBERTa

## XLM-RoBERTa

## ELECTRA

## LONGFORMER
