---
language: "id"
thumbnail: ""
tags:
- array
- of
- tags
license: "mit"
datasets:
- Indonesian Wikipedia
metrics:
- array of metric identifiers
---

# Model name

## Model description
It is BERT-base model pre-trained with indonesian Wikipedia using a masked language modeling (MLM) objective. This 
model is uncased: it does not make a difference between indonesia and Indonesia.

## Intended uses & limitations

#### How to use
You can use this model directly with a pipeline for masked language modeling:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='cahya/bert-base-indonesian-522M')
>>> unmasker("Ibu ku sedang bekerja [MASK] supermarket")

[{'sequence': '[CLS] ibu ku sedang bekerja di supermarket [SEP]',
  'score': 0.7983310222625732,
  'token': 1495},
 {'sequence': '[CLS] ibu ku sedang bekerja. supermarket [SEP]',
  'score': 0.090003103017807,
  'token': 17},
 {'sequence': '[CLS] ibu ku sedang bekerja sebagai supermarket [SEP]',
  'score': 0.025469014421105385,
  'token': 1600},
 {'sequence': '[CLS] ibu ku sedang bekerja dengan supermarket [SEP]',
  'score': 0.017966199666261673,
  'token': 1555},
 {'sequence': '[CLS] ibu ku sedang bekerja untuk supermarket [SEP]',
  'score': 0.016971781849861145,
  'token': 1572}]
```
Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import BertTokenizer, BertModel

model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in Tensorflow:
```python
from transformers import BertTokenizer, TFBertModel

model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

Describe the data you used to train the model.
If you initialized it with pre-trained weights, add a link to the pre-trained model card or repository with description of the pre-training data.

## Training procedure

Preprocessing, hardware used, hyperparameters...

## Eval results

### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```
