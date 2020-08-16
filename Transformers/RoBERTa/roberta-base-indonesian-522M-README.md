---
language: "id"
thumbnail: ""
license: "mit"
datasets:
- Indonesian Wikipedia
widget:
- text: "Ibu ku sedang bekerja <mask> supermarket."
---

# Indonesian RoBERTa base model (uncased) 

## Model description
It is RoBERTa-base model pre-trained with indonesian Wikipedia using a masked language modeling (MLM) objective. This 
model is uncased: it does not make a difference between indonesia and Indonesia.

## Intended uses & limitations

### How to use
You can use this model directly with a pipeline for masked language modeling:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='cahya/bert-base-indonesian-522M')
>>> unmasker("Ibu ku sedang bekerja <mask> supermarket")

[{'sequence': '<s> ibu ku sedang bekerja di supermarket </s>',
  'score': 0.7983310222625732,
  'token': 1495},
 {'sequence': '<s> ibu ku sedang bekerja. supermarket </s>',
  'score': 0.090003103017807,
  'token': 17},
 {'sequence': '<s> ibu ku sedang bekerja sebagai supermarket </s>',
  'score': 0.025469014421105385,
  'token': 1600},
 {'sequence': '<s> ibu ku sedang bekerja dengan supermarket </s>',
  'score': 0.017966199666261673,
  'token': 1555},
 {'sequence': '<s> ibu ku sedang bekerja untuk supermarket </s>',
  'score': 0.016971781849861145,
  'token': 1572}]
```
Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import RobertaTokenizer, RobertaModel

model_name='cahya/roberta-base-indonesian-522M'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in Tensorflow:
```python
from transformers import RobertaTokenizer, TFRobertaModel

model_name='cahya/roberta-base-indonesian-522M'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = TFRobertaModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

This model was pretrained with 522MB of indonesian Wikipedia.
The texts are lowercased and tokenized using WordPiece and a vocabulary size of 32,000. The inputs of the model are 
then of the form:

```<s> Sentence A </s> Sentence B </s>```

## BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```
