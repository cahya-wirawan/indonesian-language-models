# Transformer based Language Models for Indonesian

The transformer-based language model has been very popular since 2018, and there are already pre-trained models 
in many different languages. Unfortunately there is still no indonesian language model available for public 
(May 2020), maybe due to huge resources needed  for the training from scratch. We already have some Malaysian 
language models, but Malay and Indonesian still differ, even though they both belong to the same Austronesian 
language. For this reason, I try to train some language models myself and make them available to the public in 
the hope that they can be useful for Indonesian NLP research.

A documentation and a better GPT2 Language Model trained with more indonesian datasets (indonesian corpus from 
[OSCAR dataset](https://oscar-corpus.com/) is already in the plan) will follow. If you have dataset or just 
any suggestion which datasets I could use to train, just let me know and I will try to compile all datasets 
and to find hardware resources to train it. Thanks.

## GPT-2
### GPT2-small with indonesian Wikipedia
This is my first attempt to create a small indonesian GPT2 Language Model. It was trained only with dataset from 
indonesian Wikipedia (around 522MB) for 5 hours. It is hosted at huggingface:
[gpt2-small-indonesian-522M](https://huggingface.co/cahya/gpt2-small-indonesian-522M). 


## T5

## BERT

## RoBERTa

## XLM-RoBERTa

## ELECTRA

## LONGFORMER
