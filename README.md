# Indonesian Language Model

The language model is a probability distribution over word sequences used to predict the next word based on previous 
sentences. This ability makes the language model the core component of modern natural language processing. We use it 
for many different tasks, such as speech recognition, conversational AI, information retrieval, sentiment analysis, 
or text summarization.

For this reason, many big companies are competing to build large and larger language models, such as Google BERT, 
Facebook RoBERTa, or OpenAI GPT3, with its massive number of parameters. Most of the time, they built only 
language models in English and some other European languages. Other countries with low resource languages have big 
challenges to catch up on this technology race.

Therefore the author tries to build some language models for Indonesian, started with ULMFiT in 2018. The first 
language model has been only trained with  Indonesian Wikipedia, which is very small compared to other datasets used 
to train the English language model.


## Universal Language Model Fine-tuning (ULMFiT)
Jeremy Howard and Sebastian Ruder proposed [ULMFiT](https://arxiv.org/abs/1801.06146) in early 2018 as a novel method for 
fine-tuning language models for inductive transfer learning. The language model [ULMFiT for Indonesian](https://github.com/cahya-wirawan/indonesian-language-models/tree/master/ULMFiT) 
has been trained as part of the author's project while learning [FastAI](https://www.fast.ai). It achieved a perplexity 
of **27.67** on Indonesian Wikipedia.

## Transformers
