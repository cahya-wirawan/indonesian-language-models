# Indonesian Language Modeling
It is Language Modeling in Indonesian created with [ULMFit](https://arxiv.org/abs/1801.06146) 
implementation from [fast.ai](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html).
A state-of-the-art language modeling with perplexity of **27.67** on Indonesian Wikipedia has been achieved. 
As reference, a perplexity of **40.68** has been achieved by [Yang et al (2017)](https://arxiv.org/abs/1711.03953) 
for English WikiText-2 in June 12, 2018. Another paper by [Rae et al (2018)](https://arxiv.org/abs/1803.10049) 
achieved perplexity of **29.2** for English WikiText-103. Lower perplexity means better performance. Obviously, 
the perplexity of the language model on Indonesian Wikipedia can't be compared with both mentioned papers due to 
completely difference dataset, but as reference, I hope it can be still useful. To the best of my knowledge, 
there is no comparable research in Indonesian language at the point of writing (September 21, 2018).

The model has been trained in [jupyter notebook](https://github.com/cahya-wirawan/language-modeling/blob/master/indonesia/ulmfit.ipynb)
using the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182)
--basically LSTM with droupouts--with data from [Wikipedia](https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2) 
(last updated Sept 21, 2018). The dataset has been split to 90/10 train-validation with a vocabulary size of 60,002
and embeddings dimensions of 400.

The language model can also be used to extract text features for other downstream tasks such as text 
classification, speech recognition or machine translation.

# Text Classification
Since there is no other comparable indonesian language model, we need to create a downstream task and compare its 
accuracy. A Text Classification was chosen for this purpose, but it is a big challenge to find curated or publicly 
available dataset for indonesian text. Nevertheless, a small curated indonesian dataset was found eventually. 
It is [Word Bahasa Indonesia Corpus and Parallel English Translation](https://www.panl10n.net/english/outputs/Indonesia/BPPT/0902/BPPTIndToEngCorpusHalfM.zip) 
dataset from PAN Localization. It contains 500,000 words from various online sources translated into English.
Actually, its purpose is for indonesian-english translation, but we "misused" it for text classification, and only 
the indonesian part are used for this purpose. The corpus has 4 categories:
                                               
* Economy
* International
* Science
* Sport

## Performance Comparison
Currently, there is no comparable text classification's result using this dataset, therefore
a [performance test](https://github.com/cahya-wirawan/language-modeling/blob/master/indonesia/ulmfit_classification_comparison.ipynb) 
with various other algorithms, such as Naive Bayes (NB), Linear Classifier (LC), Support Vector Machine (SVM),
Random Forest (RF), Extreme Gradient Boosting(Xgb), Convolition Neural Network (CNN), LSTM or GRU, 
has been performed. Following is the test result:

| Name                   | Accuracy |
| ---------------------- |---------:| 
| NB, Count Vectors      |   0.9269 |
| NB, WordLevel TF-IDF   |   0.9162 |
| NB, N-Gram Vectors     |   0.7822 |
| NB, CharLevel Vectors  |   0.8433 |
| LC, Count Vectors      |   0.9265 |
| LC, WordLevel TF-IDF   |   0.9179 |
| LC, N-Gram Vectors     |   0.8085 |
| LC, CharLevel Vectors  |   0.8888 |
| SVM, N-Gram Vectors(*) |   0.2582 |
| RF, Count Vectors      |   0.8392 |
| RF, WordLevel TF-IDF   |   0.8338 |
| Xgb, Count Vectors     |   0.8087 |
| Xgb, WordLevel TF-IDF  |   0.8070 |
| Xgb, CharLevel Vectors |   0.8202 |
| CNN                    |   0.9263 |
| Kim Yoon’s CNN         |   0.9163 |
| RNN-LSTM               |   0.9305 |
| RNN-GRU                |   0.9296 |
| Biderectional RNN      |   0.9267 |
| RCNN                   |   0.9221 |
| **ULMFit**             | **0.9563** |

(*) something is wrong with the training on svm

It shows that text classification using ULMFit outperformed other algorithms using classical machine learning 
or other neural network models.

# Dependencies
* Python 3.6.5
* PyTorch 0.4.0
* fast.ai

# Version History

## v0.1

* Pretrained language model based on Indonesian Wikipedia with the perplexity of 38.78
* The pre-trained model and preprocessed training dataset of Indonesian Wikipedia can be downloaded via 
  [Nofile.io](https://nofile.io/f/NZDQB8Wo0eU/lm_data.tgz).


## v0.2

* The second version includes [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy) 
which speed up the training time. The model has been also trained with more epochs (around 30 epochs) which highly 
improved the perplexity from **38.78** to **27.67**.
* The pre-trained model will be available soon.

# Text Generation using the language model
The language model (v0.1) has been tested to generate sentences using 
[this script](https://github.com/cahya-wirawan/language-modeling/blob/master/indonesia/ulmfit_test.py). 
Various strings seeds are used for this purpose:

- "jika ibu bersedih sepanjang hari",
- "orang baduy adalah",
- "presiden soekarno adalah",
- "jatuh cinta disebabkan",
- "laki laki jatuh cinta adalah",
- "gadis jatuh cinta disebabkan",
- "seks dan cinta adalah",
- "borobudur adalah warisan",
- "anak balita adalah",
- "ibukota rusia adalah",
- "australia terletak"
 
And the results (and its translation using google) are quite interesting:

- jika ibu bersedih sepanjang hari, maka akan terjadi bencana alam yang hebat. 
_(if the mother grieves all day, there will be a great natural disaster.)_

- orang baduy adalah orang - orang yang memiliki kemampuan untuk berbicara dengan orang - orang yang tidak memiliki 
bahasa. mereka juga memiliki kemampuan untuk berbicara dengan orang lain, dan mereka dapat berbicara dalam bahasa 
yang berbeda. _(Baduy people are people who have the ability to talk to people who do not have language. they also 
have the ability to talk with other people, and they can speak in different languages.)_

- presiden soekarno adalah seorang yang sangat terkenal di indonesia. ia adalah seorang yang sangat cerdas dan 
memiliki kemampuan untuk menguasai dunia. ia juga memiliki kemampuan untuk menciptakan dan menciptakan sebuah sistem 
yang dapat digunakan untuk melakukan tugas - tugas yang bersifat khusus. ia juga memiliki kemampuan untuk menciptakan 
dan mengendalikan berbagai macam bentuk dan kemampuan untuk melakukan hal - hal yang tidak diinginkan. ia juga 
memiliki kemampuan untuk menciptakan dan mengendalikan sebuah sistem yang dapat mengubah dirinya menjadi sebuah
_(President Soekarno is a very famous person in Indonesia. he is a very smart man and
  have the ability to rule the world. he also has the ability to create and create a system
  which can be used to perform special tasks. he also has the ability to create
  and control various forms and abilities to do things that are not desirable. He also
  has the ability to create and control a system that can transform itself into a)_ 

- jatuh cinta disebabkan oleh adanya perbedaan pendapat mengenai apakah mereka akan melakukan hubungan seksual. 
_(falling in love is caused by differences of opinion about whether they will have sexual relations)_

- laki laki jatuh cinta adalah hal yang tidak biasa. _(men fall in love is unusual.)_

- gadis jatuh cinta disebabkan oleh cinta kasih yang kuat. _(The girl falls in love due to the strong love.)_

- seks dan cinta adalah sebuah hal yang sangat penting bagi kehidupan manusia. 
_(sex and love are very important things for human life)_

- borobudur adalah warisan dari kerajaan mataram kuno. _(Borobudur is a legacy of the ancient Mataram kingdom)_

- anak balita adalah anak - anak yang lahir dari keluarga yang memiliki kemampuan untuk melakukan hal - hal yang 
tidak diinginkan. anak - anak yang lahir dari keluarga yang memiliki kemampuan untuk melakukan hal tersebut, 
seperti anak - anak, anak - anak, dan orang dewasa. anak - anak yang lahir dari keluarga yang memiliki kemampuan 
untuk melakukan hal tersebut dapat menjadi orang tua yang baik, dan anak - anak mereka akan menjadi anak yang 
baik dan mampu 
_(Toddlers are children born to families who have the ability to do things
  undesirable. children born to families who have the ability to do this,
  like children, children, and adults. children born to families who have abilities
  to do this can be a good parent, and their children will become children
  good and capable)_
  
- ibukota rusia adalah kota terbesar di dunia, yaitu kota moskow. 
_(Russia's capital is the largest city in the world, the city of Moscow)_

- australia terletak di sebelah barat laut pulau papua. 
_(Australia is located in the northwest of the island of Papua.)_
