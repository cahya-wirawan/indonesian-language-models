# Indonesian Language Modeling
State-of-the-Art Language Modeling in Indonesian created with [ULMFit](https://arxiv.org/abs/1801.06146) 
implementation from [fast.ai](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

The pre-trained model and preprocessed training dataset of Indonesian Wikipedia can be downloaded via [Nofile.io](https://nofile.io/f/NZDQB8Wo0eU/lm_data.tgz)
We provide state-of-the-art language modeling (perplexity of 38.78 on Indoenesian wikipedia). We will try to
improve it with more training time since the curve for validation loss still have good trend toward lower loss.

Due to some difficulties to find curated and publicly available dataset for indonesian text, we can't 
provide a benchmark for text classification yet, but as soon as we can find one (please contact us if you have one),
we will update our research.

The language model can also be used to extract text features for other downstream tasks.

# Dependencies
* Python 3.6.5
* PyTorch 0.4.0
* fast.ai

# Version History

## v0.1

* Pretrained language model based on Indonesian Wikipedia with the perplexity of 38.78


# Language Modeling

The Indonesian language model was trained using the [fast.ai](http://www.fast.ai/) version of 
[AWD LSTM Language Model](https://arxiv.org/abs/1708.02182)
--basically LSTM with droupouts--with data from [Wikipedia](https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2) 
(last updated Sept 21, 2018). Using 90/10 train-validation split, we achieved perplexity of **38.78 with 60,002 
embeddings at 400 dimensions**, compared to state-of-the-art as of June 12, 2018 at **40.68 for English WikiText-2 
by [Yang et al (2017)](https://arxiv.org/abs/1711.03953)** and **29.2 for English WikiText-103 by 
[Rae et al (2018)](https://arxiv.org/abs/1803.10049)**. 
To the best of our knowledge, there is no comparable research in Indonesian language at the point of writing 
(September 21, 2018).

# Text Generation using the language model
We tested our language model to generate sentences using some strings seeds:
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
 
And the results are quite interesting:

- jika ibu bersedih sepanjang hari, maka akan terjadi bencana alam yang hebat. 

- orang baduy adalah orang - orang yang memiliki kemampuan untuk berbicara dengan orang - orang yang tidak memiliki bahasa. mereka juga memiliki kemampuan untuk berbicara dengan orang lain, dan mereka dapat berbicara dalam bahasa yang berbeda. 

- presiden soekarno adalah seorang yang sangat terkenal di indonesia. ia adalah seorang yang sangat cerdas dan memiliki kemampuan untuk menguasai dunia. ia juga memiliki kemampuan untuk menciptakan dan menciptakan sebuah sistem yang dapat digunakan untuk melakukan tugas - tugas yang bersifat khusus. ia juga memiliki kemampuan untuk menciptakan dan mengendalikan berbagai macam bentuk dan kemampuan untuk melakukan hal - hal yang tidak diinginkan. ia juga memiliki kemampuan untuk menciptakan dan mengendalikan sebuah sistem yang dapat mengubah dirinya menjadi sebuah 

- jatuh cinta disebabkan oleh adanya perbedaan pendapat mengenai apakah mereka akan melakukan hubungan seksual. 

- laki laki jatuh cinta adalah hal yang tidak biasa. 

- gadis jatuh cinta disebabkan oleh cinta kasih yang kuat. 

- seks dan cinta adalah sebuah hal yang sangat penting bagi kehidupan manusia. 

- borobudur adalah warisan dari kerajaan mataram kuno. 

- anak balita adalah anak - anak yang lahir dari keluarga yang memiliki kemampuan untuk melakukan hal - hal yang tidak diinginkan. anak - anak yang lahir dari keluarga yang memiliki kemampuan untuk melakukan hal tersebut, seperti anak - anak, anak - anak, dan orang dewasa. anak - anak yang lahir dari keluarga yang memiliki kemampuan untuk melakukan hal tersebut dapat menjadi orang tua yang baik, dan anak - anak mereka akan menjadi anak yang baik dan mampu 

- ibukota rusia adalah kota terbesar di dunia, yaitu kota moskow. 

- australia terletak di sebelah barat laut pulau papua. 

# Text Classification

We are trying to find publicly available dataset for indonesian text.
