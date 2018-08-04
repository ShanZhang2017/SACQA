import numpy as np
from tensorflow.contrib import learn
import config
text_file=open('datasets/data.txt','rb').readlines()

vocab=learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
text_f=[x.decode() for x in text_file]
text=np.array(list(vocab.fit_transform(text_f)))
vocab_dict = vocab.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])

print(vocabulary)