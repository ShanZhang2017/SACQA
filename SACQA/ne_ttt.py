
import numpy as np
from tensorflow.contrib import learn
import re

x_ntext = ['there ------ are good people in this world, and they are not only muslims; you know ge', 'This is a a dog']
x_text=[]
for line in x_ntext:
    s=re.sub("[\s+\.\!\/_,\-\;$%^*(+\"\')]+|[+——()?【】“”！，;；。？、~@#￥%……&*（）_-《》]+'-,------------------------------------------------------ ;",' ',line)
    x_text.append(s.lower())
print(x_text)


max_document_length = max([len(x.split(" ")) for x in x_text])

## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(x_text)))

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])

print(vocabulary)
print(x)