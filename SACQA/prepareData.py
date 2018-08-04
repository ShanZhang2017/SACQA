import json
import re
import numpy as np
from tensorflow.contrib import learn
import config
text=[]
graph_text_train=[]
graph_text_test=[]
graph_train=[]
y_train=[]
ys_train=[]
graph_test = []
y_test=[]
ys_test=[]
with open("datasets/train",'r',encoding='utf-8') as fr:
	for line in fr:
		lls=line.strip().split('\t')
		lines=[]
		for l in lls[0:-1]:
			s = re.sub("[\s+\.\!\/_,\-\;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）_-《》]+-', ", ' ', l)
			lines.append(s.lower())
		graph_text_train.append(lines)
		y_train.append(str(lls[-1]))
		for l in lines:
			if l not in text:
				text.append(l)
fr.close()

with open("datasets/test",'r',encoding='utf-8') as fr:
	for line in fr:
		lls = line.strip().split('\t')
		lines = []
		for l in lls[0:-1]:
			s = re.sub("[\s+\.\!\/_,\-\;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）_-《》]+-', ", ' ', l)
			lines.append(s.lower())
		graph_text_test.append(lines)
		y_test.append(str(lls[-1]))
		for l in lines:
			if l not in text:
				text.append(l)
fr.close()
list_num=[]
for i in range(0,len(text)):
	list_num.append(i)
text_dict=dict(zip(text,list_num))
jsObj = json.dumps(text_dict)

fileObject = open('jsonFile.json', 'w')
fileObject.write(jsObj)
fileObject.close()
num=0
for ts in graph_text_train:
	temp=[]
	for t in ts:
		temp.append(str(text_dict[t]))
	if len(temp)==3:
		tst='\t'.join(temp)
	if tst not in graph_train:
		graph_train.append(tst)
		ys_train.append(y_train[num])
	num+=1
num=0
for ts in graph_text_test:
	if num==0:
		print(ts)
	temp=[]
	for t in ts:
		temp.append(str(text_dict[t]))
	if len(temp)==3:
		if num == 0:
			print(ts)
		tst='\t'.join(temp)
	if tst not in graph_test:
		if num == 0:
			print(tst)
		graph_test.append(tst)
		ys_test.append(y_test[num])
	num+=1

with open('datasets/data.txt','w',encoding='utf-8') as fw:
	fw.write('\n'.join(text))
fw.close()
with open('datasets/train_graph.txt','w',encoding='utf-8')as fw:
	fw.write('\n'.join(graph_train))
fw.close()
with open('datasets/train_y.txt','w',encoding='utf-8')as fw:
	fw.write('\n'.join(ys_train))
fw.close()
with open('datasets/test_graph.txt','w',encoding='utf-8')as fw:
	fw.write('\n'.join(graph_test))
fw.close()
with open('datasets/test_y.txt','w',encoding='utf-8')as fw:
	fw.write('\n'.join(ys_test))
fw.close()
pre=None
nnk=0
kn=[]
for k in graph_test:
	ks=k.strip().split('\t')
	if ks[0]!=pre:
		nnk+=1
	pre=ks[0]
	kn.append(str(nnk))
print(kn)
with open('datasets/test_q.txt', 'w', encoding='utf-8')as fw:
	fw.write('\n'.join(kn))
fw.close()




# 建立worrdvector

#
# text_file=open('datasets/data.txt','rb').readlines()
#
# vocab=learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
# text_f=[x.decode() for x in text_file]
# text=np.array(list(vocab.fit_transform(text_f)))
# vocab_dict = vocab.vocabulary_._mapping
# num_vocab=len(vocab.vocabulary_)
# print(num_vocab)
# sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
# vocabulary = list(list(zip(*sorted_vocab))[0])
# # print(vocabulary)
#
# embeddings_index = {}
# with open("datasets/glove.6B.200d.txt","r",encoding='utf-8') as fr:
# 	for line in fr:
# 		values = line.split()
# 		word = values[0]
# 		coefs = np.asarray(values[1:], dtype='float32')
# 		embeddings_index[word] = coefs
# fr.close()
# embed=[]
# for word in vocabulary:
# 	# print(word)
# 	word_vector=embeddings_index.get(word)
# 	if word_vector is None:
# 		word_vector=np.zeros(200)
# 	word_vector=map(str,word_vector)
# 	v_text=" ".join(word_vector)
# 	embed.append(v_text)
#
# with open('datasets/embedding.txt','w',encoding='utf-8')as fw:
# 	fw.write('\n'.join(embed))
# fw.close()