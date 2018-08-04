import config
import numpy as np
from tensorflow.contrib import learn
import random
class dataSet:
	def __init__(self,text_path,train_graph_path,val_graph_path,train_y_path,val_y_path,val_q_path):

		text_file,train_graph_file,val_graph_file,train_y_flie,val_y_flie,val_q_file=self.load(text_path,train_graph_path,val_graph_path,train_y_path,val_y_path,val_q_path)

		self.train_edges=self.load_edges(train_graph_file)
		self.val_edges = self.load_edges(val_graph_file)
		self.train_y=self.load_y(train_y_flie)
		self.test_y = self.load_y(val_y_flie)
		self.text, self.num_vocab, self.num_nodes = self.load_text(text_file)
		self.test_q=self.load_q(val_q_file)

	def load_y(self,y_file):
		y=[]
		for i in y_file :
			i=i.strip()
			y.append(float(i))
		return y

	def load_q(self,q_flie):
		q=[]
		for i in q_flie:
			i=i.strip()
			q.append(float(i))
		return q

	def load(self,text_path,train_graph_path,val_graph_path,train_y_path,val_y_path,val_q_path):
		text_file=open(text_path,'rb').readlines()
		train_graph_file=open(train_graph_path,'rb').readlines()
		val_graph_file = open(val_graph_path, 'rb').readlines()
		train_y_flie=open(train_y_path,'rb').readlines()
		val_y_flie = open(val_y_path, 'rb').readlines()
		val_q_flie = open(val_q_path, 'rb').readlines()
	
		return text_file,train_graph_file,val_graph_file,train_y_flie,val_y_flie,val_q_flie

	def load_edges(self,graph_file):# 每行：问题陈述部分编号，问题疑问部分编号，正确答案编号，错误编号
		edges=[]
		for i in graph_file:
			i=i.decode()
			edges.append(list(map(int,i.strip().split('\t'))))#[总行数，4]
	
		return edges

	def load_text(self,text_file): #没问题 放的是所有文本 形式大概是：问题1陈述\n 问题2疑问\n 正确答案\n 错误答案1\n 错误答案2.。。。问题2陈述\n 问题2疑问\n 正确答案\n 错误答案1\n 错误答案2.。。
		vocab=learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
		text_f=[x.decode() for x in text_file]
		text=np.array(list(vocab.fit_transform(text_f)))
		vocab_dict = vocab.vocabulary_._mapping
		sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
		vocabulary = list(list(zip(*sorted_vocab))[0])
		num_vocab=len(vocab.vocabulary_)#所有词数
		num_nodes=len(text) #文本文件里的行数

		return text,num_vocab,num_nodes

	def generate_batches(self,mode=None):

		if mode != 'validation':
			num_batch = len(self.train_edges) // config.batch_size
			print(num_batch)
			edges = self.train_edges
			ys=self.train_y
			ey=list(zip(edges,ys))
			random.shuffle(ey)
		if mode=='validation':
			num_batch = len(self.val_edges) // config.batch_size
			edges = self.val_edges
			ys=self.test_y
			qs=self.test_q
			ey=list(zip(edges,ys,qs))
			num_batch+=1
			print(len(ey))
			ey.extend(ey[:(config.batch_size-len(ey) % config.batch_size)])
			print(len(ey))

		sample_ey=ey[:int(num_batch*config.batch_size)]
		batches=[]
		y=[]
		# print (num_batch)
		sample_edges=[]
		sample_ys=[]
		if mode != 'validation':
			sample_edges[:],sample_ys[:]=zip(*sample_ey)
			for i in range(num_batch):
				batches.append(sample_edges[i*config.batch_size:(i+1)*config.batch_size])
				y.append(sample_ys[i*config.batch_size:(i+1)*config.batch_size])
			# print sample_edges[0]
			return batches,y
		else:
			q=[]
			sample_qs=[]
			sample_edges[:], sample_ys[:],sample_qs[:] = zip(*sample_ey)
			for i in range(num_batch):
				batches.append(sample_edges[i*config.batch_size:(i+1)*config.batch_size])
				y.append(sample_ys[i*config.batch_size:(i+1)*config.batch_size])
				q.append(sample_qs[i*config.batch_size:(i+1)*config.batch_size])
			# print sample_edges[0]
			return batches,y,q

