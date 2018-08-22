from __future__ import division
import torch
import numpy as np
from scipy import spatial

# Compute the weights for each word based on the formula in the paper
def getWordWeights(weightfile, a=1e-3):
	if a <=0: # when the parameter makes no sense, use unweighted
		a = 1.0

	word2weight = {}
	N = 0
	with open(weightfile, "r") as reader:
		for line in reader:
			line = line.strip()
			if line:
				line = line.split()
				if(len(line) == 2):
					word2weight[line[0]] = float(line[1])
					N += float(line[1])
				else:
					print(line)
	for key, value in word2weight.iteritems():
		word2weight[key] = a / (a + value/N)
	return word2weight

# Gen the word embeddings
def getWordEmbeddings(embeddings_file):
	embeddings = {}
	with open(embeddings_file, "r") as reader:
		for line in reader:
			line = line.strip()
			line = line.split(" ", 1)
			embeddings[line[0].strip()] = np.fromstring(line[1], dtype=np.float, sep=" ")
	return embeddings

def lookupEmbedding(embeddings, w):
	w = w.lower()
	# convert hashtag to word
	if len(w) > 1 and w[0] == '#':
		w = w.replace("#","")
	if w in embeddings:
		return embeddings[w]
	elif 'UUUNKKK' in embeddings:
		return embeddings['UUUNKKK']
	else:
		return embeddings[-1]

class SIFEmbedding(object):
	"""
	Sentence embedding generator class which implements the weighted averaging technique specified in https://openreview.net/pdf?id=SyK00v5xx
	
	Args:
		tgt_vocab (Vocab): This is the target vocab. We want the reweighted embeddings for these words
		embeddings_file (string): String of the file which contains the original PSL word embeddings
		weights_file (string): wikipedia word frequencies which will help us estimate unigram probabilities
		pc_file (string): Numpy file which contains the first principal component of sentence embeddings of the sample of sentences from training data

	"""
	def __init__(self, tgt_vocab, embeddings_file, weights_file, pc_file, pad_word_id, a=1e-3):
		self.vocab = tgt_vocab
		self.embeddings = getWordEmbeddings(embeddings_file)
		self.word2weight = getWordWeights(weights_file, a)
		# Initialize embeddings and weights for the target vocab
		# itov: index to embedding
		self.itov = list()
		for i in range(len(self.vocab.itos)):
			self.itov.append(lookupEmbedding(self.embeddings, self.vocab.itos[i]))
		# itow: index to weight
		self.itow = list()
		for i in range(len(self.vocab.itos)):
			# initialize the weight only if its corresponding embedding is present
			if self.vocab.itos[i] in self.embeddings and self.vocab.itos[i] in self.word2weight:
				self.itow.append(self.word2weight[self.vocab.itos[i]])
			else:
				self.itow.append(1.0)           # no re weighting
		self.pc = np.load(pc_file)              # pc in the shape of (1, 300)
		# print self.pc.shape
		self.dim = self.pc.shape[1]
		self.pad_word_id = pad_word_id
		self.cache = dict()

	def clear_cache(self):
		self.cache = dict()
	"""
		Args:
			word_ids (list): list of word indices of the sentence
	"""
	def sentence_embedding(self, word_ids):
		# 1) Find the weighted average of the list of words given in the arguments
		# print word_ids
		# for word_id in word_ids.data:
		#     print word_id
		word_embs = np.array([self.itov[word_id] for word_id in word_ids if word_id != self.pad_word_id])
		word_weights = [self.itow[word_id] for word_id in word_ids if word_id != self.pad_word_id]
		emb = np.average(word_embs, axis=0, weights=word_weights)

		# 2) Remove the first principal component from this embedding
		# NOTE: futher code is shamefully copied from the SIF_embddings.py file. Don't complain if it looks ugly
		X = emb.reshape(1, self.dim)
		XX = X - X.dot(self.pc.transpose()) * self.pc
		emb = XX.reshape(self.dim)
		return emb

	def remove_pad_word(self, sent):
		return tuple([word_id for word_id in sent if word_id != self.pad_word_id])

	"""
		Args:
			sent1 (list): List of word indices of the first sentence
			sent2 (list): List of word indices of the second sentence
	"""
	def Similarity(self, sent1, sent2):
		# Cache the sent1 because it is repeating
		sent1 = self.remove_pad_word(sent1)
		sent2 = self.remove_pad_word(sent2)
		if len(sent1) == 0 or len(sent1) == 0:
			return 0.0
		if sent1 in self.cache:
			emb1 = self.cache[sent1]
		else:
			emb1 = self.sentence_embedding(sent1)
			self.cache[sent1] = emb1
		emb2 = self.sentence_embedding(sent2)
		cos_similarity = 1 - spatial.distance.cosine(emb1, emb2)
		return cos_similarity

	"""
		Args:
			targets (list of list): List of target sentences (which are essentially list of word indices)
			sent (list): List of word indices
	"""
	def Targets_Similarity(self, targets, sent, max_flag=True):
		similarity = list()
		if len(targets) == 0:
			return 0.0
		for i in range(len(targets)):
			similarity.append(self.Similarity(targets[i], sent))
		similarity = np.array(similarity)
		if max_flag:
			return np.max(similarity)
		else:
			return np.avg(similarity)

