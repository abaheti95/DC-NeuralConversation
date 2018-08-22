import codecs
import numpy as np
import timeit
import math

class SyntaxTopicModel(object):
	"""This class is the partial implementation of "Integrating Topics and Syntax" T.L. Griffiths et al.
	We will only use the trained model to get the probabilities of different topics and classes
	"""

	def read_vocabulary(self):
		self.vocab_itos = list()
		self.vocab_stoi = dict()
		with codecs.open(self.vocabulary_file, "r", "utf-8") as reader:
			# Each line is a word and its frequency
			i = 0
			for line in reader:
				line = line.strip()
				line_spl = line.split()
				if line_spl[0] == "UNknown":
					line_spl[0] = "<unk>"
				self.vocab_itos.append(line_spl[0])
				self.vocab_stoi[line_spl[0]] = i
				i += 1

	def read_documents(self, max_documents = None):
		self.documents = list()
		with open(self.documents_file, "r") as reader:
			for i, line in enumerate(reader):
				if max_documents and i > max_documents:
					break
				self.documents.append(np.fromstring(line, dtype=np.int, sep=' '))

	def read_assignments_file(self, filename, variable, t_or_c, max_documents = None):
		# First line is the number of documents and number of topics/classes
		# Then rest of the lines are word labels for each document
		with open(filename, "r") as reader:
			first_line = next(reader)
			self.n_docs, num_classes_or_topics = first_line.split()
			self.n_docs = int(self.n_docs)
			if t_or_c == 'T':
				self.num_topics = int(num_classes_or_topics)
			elif t_or_c == 'C':
				self.num_classes = int(num_classes_or_topics)
			else:
				print "Incorrect T or C choice. Given parameter value =", t_or_c
				exit()
			for i, line in enumerate(reader):
				if max_documents and i > max_documents:
					break
				variable.append(np.fromstring(line.strip(), dtype=np.int, sep=' '))

	def run_counts(self):
		self.n_docs = len(self.documents)
		self.vocab_size = len(self.vocab_itos)
		# We want the following probabilities
		# P(w|C) a V * n_C dimensional matrix which gives probability of word given the class count(word_class)/count(class)
		# P(w|T) a V * n_T dimensional matrix which gives probability of word given the topic count(word_topic)/count(topic)
		# P(T|w) a n_T * V dimensional matrix which gives probability of Topic given the word count(word_topic)/count(word in all topic)
		# P(C_|C) a n_C * n_C dimensional matrix which is the transition probability of going from a class to another class

		# Calculate counts
		import os
		CACHE_FOLDER = "syntax_topic_cache"
		self.num_word_in_class_cache_file = os.path.join(CACHE_FOLDER, "num_word_in_class_cache.npy")
		self.num_word_in_topic_cache_file = os.path.join(CACHE_FOLDER, "num_word_in_topic_cache.npy")
		self.num_transitions_cache_file = os.path.join(CACHE_FOLDER, "num_transitions_cache.npy")

		if not self.recount and os.path.isfile(self.num_word_in_class_cache_file) and os.path.isfile(self.num_word_in_topic_cache_file) and os.path.isfile(self.num_transitions_cache_file):
			# if possible load from cache
			print "WARNING: Reloading stuff from cache!!"
			self.num_word_in_class = np.load(self.num_word_in_class_cache_file)
			self.num_word_in_topic = np.load(self.num_word_in_topic_cache_file)
			self.num_transitions = np.load(self.num_transitions_cache_file)
		else:
			# Compute from the data
			self.num_word_in_class = np.zeros((self.vocab_size, self.num_classes), dtype=np.float)
			self.num_word_in_topic = np.zeros((self.vocab_size, self.num_topics), dtype=np.float)
			self.num_transitions = np.zeros((self.num_classes, self.num_classes), dtype=np.float)
			for doc_id, document in enumerate(self.documents):
				prev_class = -1
				for i, word_id in enumerate(document):
					class_assign = self.class_assignments[doc_id][i]
					topic_assign = self.topic_assignments[doc_id][i]
					if class_assign == 0:
						# Topics class
						self.num_word_in_topic[word_id][topic_assign] += 1
					self.num_word_in_class[word_id][class_assign] += 1
					if prev_class != -1:
						self.num_transitions[prev_class][class_assign] += 1
					prev_class = class_assign
			# Save in cache
			np.save(self.num_word_in_class_cache_file, self.num_word_in_class)
			np.save(self.num_word_in_topic_cache_file, self.num_word_in_topic)
			np.save(self.num_transitions_cache_file, self.num_transitions)
		# Smooth the word counts by additive smoothing
		self.num_word_in_class += self.alpha
		self.num_word_in_topic += self.alpha
		# self.num_transitions += self.alpha
		# Calculate the desired probabilities from counts
		self.P_w_given_c = np.nan_to_num(self.num_word_in_class / np.sum(self.num_word_in_class, axis=0))
		self.P_w_given_t = np.nan_to_num(self.num_word_in_topic / np.sum(self.num_word_in_topic, axis=0))
		self.P_t_given_w = np.nan_to_num(self.num_word_in_topic.T / np.sum(self.num_word_in_topic.T, axis=0))
		self.P_c_given_w = np.nan_to_num(self.num_word_in_class.T / np.sum(self.num_word_in_class.T, axis=0))
		self.P_c_given_prev_c = self.num_transitions.T / np.sum(self.num_transitions, axis=1)

		# Calcualte the probability of Topic class for the next word given previous word
		# Summation(P(C=0| C_prev) * P(C_prev|word))
		self.P_t_given_w_prev = self.P_c_given_prev_c[0,:].dot(self.P_c_given_w)
		print(self.P_t_given_w_prev.shape)
		print "Transition probability matrix"
		print(self.P_c_given_prev_c)
		print(np.sum(self.P_c_given_prev_c, axis=0))
		print(np.sum(self.P_c_given_prev_c, axis=1))

	def read_stop_words(self, stop_words_file):
		stop_words = set()
		with open(stop_words_file, "r") as reader:
			for line in reader:
				line = line.strip()
				if line:
					line_spl = line.split("'")
					if len(line_spl) == 2:
						stop_words.add(line_spl[0])
						stop_words.add("'" + line_spl[1])
					else:
						stop_words.add(line)
		return stop_words

	def __init__(self, vocabulary_file, documents_file, class_assignment_file, topic_assignment_file, stop_words_file, max_documents = None, recount = False):
		start = timeit.default_timer()
		self.max_documents = max_documents
		self.vocabulary_file = vocabulary_file
		self.read_vocabulary()
		self.documents_file = documents_file
		self.read_documents(self.max_documents)
		self.class_assignment_file = class_assignment_file
		self.topic_assignment_file = topic_assignment_file
		self.class_assignments = list()
		self.read_assignments_file(self.class_assignment_file, self.class_assignments, "C", self.max_documents)
		self.START_CLASS = self.num_classes - 2
		self.END_CLASS = self.num_classes - 1
		self.topic_assignments = list()
		self.read_assignments_file(self.topic_assignment_file, self.topic_assignments, "T", self.max_documents)
		self.stop_words_file = stop_words_file
		self.stop_words = self.read_stop_words(self.stop_words_file)
		self.cache = dict()			# A dictionary of tuples which will cache the (word_id, prev_class) pair values
		self.recount = recount		# Tells whether to recompute the counts from read data
		self.alpha = 0.01			# Smoothing the word counts
		self.beta = 0.000001				# Smoothing for the topic prior of the stop words
		self.run_counts()
		print "Total time taken for Syntax and Topics Model initialization = {}sec".format(timeit.default_timer() - start)

	def get_word_id(self, word):
		# Given a word check in vocabulary and return the correct word id if present
		# if the word is of the format word + _ + tag then change it to word
		# print word
		word = word.rsplit('_', 1)[0]
		# print word, "$$"
		if word in self.vocab_stoi:
			# print "Hit", word
			return self.vocab_stoi[word]
		return self.vocab_stoi["<unk>"]

	def clear_cache():
		self.cache = dict()

	def get_class_prior_for_word(self, word):
		word_id = self.get_word_id(word)
		return self.P_c_given_w[:, word_id]

	def get_topic_prior_for_word(self, word):
		word_id = self.get_word_id(word)
		# class_probs = self.P_c_given_w[:, word_id]
		# if class_probs[0] == np.max(class_probs) and any(c.isalpha() for c in word) and "<unk>" not in word:
		# if word == "handsomer":
		# 	print word not in self.stop_words 
		# 	print any(c.isalpha() for c in word)
		# 	print "<unk>" not in word
		# 	print np.array(self.P_t_given_w[:,word_id])
		# 	print self.vocab_itos[word_id]
		# 	print word_id
		# 	print self.num_word_in_topic[word_id]
		# 	print self.num_word_in_class[word_id]
		if word not in self.stop_words and any(c.isalpha() for c in word) and "<unk>" not in word and "s>" not in word:
			prior = list(self.P_t_given_w[:,word_id])
			prior.append(self.beta)
			return np.array(prior)
		else:
			prior = np.full((self.num_topics+1), self.beta, dtype=np.float)
			prior[self.num_topics] = 1.0
			return np.array(prior)


	# Return log probability and the chosen class/topic and type of return
	def get_log_prob(self, word, prev_class):
		# retun log(P(w|C)*P(C|prev_class))
		word_id = self.get_word_id(word)
		if (word_id, prev_class) in self.cache:
			return self.cache[(word_id, prev_class)]
		probs_for_classes = self.P_w_given_c[word_id,:] * self.P_c_given_prev_c[:,prev_class]
		probs_for_classes[0] = 0.0
		best_class = np.argmax(probs_for_classes)
		best_class_prob = probs_for_classes[best_class]
		probs_for_topics = self.P_w_given_t[word_id,:] * self.P_c_given_prev_c[0,prev_class]			# 0 is the topics class
		best_topic = np.argmax(probs_for_topics)
		best_topic_prob = self.beta * probs_for_topics[best_topic]
		return_val = -1000000.0
		if best_topic_prob > best_class_prob:
			try:
				# print "Hit", best_topic_prob, best_class_prob, word, prev_class
				return_val = (math.log(best_topic_prob), best_topic, "T")
			except ValueError:
				if best_topic_prob == 0.0:
					return_val = (-1000000.0, -1, "T")
				else:
					print "Best topic prob is for word {} is topic{} and the value is {}".format(word, best_topic, best_topic_prob)
					print prev_class
					print probs_for_classes
					print probs_for_topics
					exit()
		else:
			try:
				return_val = (math.log(best_class_prob), best_class, "C")
			except ValueError:
				if best_class_prob == 0.0:
					return_val = (-1000000.0, -1, "C")
				else:
					print "Best class prob is for word {} is class{} and the value is {}".format(word, best_class, best_class_prob)
					print prev_class
					print probs_for_classes
					print probs_for_topics
					exit()
		self.cache[(word_id, prev_class)] = return_val
		return return_val

	def KL_divergence(self, P1, P2):
		a = np.asarray(P1, dtype=np.float)
		b = np.asarray(P2, dtype=np.float)
		# Want to do all the computations in 2 loops
		sum_a = 0.0
		sum_b = 0.0
		for i in range(a.shape[0]):
			if a[i] != 0.0 and b[i] != 0.0:
				sum_a += a[i]
				sum_b += b[i]
		if sum_a == 0.0 or sum_b == 0.0:
			return 0.0


		sum = 0.0
		for i in range(a.shape[0]):
			if a[i] != 0.0 and b[i] != 0.0:
				sum += a[i]/sum_a * np.log((a[i] / sum_a) / (b[i] / sum_b))
		return sum

		# return np.sum(np.where(a != 0, a * np.log(a / b), 0))

	def get_weighted_topic_word_probability(self, word, word_prev):
		word_id = self.get_word_id(word)
		word_prev_id = self.get_word_id(word_prev)
		# We hope that this prior will be small for stop words automatically
		# prior = list(self.P_t_given_w[:,word_id] * self.P_c_given_w_prev[word_prev_id])
		prior = list(self.P_t_given_w[:,word_id] * self.P_c_given_w[0,word_id])
		return np.array(prior)
