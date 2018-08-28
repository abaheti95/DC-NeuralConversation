import csv
import os
from collections import Counter
import numpy as np


RESULTS_DIR = "Mturk2 Movie C Results"
results_files = list()
results_files.append(os.path.join(RESULTS_DIR, "Batch_3213372_batch_results.csv"))
results_files.append(os.path.join(RESULTS_DIR, "Batch_3220337_batch_next_900_results.csv"))
blocked_workers_file = os.path.join(RESULTS_DIR, "blocked_movie_C_workers.txt")

time_column = "WorkTimeInSeconds"
first_input = "Input.input0"
first_model_column = "Input.model0"
AssignmentId = "AssignmentId"
WorkerId = "WorkerId"

def read_blocked_workers(blocked_workers_file):
	blocked_workers = list()
	with open(blocked_workers_file, "r") as reader:
		for line in reader:
			line = line.strip()
			blocked_workers.append(line)
	return blocked_workers

blocked_workers = read_blocked_workers(blocked_workers_file)

all_workers = set()
all_questions = dict()
# model type and dict of questions

for results_file in results_files:
	with open(results_file, "r") as results:
		reader = csv.reader(results)
		header = next(reader)
		time_column_id = header.index(time_column)
		input0_id = header.index(first_input)
		model0_id = header.index(first_model_column)
		worker_id = header.index(WorkerId)
		assignment_id = header.index(AssignmentId)
		C0_id = input0_id+40
		print(header)

		print(header[C0_id])
		print(input0_id, model0_id)
		print(header[model0_id])
		print(header[worker_id])
		print(header[assignment_id])
		# exit()
		count = 0
		count2 = 0
		for row in reader:
			# if row[worker_id] in blocked_workers:
			# 	print(row[time_column_id])
			# 	count2 += 1
			# continue
			# print(row)
			worker = row[worker_id]
			if worker not in blocked_workers and worker not in all_workers:
				all_workers.add(worker)
			# We will extract all the conversations and responses into a tuple
			if worker in blocked_workers:
				continue
			# print(row[time_column_id])
			# print(row[worker_id])
			# print(row[assignment_id])
			for i in range(10):
				source = row[input0_id + i*4]
				response = row[input0_id + 1 + i*4]
				srno = row[input0_id + 2 + i*4]
				model_type = row[input0_id + 3 + i*4]
				C_score = int(row[C0_id + i])
				if C_score > 1:
					C_score = 1
				elif C_score < -1:
					C_score = -1
				# print(srno, model_type, source, response)
				all_questions.setdefault(model_type, dict())
				all_questions[model_type].setdefault((srno, source, response), list())
				all_questions[model_type][(srno, source,response)].append(C_score)
			count += 1
			# print("\n\n")

		print(count)
		print(count2)
		print(len(all_workers))
		print(len(all_questions))

# Compute the Majority votes
# Ref: https://stackoverflow.com/a/20038135/4535284
gold_scores = dict()
p_count = 0
total = 0
for model, model_questions in all_questions.iteritems():
	gold_scores[model] = dict()
	print(model,";", len(model_questions))

	for question, votes in model_questions.iteritems():
		# if len(question[1].split()) <= 6:
		# 	continue
		p_scores = [vote for vote in votes]
		p_majority, p_majority_count = Counter(p_scores).most_common()[0]
		if p_majority_count <= (len(p_scores) / 2):
			p_count += 1
			p_majority = 0
			# print("C", question, p_scores)
		gold_scores[model][question] = p_majority

		total += 1

mmi_model = 'MMI B=(200)'
ta_model = 'TA seq2seq B=10 Bias=2'
our_model = 'tsim esim MMI B=200'

# Take the (question) --> majority vote dict and creates a sample of majority votes of the same size with replacement
def create_sample_scores(dict1):
	n = len(dict1)
	scores = np.array(list(dict1.values()))
	return np.random.choice(scores, scores.shape)

def compute_delta(scores1, scores2):
	# return np.average(scores1) - np.average(scores2)

	scores_1 = np.zeros_like(scores1)
	scores_1[scores1 == 1] = 1
	scores_2 = np.zeros_like(scores2)
	scores_2[scores2 == 1] = 1
	return np.average(scores_1) - np.average(scores_2)

# Computes the significance between models A and B
def significance_test(A, B):
	# First we compute the delta between A and B
	A_scores = np.array(list(gold_scores[A].values()))
	B_scores = np.array(list(gold_scores[B].values()))

	delta = compute_delta(A_scores, B_scores)

	b = 100000
	s = 0.0
	for i in range(b):
		A_scores_sample = create_sample_scores(gold_scores[A])
		B_scores_sample = create_sample_scores(gold_scores[B])

		delta_sample = compute_delta(A_scores_sample, B_scores_sample)

		if delta_sample > 2 * delta:
			s += 1.0
	return delta, s/float(b)

import time
start = time.time()
np.random.seed()
print significance_test(our_model, mmi_model)
print significance_test(our_model, ta_model)
print significance_test(mmi_model, ta_model)






























