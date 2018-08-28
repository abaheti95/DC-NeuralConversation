from __future__ import print_function
# changing print precision globally
def print(*args):
	__builtins__.print(*("%.3f" % a if isinstance(a, float) else a
						 for a in args))
# We will read the results file and just print the fishy responses.

import csv
import os
from collections import Counter


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
c_count = 0
total = 0
all_model_results_filename = os.path.join(RESULTS_DIR, "first_batch_all_model_votes.txt")
with open(all_model_results_filename, "w") as all_results_writer:
	for model, model_questions in all_questions.iteritems():
		gold_scores[model] = dict()
		print(model,";", len(model_questions))

		save_filename = os.path.join(RESULTS_DIR, "first_batch_{}_votes.txt".format(model))
		with open(save_filename, "w") as writer:
			for question, votes in model_questions.iteritems():
				# if len(question[1].split()) <= 15:
				# 	continue
				c_scores = [vote for vote in votes]
				c_majority, c_majority_count = Counter(c_scores).most_common()[0]
				if c_majority_count <= (len(c_scores) / 2):
					c_count += 1
					c_majority = 0
					print("C", question, c_scores)
				gold_scores[model][question] = c_majority

				writer.write("{} C:{}:{}\n".format(question, c_majority, c_scores))
				all_results_writer.write("Model:{}\t{} C:{}:{}\n".format(model, question, c_majority, c_scores))
				total += 1
		all_results_writer.write("\n\n")

print(c_count, total)

# Save all the votes in a file
model_seq = ["true", "MMI B=(200)", "TA seq2seq B=10 Bias=2", "tsim esim length penalty B=10", "tsim esim MMI B=10", "tsim esim MMI B=200"]


def count_to_percent(_dict):
	# We get the counts of 1, 0 and -1 and we get the percentage values from that
	percent_dict = dict()
	total = 0.0
	for key, value in _dict.iteritems():
		if key < -1:
			continue
		total += value

	for key, value in _dict.iteritems():
		if key < -1:
			continue
		percent_dict[key] = round(float(value) / total * 100.0, 3)
	return percent_dict

def print_dict_nicely(_dict):
	try:
		print(_dict[-1],",\t", _dict[0],",\t", _dict[1])
	except KeyError:
		print(_dict)

for model, questions_gold in gold_scores.iteritems():
	print(model)
	c_count = g_count = c_count = t_count = 0
	c_dict = dict()
	for qeustion, gold_score in questions_gold.iteritems():
		# print(type(gold_score))
		# print(gold_score)
		c_dict.setdefault(gold_score, 0)
		c_dict[gold_score] += 1
	print(-1, 0, 1)
	print("c",)
	print_dict_nicely(count_to_percent(c_dict))

