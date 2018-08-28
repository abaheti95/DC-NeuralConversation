# We will generate csv file for the 100 sentences from 6 model outputs

import csv
import os
from itertools import izip
from random import shuffle

DATA_DIR = "MTurk2 model responses"
RESULTS_DIR = "MTurk2 Results"
save_file = os.path.join(RESULTS_DIR, "mturk2_next_900_sentences.csv")
source_file = os.path.join(DATA_DIR, "s_cornell_mturk2_test.txt")

responses_files = ["t_cornell_mturk2_test.txt",
					"MMI_predictions_cornell_mturk2_test.txt",
					"ta_seq2seq_B10_bias_2_predictions_cornell_mturk2_test.txt",
					"full_model_tsim_esim_B10_length_decoding_cornell_mturk2_test_predictions.txt",
					"full_model_tsim_esim_B10_MMI_decoding_cornell_mturk2_test_predictions.txt",
					"full_model_tsim_esim_B200_MMI_decoding_cornell_mturk2_test_predictions.txt"]
responses_files = [os.path.join(DATA_DIR, filename) for filename in responses_files]

model_types = ["true",
				"MMI B=(200)",
				"TA seq2seq B=10 Bias=2",
				"tsim esim length penalty B=10",
				"tsim esim MMI B=10",
				"tsim esim MMI B=200"]

first_row = ["input0","output0","srno0","model0",
			"input1","output1","srno1","model1",
			"input2","output2","srno2","model2",
			"input3","output3","srno3","model3",
			"input4","output4","srno4","model4",
			"input5","output5","srno5","model5",
			"input6","output6","srno6","model6",
			"input7","output7","srno7","model7",
			"input8","output8","srno8","model8",
			"input9","output9","srno9","model9"]

def get_unbaised_sample(file):
	srno_lines = list()
	# 1-334 is from b1
	# 335 - 667 is from b2
	# 668 - 1000 is from b3
	# indices = list(range(34))
	# indices.extend(list(range(334, 334+33)))
	# indices.extend(list(range(667, 667+33)))
	indices = list(range(1000))
	with open(file, "r") as reader:
		for i, line in enumerate(reader):
			if i in indices:
				if i<100:
					continue
				line = line.strip()
				srno_lines.append((i, line))
	return srno_lines

def get_list_of_srno_lines(file):
	lines = list()
	with open(file, "r") as reader:
		for i, line in enumerate(reader):
			if i<100:
				continue
			line = line.strip()
			lines.append(line)
	return lines

def save_all_pairs_to_file(file, all_pairs):
	with open(file, "w") as writer:
		for source, response, srno, model_type in all_pairs:
			writer.write("{}\t{}::{}\t{}\n".format(srno, source, response, model_type))

def get_all_model_results(save_file):
	all_pairs = list()
	source_lines = get_unbaised_sample(source_file)
	for j, filename in enumerate(responses_files):
		responses_lines = get_unbaised_sample(filename)
		if len(source_lines) == len(responses_lines) and len(source_lines)%10 == 0:
			# Create Tuple and add to global list
			for srno_source, srno_response in izip(source_lines, responses_lines):
				source = srno_source[1]
				response = srno_response[1]
				if srno_source[0] == srno_response[0]:
					srno = srno_source[0]
				else:
					print "Error", srno_source, srno_response
					exit()
				all_pairs.append((source, response, srno, model_types[j]))
		else:
			print "Something is wrong with source file {} or the response file {}".format(source_file, file)

	save_all_pairs_to_file("next_900_mturk2_results.txt", all_pairs)
	# exit()
	shuffle(all_pairs)
	shuffled_all_pairs = all_pairs
	print "Saving to ", save_file
	with open(save_file, "w") as save_file_writer:
		writer = csv.writer(save_file_writer, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
		writer.writerow(first_row)
		# chunk and create a batch from the shuffled pairs
		chunk = list()
		for source, response, srno, model_type in shuffled_all_pairs:
			# Chunk 10 reponses into one list
			chunk.append(source)
			chunk.append(response)
			chunk.append(srno)
			chunk.append(model_type)
			if len(chunk) == 40:
				writer.writerow(chunk)
				chunk = list()

get_all_model_results(save_file)


# with open(save_file, "w") as save_file_writer:
# 	writer = csv.writer(save_file_writer, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)
# 	writer.writerow(first_row)
# 	source_lines = get_list_of_lines(source_file)
# 	for j, file in enumerate(responses_files):
# 		responses_lines = get_list_of_lines(file)
# 		if len(source_lines) == len(responses_lines) and len(source_lines)%10 == 0:
# 			# Chunk 10 lines into one list
# 			chunk = list()
# 			for i in range(len(source_lines)):
# 				chunk.append(source_lines[i])
# 				chunk.append(responses_lines[i])
# 				if (i+1)%10 == 0:
# 					chunk.append(model_types[j])
# 					chunk.append("pilot")
# 					writer.writerow(chunk)
# 					chunk = list()
# 		else:
# 			print "Something is wrong with source file {} or the response file {}".format(source_file, file)

