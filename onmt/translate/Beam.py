from __future__ import division
import torch
from torch.autograd import Variable
import onmt.io

import numpy as np
#Adding the topic and syntax probability scores to the scores from Decoder

class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
        size (int): beam size
        pad, bos, eos (int): indices of padding, beginning, and ending.
        vocab (vocab): vocab of the target
        syntax_topics_model (Syntax and Topics module): This is an object of the class which will have the topics 
                                                        and classes word probabilities
        source (list): list of source indices which will be the source sentence
        targets (list of list): list of taget sentences each of which is a list of indices of the full hypothesis generated till now
        n_best (int): nbest size to use
        cuda (bool): use gpu
        global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos,
                 vocab, similarity_scorer, syntax_topics_model,
                 source,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos
        self.next_ys_topic_prior_sum = []        # Store the sum of the weighted topic prior for the entire hypothesis

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False
        self._bos = bos

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length
        # self.min_length = 10



        ##NOTE: Custom code
        self.vocab = vocab
        self.finished_marker = [-1]*size
        self.similarity_scorer = similarity_scorer
        self.syntax_topics_model = syntax_topics_model
        self.source = source
        self.source_text = [self.vocab.itos[word_id] for word_id in source]
        print self.source_text
        self.source_topic_prior = np.zeros(self.syntax_topics_model.num_topics, dtype=np.float)
        word_prev = "<s>"
        for word in self.source_text:
            self.source_topic_prior += self.syntax_topics_model.get_weighted_topic_word_probability(word, word_prev)
            word_prev = word
        self.source_topic_prior /= float(np.linalg.norm(self.source_topic_prior))
        # print self.source_topic_prior
        self.L = 1000         # Number of words to be chosen for the similarity consideration
        # Note: alpha 0.5 was used for the interpolation objective
        # self.alpha = 0.5        # multiplicative factor for the Syntax Topic log probability
        # self.alpha = 1.0        # multiplicative factor for the Syntax Topic log probability
        self.alpha = 1.5        # multiplicative factor for the Syntax Topic log probability
        # self.alpha = 0.0        # multiplicative factor for the Syntax Topic log probability
        # self.iteration_multiplier = 0.75
        self.iteration_multiplier = 1
        #TODO: Changing similarity to 0 and seeing the changes
        # self.gamma = 1.5
        self.gamma = 2.0
        # self.gamma = 0.0
        self.delta = 3.5        # repeatition penalty
        # self.delta = 2.5        # repeatition penalty
        # self.delta = 0.0        # repeatition penalty
        self.theta = 0.05       # short length response penalty in final scoring

        # import os
        # stop_words_file = os.path.join("data", "syntax_topics_models", "stop_words_short.txt")
        # self.stop_words_set = self.read_words_from_file(stop_words_file)
        # self.stop_words_set.add("<unk>")
        # topic_words_file = os.path.join("data", "syntax_topics_models", "top10_topic_words.txt")
        # self.topic_words_set = self.read_words_from_file(topic_words_file)
        # print self.stop_words_set
        # print self.topic_words_set
        # print vocab.itos[0]
        # print vocab.itos[bos]
        # print vocab.itos[self._eos]
        # print vocab.__dict__.keys()
        # print vocab.unk_init
        # print type(vocab)

    # Temporary code for reading stop words and topic words list from files. Will be used to draw the figure later
    def read_words_from_file(self, filename):
        words_set = set()
        with open(filename, "r") as reader:
            for line in reader:
                line = line.strip()
                if line and line != "UNknown":
                    words_set.add(line)
        return words_set


    def print_hyp(self, hyp, class_or_topic = None):
        for i, word_id in enumerate(hyp):
            if class_or_topic:
                print "{}_{} ".format(self.vocab.itos[word_id], class_or_topic[i]),
            else:
                print "{} ".format(self.vocab.itos[word_id]),

    ##NOTE: Custom function for debugging
    def print_all_words_in_beam(self):
        # Uses the vocab and prints the list of next_ys
        for i in range(self.size):
            timestep = len(self.next_ys)
            # if self.finished_marker[i] != -1:
            #     timestep = self.finished_marker[i]
            if timestep > 1:
                hyp, _ = self.get_hyp(timestep, i)
                self.print_hyp(hyp)
                print "$$$"
            # print ""

    def print_received_targets(self):
        for i in range(len(self.targets)):
            self.print_hyp(self.targets[i])
            print ""
        print "############### Received Targets ############"

    def print_the_top_choices(self, best_choices):
        w, h = best_choices.size()
        for i in range(w):
            for j in range(h):
                print "{} ".format(self.vocab.itos[best_choices[i,j]]),
            print ""


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def delay_function(self, iteration):
        if iteration<=1:
            return 0.3
        else:
            return 1
        # http://www.wolframalpha.com/input/?i=y%3D(tanh((x-3))+%2B1)%2F2
        # return (np.tanh((iteration-3.0) * self.iteration_multiplier) + 1)/2.0

    def delay_function2(self, iteration):
        if iteration<=2:
            return .8
        elif iteration<=5:
            return 0.6
        else:
            return 0.2

    def advance(self, word_probs, attn_out):
        print "Advancing beam"
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]

        #TODO: 1) find the last word for each beam
        #       2) For each possible hypothesis find the Topic modeling probability as Source sentence weighted topic prior * current word weighted topic prior
        #       3) Weighted add the syntax and topic scores to beam_scores and just calculate the next_ys and backpointers as normal
        
        if len(self.prev_ks) > 0:
            per_beam_best_scores, per_beam_best_scores_ids = beam_scores.topk(self.L)
            # Tensors to store the similarity scores
            source_similarity_scores = self.tt.zeros_like(per_beam_best_scores)

            per_beam_words = self.next_ys[-1]
            per_beam_prev_topic_probability = self.next_ys_topic_prior_sum[-1]
            # print "Next iter"

            syntax_topic_scores = self.tt.zeros_like(per_beam_best_scores)

            # We want to penalize the words which are occuring again in the sentence
            repetition_penalty = self.tt.zeros_like(per_beam_best_scores)
            timestep = len(self.next_ys)
            unk_ids = list()
            # timestep_no = None
            # if timestep == timestep_no:
            #     beam_no = 6
            #     stop_words_scores = dict()
            #     topic_words_scores = dict()

            for i in range(self.size):
                word_prev = self.vocab.itos[per_beam_words[i]]
                hyp_topic_probability = per_beam_prev_topic_probability[i]
                # Generate the curret hypothesis of this beam
                hyp, _ = self.get_hyp(timestep, i)
                # Now we append a dummy value which will be replaced below as the final word in the hypothesis
                hyp.append(0)
                for j in range(self.L):
                    word_id = per_beam_best_scores_ids[i,j]
                    hyp[-1] = word_id
                    word = self.vocab.itos[word_id]
                    # Manually discourage unk by setting large negative syntax_topic_probability
                    if "<unk>" in word:
                        unk_ids.append((i,j))
                        syntax_topic_score = -100000.0
                        syntax_topic_scores[i,j] = -100000.0
                        source_similarity_scores[i,j] = -100000.0
                    else:
                        syntax_topic_score = (self.syntax_topics_model.get_weighted_topic_word_probability(word, word_prev) + hyp_topic_probability).dot(self.source_topic_prior)
                        syntax_topic_scores[i,j] = syntax_topic_score
                        source_similarity_scores[i,j] = self.similarity_scorer.Similarity(self.source, hyp)

                    # if timestep == timestep_no:
                    #     # Temporary code for the first figure in the paper
                    #     if word in self.stop_words_set:
                    #         stop_words_scores[word] = per_beam_best_scores[beam_no,j]
                    #     if word in self.topic_words_set:
                    #         topic_words_scores[word] = per_beam_best_scores[beam_no,j]
                    # if "</s>" in word:
                    #     # print syntax_topic_scores[i,j], self.tt.max(syntax_topic_scores), self.tt.min(syntax_topic_scores)
                    #     syntax_topic_scores[i,j] /= 4.0
                    # Count the number of times the word has occured in the hypothesis
                    # reweight it with syntax topic score so that score is high only for topic words
                    repetition_penalty[i,j] = abs(hyp[:-1].count(word_id) * self.alpha * syntax_topic_score)

                # if timestep == timestep_no:
                #     # print hypothesis
                #     curr_hyp, _ = self.get_hyp(len(self.next_ys) - 1,beam_no)
                #     word_hyp = self.hyp_to_words(curr_hyp)
                #     print word_hyp
                #     import operator
                #     sorted_topic_words_scores = sorted(topic_words_scores.items(), key=operator.itemgetter(1), reverse=True)
                #     sorted_stop_words_scores = sorted(stop_words_scores.items(), key=operator.itemgetter(1), reverse=True)
                #     for i, (word, score) in enumerate(sorted_stop_words_scores):
                #         if i == 20:
                #             break
                #         print word, ":", score
                #     print ""
                #     for i, (word, score) in enumerate(sorted_topic_words_scores):
                #         # if i == 20:
                #         #     break
                #         print word, ":", score
                #     print ""
                #     exit()

            syntax_topic_scores /= float(len(self.source) + len(self.next_ys))
            # Note: Probability interpolation doesn't work because the model then chooses all the sentences mainly from the RNN and barely any words from the topic part are chosen
            # overall_scores = self.tt.log(self.tt.exp(beam_scores) + self.alpha * syntax_topic_scores)
            iteration = len(self.prev_ks)
            overall_scores = per_beam_best_scores + (self.delay_function(iteration) * self.alpha * self.tt.log(syntax_topic_scores)) + (self.gamma * iteration * source_similarity_scores) - self.delta * repetition_penalty
            # adaptive_alpha = beam_scores.max() / self.tt.log(syntax_topic_scores).max() * 0.5
            # overall_scores = beam_scores + adaptive_alpha * self.tt.log(syntax_topic_scores)

            for i, j in unk_ids:
                overall_scores[i,j] = -1000000.0

            beam_mean = beam_scores.topk(30)[0].mean()
            syntax_topic_scores_mean = self.tt.log(syntax_topic_scores.topk(30)[0]).mean()
            # print adaptive_alpha
            # print beam_mean, syntax_topic_scores_mean, beam_mean / syntax_topic_scores_mean
            # print beam_scores.max(), self.tt.log(syntax_topic_scores).max(), beam_scores.max() / self.tt.log(syntax_topic_scores).max()
            # print beam_scores.min(), self.tt.log(syntax_topic_scores).min(), beam_scores.min() / self.tt.log(syntax_topic_scores).min()
            # print "Overall Score"
            # print overall_scores
            # print overall_scores.size()
        
            # self.print_all_words_in_beam()
            # print "############## Before new words chosen ###########"

            size = int(overall_scores.size(1))
            # print "Size of the overall_scores = ", size
            flat_beam_scores = overall_scores.view(-1)
            # flat_beam_scores = syntax_topic_scores.view(-1)
            # flat_beam_scores = per_beam_best_scores.view(-1)
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)
            # We will debug the individual scores of the best candidates
            # Only keep the word probability part of the best score
            word_prob_best_scores = self.tt.zeros_like(best_scores)
            for i in range(self.size):
                word_prob_best_scores[i] = per_beam_best_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]
            # Find Syntax topic scores of the best candidates
            syntax_topic_best_scores = self.tt.zeros_like(best_scores)
            for i in range(self.size):
                syntax_topic_best_scores[i] = syntax_topic_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]
            # Find the similarity score component of the best candidates
            best_hyp_similairty_scores = self.tt.zeros_like(best_scores)
            for i in range(self.size):
                best_hyp_similairty_scores[i] = source_similarity_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]


            # print best_scores
            # print word_prob_best_scores
            # print best_hyp_similairty_scores
            # print self.gamma * self.delay_function(iteration)
            # print self.gamma * self.delay_function(iteration) * best_hyp_similairty_scores
            # print syntax_topic_best_scores
            # print self.tt.log(syntax_topic_best_scores)
            # print iteration, self.alpha * self.delay_function(iteration)
            # print self.alpha * self.delay_function(iteration) * self.tt.log(syntax_topic_best_scores)

            prev_k = best_scores_id / size

            self.prev_ks.append(prev_k)
            # Recalculate the next_ys in a new for loop
            next_y = list()
            for i in range(self.size):
                # print per_beam_best_scores_ids[prev_k[i]][best_scores_id[i] - prev_k[i] * size]
                next_y.append(per_beam_best_scores_ids[prev_k[i]][best_scores_id[i] - prev_k[i] * size])
            next_y = self.tt.LongTensor(next_y)
            
            best_hyp_topic_prior_sum = list()
            for i in range(self.size):
                word = self.vocab.itos[next_y[i]]
                word_prev_id = self.next_ys[-1][prev_k[i]]
                word_prev = self.vocab.itos[word_prev_id]
                word_topic_prior = self.syntax_topics_model.get_weighted_topic_word_probability(word, word_prev)
                best_hyp_topic_prior_sum.append(self.next_ys_topic_prior_sum[-1][int(best_scores_id[i] / size)] + word_topic_prior)
            self.next_ys_topic_prior_sum.append(best_hyp_topic_prior_sum)

            self.next_ys.append(next_y)
            self.attn.append(attn_out.index_select(0, prev_k))
            # exit()
            # for i in range(self.size):
            #     word_id = next_y[i]
            #     word = self.vocab.itos[word_id]
            #     print word, syntax_topic_best_scores[i], word_prob_best_scores[i]
            # print ""
            # self.print_all_words_in_beam()
            # print "############## After new words chosen ###########"
        else:
            # beam_scores is only V dimensional vector
            # Thus for every word add the syntax_topic probability to the beam score
            syntax_topic_scores = self.tt.zeros_like(beam_scores)
            for i in range(num_words):
                word = self.vocab.itos[i]
                if "<unk>" in word:
                    unk_id = i
                syntax_topic_score = self.syntax_topics_model.get_weighted_topic_word_probability(word, "<s>").dot(self.source_topic_prior)
                syntax_topic_scores[i] = syntax_topic_score

            # overall_scores = beam_scores + self.alpha * self.tt.log(syntax_topic_scores)
            # NOTE: Not calculating the syntax topic or the similarity score for the first word
            overall_scores = beam_scores
            # print "Unk id is", unk_id
            overall_scores[unk_id] = -100000.0

            flat_beam_scores = overall_scores.view(-1)
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)

            self.all_scores.append(self.scores)
            self.scores = best_scores           # will store the word_prob + the topic log prob for hypothesis

            # best_scores_id is flattened, beam * word array, so calculate which
            # word and beam each score came from
            size = int(overall_scores.size(0))
            prev_k = best_scores_id / num_words
            # next_class_or_topic = self.tt.zeros_like(prev_k)
            # for i in range(self.size):
            #     next_class_or_topic[i] = int(class_or_topics[best_scores_id[i]])
            # print prev_k
            # print overall_scores
            # print next_class_or_topic
            # print next_class_or_topic_number
            # print best_scores_id

            best_hyp_topic_prior_sum = list()
            for i in range(self.size):
                word = self.vocab.itos[best_scores_id[i]]
                word_topic_prior = self.syntax_topics_model.get_weighted_topic_word_probability(word, "<s>")
                best_hyp_topic_prior_sum.append(word_topic_prior)
            self.next_ys_topic_prior_sum.append(best_hyp_topic_prior_sum)

            self.prev_ks.append(prev_k)
            self.next_ys.append((best_scores_id - prev_k * num_words))
            self.attn.append(attn_out.index_select(0, prev_k))

        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                timestep = len(self.next_ys)
                # MMI Objective stuff
                curr_hyp, _ = self.get_hyp(len(self.next_ys) - 1,i)
                word_hyp = self.hyp_to_words(curr_hyp)
                # s = self.scores[i]
                # s = self.scores[i] + (self.alpha * np.log(syntax_topic_best_scores[i])) + self.gamma * best_hyp_similairty_scores[i]
                # s = (self.alpha * np.log(syntax_topic_best_scores[i])) + self.gamma * best_hyp_similairty_scores[i] + self.theta * timestep

                word_prev = "<s>"
                hyp_topic_prior = np.zeros(self.syntax_topics_model.num_topics, dtype=np.float)
                for word in word_hyp:
                    hyp_topic_prior += self.syntax_topics_model.get_weighted_topic_word_probability(word, word_prev)
                    word_prev = word
                hyp_topic_prior /= np.linalg.norm(hyp_topic_prior)
                
                # s = (np.log(hyp_topic_prior.dot(self.source_topic_prior)))
                # print s, self.theta * timestep
                s = self.alpha * (np.log(hyp_topic_prior.dot(self.source_topic_prior))) + self.theta * timestep
                # s = self.scores[i]
                # s = (self.alpha * self.delay_function(iteration) * np.log(syntax_topic_best_scores[i])) + self.gamma * best_hyp_similairty_scores[i]
                # s = (self.alpha * np.log(syntax_topic_best_scores[i])) - (2 * (len(self.source) - iteration) if iteration < len(self.source) else 0.5 * (len(self.source) - iteration))
                # s = - (1 * (len(self.source) - iteration) if iteration < len(self.source) else 0.0)
                # s = self.gamma * best_hyp_similairty_scores[i] - (0.5 * (len(self.source) - iteration) if iteration < len(self.source) else 0.0)
                if self.global_scorer is not None:
                    s -= self.theta * timestep
                    # global_scores = self.global_scorer.score(self, self.scores)
                    # s = global_scores[i]
                    # MMI score
                    global_score = self.global_scorer.mmi_score(self, word_hyp, s)
                    s = global_score[0] / float(len(self.next_ys) - 1)

                self.finished.append((s, len(self.next_ys) - 1, i))
                # TODO: Experimental!! Dividing the finished scores by their lenghts to be fair
                # self.finished.append((s/float(len(self.next_ys) - 1), len(self.next_ys) - 1, i))


                ##NOTE: Custom code
                if self.finished_marker[i] == -1:
                    # print "SET AND FORGET FOR ", i, "#$#$#$#$##$"
                    self.finished_marker[i] = len(self.next_ys) - 1



        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            # self.all_scores.append(self.scores)
            self.eos_top = True



        ##NOTE: Debugging
        # print word_probs, "$$"
        # print self.get_current_state(), "$$"        
        # self.print_all_words_in_beam()
        # print "############## Beam Advance ###########"

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # print self.finished
        # exit()

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp_with_class(self, timestep, k):
        """
        Walk back to construct the full hypothesis while also storing the class/topic number
        """
        hyp, class_or_topic = [], []
        # print len(self.next_ys), len(self.next_class_or_topics), len(self.next_class_or_topic_numbers)
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            class_or_topic.append("{}{}".format("C" if self.next_class_or_topics[j][k] == 1 else "T", self.next_class_or_topic_numbers[j][k]))
            k = self.prev_ks[j][k]
        class_or_topic.reverse()
        return hyp[::-1], class_or_topic

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

    def hyp_to_words(self, hyp):
        return [self.vocab.itos[e] for e in hyp]

class MMIGlobalScorer(object):
    """
    Global scorer which rescores the candidates in beam using the (S|T) model
    ref: http://www.aclweb.org/anthology/N16-1014
    """
    def __init__(self, model, fields, cuda=False):
        self.model = model
        self.fields = fields
        self.src_vocab = self.fields["src"].vocab
        self.tgt_vocab = self.fields["tgt"].vocab
        self.tt = torch.cuda if cuda else torch
        self.beta = .0 # Topic score multiplier
        self.alpha = 1.0 # S|T score multiplier
        self.gamma = 0.1 # Length penalty multipler
        # self.gamma = 0.0 # Length penalty multipler

    def make_features(self, data):
        levels = [data]
        return torch.cat([level.unsqueeze(2) for level in levels], 2)


    def _run_target(self, tgt_data, src_data, tgt_lengths):
        tgt = Variable(self.make_features(tgt_data))
        src_in = Variable(self.make_features(src_data)[:-1])
        # print "tgt"
        # print tgt
        # print "tgt_lengths"
        # print tgt_lengths
        # print "src_in"
        # print src_in
        #  (1) run the encoder on the tgt
        enc_states, memory_bank = self.model.encoder(tgt, tgt_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(tgt, memory_bank, enc_states)

        #  (2) Compute the 'goldScore'
        #  (i.e. log likelihood) of the source given target under the model (S|T)
        gold_scores = self.tt.FloatTensor(1).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            # src_in, memory_bank, dec_states, memory_lengths=tgt_lengths)      #NOTE: For some unknown reasons it doesn't want memory_lenghts here
            src_in, memory_bank, dec_states)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, src in zip(dec_out, src_data[1:]):         # src data is src + bos and eos
            # Log prob of each word.
            # print "dec"
            # print dec
            # print "src"
            # print src
            # print self.model.generator.forward
            out = self.model.generator.forward(dec)
            src = src.unsqueeze(1)
            scores = out.data.gather(1, src)
            scores.masked_fill_(src.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def tgt_to_index(self, tgt):
        return [self.tgt_vocab.stoi[word] for word in tgt]

    def src_to_index(self, src):
        return [self.src_vocab.stoi[word] for word in src]
    
    def mmi_score(self, beam, hyp_word, prediction_score):
        # return 1
        # Get the source
        src_list = self.tgt_to_index(beam.source_text) # beam's source is target here
        hyp_list = self.src_to_index(hyp_word)
        # Src list to torch LongTensor 
        src_list.insert(0, beam._bos)
        src_list.append(beam._eos)
        src_data = self.tt.LongTensor(src_list)
        hyp_lengths = self.tt.LongTensor([len(hyp_list) - 1])
        

        hyp = self.tt.LongTensor(hyp_list[:-1])         # Remove EOS from the hyp which is the src to the MMI model
        src_data.unsqueeze_(1)
        hyp.unsqueeze_(1)
        # print hyp
        score = self._run_target(hyp, src_data, hyp_lengths) / float(len(src_list))
        # print hyp_word
        # print score
        # print self.beta * prediction_score, self.alpha * float(score), self.gamma * len(hyp_word)
        # print hyp_word
        # print ""
        return self.beta * prediction_score + self.alpha * score + self.gamma * len(hyp_word)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])




































