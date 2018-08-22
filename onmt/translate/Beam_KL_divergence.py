from __future__ import division
import torch

# Beam decoding with KL divergence
import numpy as np
from scipy.stats import entropy

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
                 vocab, syntax_topics_model,
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
        ##NOTE: speical marker which tells which class is the previous word from
        self.next_ys_topic_prior_sum = []        # Store the sum of the topic prior for the entire hypothesis
        self.next_ys_class_prior_sum = []        # Store the sum of the class prior for the entire hypothesis
        self.TOPIC_FLAG = 0
        self.CLASS_FLAG = 1


        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

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



        ##NOTE: Custom code
        self.vocab = vocab
        self.finished_marker = [-1]*size
        self.syntax_topics_model = syntax_topics_model
        self.source = [self.vocab.itos[word_id] for word_id in source]
        # Compute the topic prior probability for the source sentence
        self.source_topic_prior = np.zeros(self.syntax_topics_model.num_topics+1, dtype=np.float)
        print self.source
        self.src_topic_word_count = 0
        for word in self.source:
            word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
            if word_topic_prior[self.syntax_topics_model.num_topics] != 1.0:
                print word
                self.src_topic_word_count += 1
                self.source_topic_prior += word_topic_prior
        self.source_class_prior = np.zeros(self.syntax_topics_model.num_classes, dtype=np.float)
        for word in self.source:
            self.source_class_prior += self.syntax_topics_model.get_class_prior_for_word(word)
        
        self.source_topic_prior /= len(self.source)
        self.source_class_prior /= len(self.source)
        self.beta = 0.0         # Additive smoothing
        self.source_topic_prior += self.beta
        self.source_class_prior += self.beta
        print(self.source_topic_prior)
        print(self.source_class_prior)
        self.L = 50         # Number of words to be chosen for the similarity consideration
        # temp = np.zeros_like(self.source_topic_prior) + self.beta
        # least_KL = entropy(self.source_topic_prior, temp)
        self.alpha = 1.5                # Multiplicative factor for topic KL divergence
        self.gamma = 1.5                 # Multiplicative factor for class KL divergence
        # self.alpha = 8/least_KL        # multiplicative factor for the KL divergence
        # print vocab.itos[0]
        # print vocab.itos[bos]
        # print vocab.itos[self._eos]
        # print vocab.__dict__.keys()
        # print vocab.unk_init
        # print type(vocab)
        self.topic_KL_flag = False
        self.class_KL_flag = True

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
                # print hyp
                # print type(hyp)
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
        #       2) For each possible hypothesis find the Topic modeling probability
        #       3) Weighted add the syntax and topic scores to beam_scores and just calculate the next_ys and backpointers as normal
        
        if len(self.prev_ks) > 0:
            per_beam_words = self.next_ys[-1]
            if self.topic_KL_flag:
                per_beam_hyp_topic_prior_sum = self.next_ys_topic_prior_sum[-1]
            if self.class_KL_flag:
                per_beam_hyp_class_prior_sum = self.next_ys_class_prior_sum[-1]

            if self.topic_KL_flag:
                topic_KL_divergence_scores = self.tt.zeros_like(beam_scores)
            if self.class_KL_flag:
                class_KL_divergence_scores = self.tt.zeros_like(beam_scores)
            for i in range(self.size):
                if self.topic_KL_flag:
                    hyp_topic_prior = per_beam_hyp_topic_prior_sum[i]
                if self.class_KL_flag:
                    hyp_class_prior = per_beam_hyp_class_prior_sum[i]

                len_hyp = len(self.next_ys)             # Includes current word because we want to skip the count added by the start word <bos>
                for j in range(num_words):
                    word = self.vocab.itos[j]
                    #KL divergence for Topic Priors
                    if self.topic_KL_flag:
                        word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
                        hyp_topic_prior_sum = (hyp_topic_prior + word_topic_prior) + self.beta
                        # topic_KL_divergence_scores[i][j] = entropy(self.source_topic_prior, hyp_topic_prior_sum)
                        topic_KL_divergence_scores[i][j] = self.syntax_topics_model.KL_divergence(self.source_topic_prior, hyp_topic_prior_sum)
                        if topic_KL_divergence_scores[i][j] == float('Inf'):
                            print word
                            print topic_KL_divergence_scores[i][j]
                            print self.source_topic_prior
                            print hyp_topic_prior_sum
                    #KL divergence for Class Priors
                    if self.class_KL_flag:
                        if "<unk>" in word:
                            class_KL_divergence_scores[i][j] = 1000000.0
                            continue
                        word_class_prior = self.syntax_topics_model.get_class_prior_for_word(word)
                        hyp_class_prior_sum = (hyp_class_prior + word_class_prior) + self.beta
                        # class_KL_divergence_scores[i][j] = entropy(self.source_class_prior, hyp_class_prior_sum)
                        class_KL_divergence_scores[i][j] = self.syntax_topics_model.KL_divergence(self.source_class_prior, hyp_class_prior_sum)
                        if class_KL_divergence_scores[i][j] == float('Inf'):
                            print word
                            print class_KL_divergence_scores[i][j]
                            print self.source_class_prior
                            print hyp_class_prior_sum

            #TODO: Convert the zeros to max KL divergence
            if self.topic_KL_flag:
                max_topic_KL = topic_KL_divergence_scores.mean()
                for i in range(self.size):
                    for j in range(num_words):
                        word = self.vocab.itos[j]
                        if "<unk>" in word:
                            topic_KL_divergence_scores[i][j] = 1000000.0
                            continue
                        if topic_KL_divergence_scores[i][j] == 0.0:
                            topic_KL_divergence_scores[i][j] = max_topic_KL
                print "########\n", max_topic_KL, "\n#########"
                    # Manually discourage unk by setting large negative syntax_topic_probability
                    # if "<unk>" in word:
                    #     syntax_topic_score, best_class_or_topic, class_or_topic = -1000000.0, -1, "C"
                    # else:
                    #     syntax_topic_score, best_class_or_topic, class_or_topic = self.syntax_topics_model.get_log_prob(word, 0 if per_beam_words_class_or_topic[i] == self.TOPIC_FLAG else per_beam_words_class_or_topic_number[i])
            overall_scores = beam_scores - ((self.alpha * topic_KL_divergence_scores) if self.topic_KL_flag else 0.0) - ((self.gamma * class_KL_divergence_scores) if self.class_KL_flag else 0.0)
            # print "Overall Score"
            # print overall_scores
            # print overall_scores.size()
        

            size = int(overall_scores.size(1))
            # print "Size of the overall_scores = ", size
            flat_beam_scores = overall_scores.view(-1)
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)

            # We will debug the individual scores of the best candidates
            word_prob_best_scores = self.tt.zeros_like(best_scores)
            for i in range(self.size):
                word_prob_best_scores[i] = beam_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]
            if self.topic_KL_flag:
                topic_KL_divergence_best_scores = self.tt.zeros_like(best_scores)
                for i in range(self.size):
                    topic_KL_divergence_best_scores[i] = topic_KL_divergence_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]
            if self.class_KL_flag:
                class_KL_divergence_best_scores = self.tt.zeros_like(best_scores)
                for i in range(self.size):
                    class_KL_divergence_best_scores[i] = class_KL_divergence_scores[int(best_scores_id[i] / size)][best_scores_id[i] - int(best_scores_id[i] / size) * size]
            print best_scores
            print word_prob_best_scores
            # print topic_KL_divergence_best_scores
            # print class_KL_divergence_best_scores
            if self.topic_KL_flag:
                print self.alpha
                print self.alpha * topic_KL_divergence_best_scores
            if self.class_KL_flag:
                print self.gamma
                print self.gamma * class_KL_divergence_best_scores

            if self.topic_KL_flag:
                KL_size = int(topic_KL_divergence_scores.size(1))
                # print "Size of the overall_scores = ", size
                flat_beam_scores = topic_KL_divergence_scores.view(-1)
                debug_size = 25
                best_topic_KL_scores, best_topic_KL_scores_id = flat_beam_scores.topk(debug_size, 0,
                                                                False, True)
                print "Best words from topic KL"
                for i in range(debug_size):
                    # print best_topic_KL_scores_id[i], best_topic_KL_scores_id[i] - int(best_topic_KL_scores_id[i] / KL_size) * KL_size
                    word = self.vocab.itos[best_topic_KL_scores_id[i] - int(best_topic_KL_scores_id[i] / KL_size) * KL_size]
                    word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
                    # print word, word_topic_prior, entropy(self.source_topic_prior, word_topic_prior + self.beta), \
                    print word, \
                                overall_scores[int(best_topic_KL_scores_id[i] / KL_size)][best_topic_KL_scores_id[i] - int(best_topic_KL_scores_id[i] / KL_size) * KL_size], \
                                best_topic_KL_scores[i], \
                                topic_KL_divergence_scores[int(best_topic_KL_scores_id[i] / KL_size)][best_topic_KL_scores_id[i] - int(best_topic_KL_scores_id[i] / KL_size) * KL_size]
                                # class_KL_divergence_scores[int(best_topic_KL_scores_id[i] / KL_size)][best_topic_KL_scores_id[i] - int(best_topic_KL_scores_id[i] / KL_size) * KL_size], \
                # print word_prob_best_scores + self.alpha * syntax_topic_best_scores

            prev_k = best_scores_id / num_words
            # Update next_ys_topic_prior_sum for all beams
            if self.topic_KL_flag:
                best_hyp_topic_prior_sum = list()
                for i in range(self.size):
                    word = self.vocab.itos[best_scores_id[i] - int(best_scores_id[i] / size) * size]
                    word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
                    if word_topic_prior[self.syntax_topics_model.num_topics] != 1.0:
                        best_hyp_topic_prior_sum.append(self.next_ys_topic_prior_sum[-1][int(best_scores_id[i] / size)] + word_topic_prior)
                    else:
                        # Add the word_topic_prior only if its a topic word
                        best_hyp_topic_prior_sum.append(self.next_ys_topic_prior_sum[-1][int(best_scores_id[i] / size)])
                self.next_ys_topic_prior_sum.append(best_hyp_topic_prior_sum)
            # Update next_ys_class_prior_sum for all beams
            if self.class_KL_flag:
                best_hyp_class_prior_sum = list()
                for i in range(self.size):
                    word = self.vocab.itos[best_scores_id[i] - int(best_scores_id[i] / size) * size]
                    word_class_prior = self.syntax_topics_model.get_class_prior_for_word(word)
                    best_hyp_class_prior_sum.append(self.next_ys_class_prior_sum[-1][int(best_scores_id[i] / size)] + word_class_prior)
                self.next_ys_class_prior_sum.append(best_hyp_class_prior_sum)
            self.prev_ks.append(prev_k)
            self.next_ys.append((best_scores_id - prev_k * num_words))
            self.attn.append(attn_out.index_select(0, prev_k))
            # exit()

            self.print_all_words_in_beam()
            print "############## After new words chosen ###########"
        else:
            # beam_scores is only V dimensional vector
            # Thus for every word add the KL divergence score to the beam score
            if self.topic_KL_flag:
                topic_KL_divergence_scores = self.tt.zeros_like(beam_scores)
            if self.class_KL_flag:
                class_KL_divergence_scores = self.tt.zeros_like(beam_scores)
            for i in range(num_words):
                word = self.vocab.itos[i]
                if "<unk>" in word:
                    if self.topic_KL_flag:
                        topic_KL_divergence_scores[i] = 1000000.0
                    if self.class_KL_flag:
                        class_KL_divergence_scores[i] = 1000000.0
                    beam_scores[i] = -1000000.0
                    continue
                # KL for Topic priors of the first word
                if self.topic_KL_flag:
                    word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
                    word_topic_prior += self.beta
                    # topic_KL_divergence_scores[i] = entropy(self.source_topic_prior, word_topic_prior)
                    topic_KL_divergence_scores[i] = self.syntax_topics_model.KL_divergence(self.source_topic_prior, word_topic_prior)
                # KL for Class priors of the first word
                if self.class_KL_flag:
                    word_class_prior = self.syntax_topics_model.get_class_prior_for_word(word)
                    word_class_prior += self.beta
                    # class_KL_divergence_scores[i] = entropy(self.source_class_prior, word_class_prior)
                    class_KL_divergence_scores[i] = self.syntax_topics_model.KL_divergence(self.source_class_prior, word_class_prior)
                    
            overall_scores = beam_scores - ((self.alpha * topic_KL_divergence_scores) if self.topic_KL_flag else 0.0) -  ((self.gamma * class_KL_divergence_scores) if self.class_KL_flag else 0.0)
            # flat_beam_scores = overall_scores.view(-1)
            flat_beam_scores = beam_scores.view(-1)             # For the first iteration use only the word probabilities
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)

            self.all_scores.append(self.scores)
            # self.scores = best_scores           # will store the word_prob + the KL divergence for hypothesis
            self.scores = best_scores           # will store the word_prob log prob for hypothesis

            # best_scores_id is flattened, beam * word array, so calculate which
            # word and beam each score came from
            size = int(overall_scores.size(0))
            prev_k = best_scores_id / num_words
            # Update next_ys_topic_prior_sum for all beams
            if self.topic_KL_flag:
                best_hyp_topic_prior_sum = list()
                for i in range(self.size):
                    word = self.vocab.itos[best_scores_id[i]]
                    word_topic_prior = self.syntax_topics_model.get_topic_prior_for_word(word)
                    if word_topic_prior[self.syntax_topics_model.num_topics] != 1.0:
                        best_hyp_topic_prior_sum.append(word_topic_prior)
                    else:
                        # starting word is a syntax word. Therefore we will set the prior to zeros
                        best_hyp_topic_prior_sum.append(np.zeros((self.syntax_topics_model.num_topics+1), dtype=np.float))
                self.next_ys_topic_prior_sum.append(best_hyp_topic_prior_sum)

            # Update next_ys_class_prior_sum for all beams
            if self.class_KL_flag:
                best_hyp_class_prior_sum = list()
                for i in range(self.size):
                    word = self.vocab.itos[best_scores_id[i]]
                    word_class_prior = self.syntax_topics_model.get_class_prior_for_word(word)
                    best_hyp_class_prior_sum.append(word_class_prior)
                self.next_ys_class_prior_sum.append(best_hyp_class_prior_sum)

            self.prev_ks.append(prev_k)
            self.next_ys.append((best_scores_id - prev_k * num_words))
            self.attn.append(attn_out.index_select(0, prev_k))

        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                s = self.scores[i]
                timestep = len(self.next_ys)
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
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




































