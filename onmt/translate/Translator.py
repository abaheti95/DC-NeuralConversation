import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.translate.SIFEmbedding
import onmt.io

import nltk

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        # self.min_length = min_length
        #TODO: Experimental
        self.min_length = 3

        tgt_vocab = self.fields["tgt"].vocab
        import os
        embeddings_file = os.path.join("data", "sif_data", "paragram_sl999_small.txt")
        weights_file = os.path.join("data", "sif_data", "enwiki_vocab_min200.txt")
        pc_file = os.path.join("data", "sif_data", "opensubtitles_train_data_first_component.npy")
        self.similarity_scorer = onmt.translate.SIFEmbedding(tgt_vocab, embeddings_file, weights_file, pc_file, pad_word_id=tgt_vocab.stoi[onmt.io.PAD_WORD], a=1e-3)
        # Initializing the Topics Syntax model
        documents_file = os.path.join("data", "syntax_topics_models", "mallet_topics_model", "t_given_s_dialogue_length3_6_words_50_encoded.txt")
        vocabulary_file = os.path.join("data", "syntax_topics_models", "mallet_topics_model", "t_given_s_dialogue_length3_6_words_50_vocab.txt")
        topic_assignment_file = os.path.join("data", "syntax_topics_models", "mallet_topics_model", "2500_results", "topic_assignments.txt")
        class_assignment_file = os.path.join("data", "syntax_topics_models", "mallet_topics_model", "2500_results", "class_assignments.txt")
        stop_words_file = os.path.join("data", "syntax_topics_models", "stop_words_long.txt")
        # stop_words_file = os.path.join("data", "syntax_topics_models", "stop_words_short.txt")

        self.syntax_topics_model = onmt.translate.SyntaxTopicModel(vocabulary_file, documents_file, class_assignment_file, topic_assignment_file, stop_words_file, max_documents = None, recount = False)

        # self.M = 10             # Number of diverse n-best lists

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def print_targets(self, targets_batch, batch_size):
        vocab = self.fields["tgt"].vocab
        for i in range(batch_size):
            targets = targets_batch[i]
            # Print the targets of M iterations
            print len(targets)
            for j in range(len(targets)):
                # each element is a n-best list of indices
                for k in range(len(targets[j])):
                    word_ids = targets[j][k]
                    for word_id in word_ids:
                        print "{} ".format(vocab.itos[word_id]) ,
                    print ""
                print "$$$$$$$$$$$$$$$$$$$$$$$$"
            print "######### next batch ##########"

    def get_oracle_bleu_targets(self, batch_targets, gold_targets):
        chencherry = nltk.translate.bleu_score.SmoothingFunction()
        batch_best_targets = list()
        batch_best_bleu = list()
        for i in range(len(gold_targets)):
            # Iterating on the batch level
            # There is 1 gold target for each batch
            gold_target = gold_targets[i]
            # There are M diverse * N best targets out of which we have to find the best
            current_targets = batch_targets[i]
            best_target = None
            max_bleu = -1000000.0
            for j in range(len(current_targets)):
                for k in range(len(current_targets[j])):
                    current_target = current_targets[j][k]
                    # Find the bleu score of current_target with the gold and update the best
                    bleu = nltk.translate.bleu_score.sentence_bleu([gold_target], current_target, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=chencherry.method4)
                    if max_bleu < bleu:
                        max_bleu = bleu
                        best_target = current_target
            batch_best_targets.append(best_target)
            batch_best_bleu.append(max_bleu)
        return (batch_best_targets, batch_best_bleu)

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        if "tgt" in batch.__dict__:
            tgt_tensor = batch.tgt.data.t()
            gold_targets = tgt_tensor.tolist()

        ##NOTE: We have to run the subroutine below M times i.e. run the beam search M times 
        #       where each time we will select the previous best output and add it to the list of targets
        # We will have list of targets for each beam in the batch
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    # global_scorer=None,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    vocab=vocab,
                                    similarity_scorer=self.similarity_scorer,
                                    syntax_topics_model=self.syntax_topics_model,
                                    source=tuple(batch.src[0][:,i].data),
                                    min_length=self.min_length)
                for i in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, context, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(context.data)\
                                                  .long()\
                                                  .fill_(context.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        context = rvar(context.data)
        context_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, context, dec_states, context_lengths=context_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab)
                # beam x tgt_vocab
                out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :context_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.

        self.similarity_scorer.clear_cache()
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        # Update the list of targets based on the ret["predictions"]
            # dummy = [m_diverse_targets[0] for m_diverse_targets in targets[i]]
            # print dummy
            # print ret["predictions"][i]
        # print "Targets"
        # print targets
        # print targets[0][:][:][0]
        # print "Predictions"
        # print ret["predictions"]
        # self.print_targets(targets, batch_size)
        # exit()
        
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src,
                                                           context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states, context_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
