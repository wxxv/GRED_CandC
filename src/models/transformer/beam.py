#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np


class Beam(object):
    ''' Store the necessary info for beam search '''
    def __init__(self, size, vocab, use_cuda=False):
        self.size = size
        self.done = False
        self.vocab = vocab

        self.tt = torch.cuda if use_cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(vocab.pad)]
        self.next_ys[0][0] = vocab.go

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort TODO
        
        
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam_size * tgt_vocab_size array, so calculate
        # which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        #print('best scores', best_scores, 'best_scores_id', best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == self.vocab.eos:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.vocab.go] + h for h in hyps]
            
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        for j in range(len(self.prev_ks)-1, -1, -1):
            hyp.append(self.next_ys[j + 1][k].cpu().long())
            k = self.prev_ks[j][k]

        return hyp[::-1]

