#!/usr/bin/env python
# coding=utf-8
#@summerzhao:2021/07/29

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from src.models.revision.copyNet import CopyNet
from src.config import cfg
from src.models.transformer.beam import Beam
#from src.models.common_modules.transformer import Decoder
from src.models.transformer.models import Decoder

class TransDecoder(nn.Module):
    def __init__(self, vocab, max_seq_len, num_layers, d_model, n_heads, dim_feedforward, dropout = 0.0, embed = None, copy = False):
        super(TransDecoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size_out
        self.copy = copy
        self.use_cuda = cfg.cuda
        self.d_model = d_model
        self.tt = torch.cuda if self.use_cuda else torch
        self.seq_embedding = nn.Embedding(vocab.size, d_model)
        
        self.decoder = Decoder(vocab, num_layers, d_model, d_model, d_model, dim_feedforward, n_heads, max_seq_len, self.vocab_size, dropout)
        self.tgt_project = nn.Linear(d_model, self.vocab_size, bias = False)

        
    def forward(self, decoder_inputs, decoder_inputs_len, encoder_outputs, encoder_inputs, src_map = None):
        decoder_outputs, _, _ = self.decoder(decoder_inputs, decoder_inputs_len, encoder_outputs, encoder_inputs)
        decoder_outputs = self.tgt_project(decoder_outputs)
        return decoder_outputs


    def parse(self, batch, encoder_outputs, encoder_hidden = None, ifTest = False, beam_size = 3):
        #if ifTest:
        if False:
            with torch.no_grad():
                preds, tgts = self.beamParse(batch, encoder_outputs)
            return preds, tgts
        tgts = batch.targets
        tgts_len = tgts.size()[1]
        encoder_inputs = batch.text_seqs_input
        
        batch_size = len(encoder_outputs)
        output = torch.zeros(batch_size, tgts_len).long().cuda()
        output[:, 0] = torch.ones(batch_size).long().cuda()*self.vocab.go
        for i in range(1, tgts_len):
            preds = output[:, :i]
            preds_len = [i] * batch_size
            
            decoder_outputs = self.forward(preds, preds_len, encoder_outputs, encoder_inputs)
            if batch.targets_mask is not None:
                decoder_outputs = decoder_outputs.masked_fill(batch.targets_mask, -1e9)
            preds_new = torch.argmax(decoder_outputs, axis = -1)
            output[:, i] = preds_new[:, -1]
            
        return output[:, 1:], tgts[:, 1:]

    def beamParse(self, batch, enc_outputs, beam_size = 5):
        tgts = batch.targets
        tgts_len = tgts.size()[1]
        enc_inputs = batch.text_seqs_input
        
        batch_size = len(enc_outputs)
        # Repeat data for beam
        enc_inputs = enc_inputs.repeat(1, beam_size).view(batch_size * beam_size, -1)
        
        enc_outputs =  enc_outputs.repeat(1, beam_size, 1).view(
            batch_size * beam_size, enc_outputs.size(1), enc_outputs.size(2))

        # Prepare beams
        beams = [Beam(beam_size, self.vocab, self.use_cuda) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))
        }
        n_remaining_sents = batch_size

        # Decode
        for i in range(1, tgts_len + 1):
            len_dec_seq = i
            # Preparing decoded data_seq
            # size: [batch_size x beam_size x seq_len]
            dec_partial_inputs = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            # size: [batch_size * beam_size x seq_len]
            dec_partial_inputs = dec_partial_inputs.view(-1, len_dec_seq)
            # wrap into a Variable
            dec_partial_inputs = Variable(dec_partial_inputs)

            # # wrap into a Variable
            # dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True)
            dec_partial_inputs_len = torch.LongTensor(n_remaining_sents,).fill_(len_dec_seq) # TODO: note
            dec_partial_inputs_len = dec_partial_inputs_len.repeat(beam_size)
            #dec_partial_inputs_len = Variable(dec_partial_inputs_len, volatile=True)

            if self.use_cuda:
                dec_partial_inputs = dec_partial_inputs.cuda()
                dec_partial_inputs_len = dec_partial_inputs_len.cuda()

            # Decoding
            dec_outputs  = self.forward(dec_partial_inputs, dec_partial_inputs_len,
                                                 enc_outputs, enc_inputs) # TODO:
            if batch.targets_mask is not None:
                targets_mask = torch.stack([batch.targets_mask[i] for i, b in enumerate(beams) if not b.done])
                targets_mask = targets_mask.repeat(1, beam_size, 1).view(beam_size * n_remaining_sents, 1, -1)
                dec_outputs = dec_outputs.masked_fill(targets_mask, -1e9)
            dec_outputs = dec_outputs[:,-1,:] # [batch_size * beam_size x d_model]
            
            out = F.log_softmax(dec_outputs, dim = -1)

            # [batch_size x beam_size x tgt_vocab_size]
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx] # 해당 beam_idx 의 데이터가 실제 data 에서 몇번째 idx인지
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list: # all instances have finished their path to <eos>
                break

            # In this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list]) # TODO: fix

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                #return Variable(active_seq_data, volatile=True)
                return Variable(active_seq_data)


            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                #return Variable(active_enc_info_data, volatile=True)
                return Variable(active_enc_info_data)


            enc_inputs = update_active_seq(enc_inputs, active_inst_idxs)
            enc_outputs = update_active_enc_info(enc_outputs, active_inst_idxs)

            # update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # Return useful information
        all_hyp, all_scores = [], []
        n_best = 2

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        preds = [hyp[0] for hyp in all_hyp]
        return preds, tgts[:, 1:]


