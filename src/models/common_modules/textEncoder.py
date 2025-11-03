#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/11/2
import torch
import torch.nn as nn

from src.models.common_modules.transformer import Encoder
from src.models.common_modules.globalAttention import GlobalAttention

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bidirectional = False, dropout = 0.0, embed = None, return_embed = False):
        super(LSTMEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional = bidirectional, dropout = dropout, batch_first = True)

        self.embed = embed
        self.return_embed = return_embed


    def forward(self, inputs, inputs_len, pos = None):
        # pos is not useful
        if self.embed is not None:
            inputs = self.embed(inputs)
            if self.return_embed:
                return inputs, inputs
        
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, inputs_len, batch_first = True, enforce_sorted = False)
        outputs, states = self.gru(inputs)
        outputs, lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True)
        if self.bidirectional is True:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            states = states[-1] + states[-2]
        else:
            states = states[-1]
        return outputs, states


class NLEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len, transformer_layers, d_model, n_heads, dim_feedforward, dropout = 0.0):
        super(NLEncoder, self).__init__()
        self.attention = GlobalAttention(hidden_size)
        #self.encoder = Encoder.TransformerEncoder(max_seq_len, transformer_layers, d_model, n_heads, dim_feedforward, dropout)
        self.encoder = LSTMEncoder(hidden_size, hidden_size, num_layers = 1, bidirectional = True)#, dropout = dropout)

    def forward(self, batch, attn_mask = None):
        text_seqs = batch.text_seqs_embedding
        text_lens = batch.text_seqs_len
        text_inputs = text_seqs
        
        schema_seqs = batch.schema_seqs_embedding
        
        scores = self.attention(text_seqs, schema_seqs)
        text_inputs = text_seqs + torch.matmul(scores, schema_seqs)
        #text_outputs, _ = self.encoder(text_inputs, text_lens, pos = True, attn_mask = attn_mask)
        text_outputs, _ = self.encoder(text_inputs, text_lens)
        return text_outputs
