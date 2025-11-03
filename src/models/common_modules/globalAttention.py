#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/8/10
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, attn_type = 'Luong'):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        if self.attn_type == 'Luong':
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias = False)
        elif self.attn_type == 'Bahdanau':
            self.linear_context = nn.Linear(hidden_size, hidden_size, bias = False)
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias = True)
            self.v = nn.Linear(hidden_size, 1, bias = False)
        else:
            raise TypeError


        self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, hidden, encoder_outputs, padding_mask = None):
        '''
            hidden: batch * tgt_len * hidden_size
            encoder_outputs: batch * src_len * hidden_size
        '''
        
        scores = self.score(hidden, encoder_outputs)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
        scores = self.softmax(scores)
        return scores

    def score(self, hidden, encoder_outputs):
        '''
            attn_type: 'Luong' / 'Bahdanau'

            * 'Luong': 
                * dot: `score(H_j,q) = H_j^T q`
                * general: `score(H_j, q) = H_j^T W_a q`
            * Bahdanau Attention (mlp):
                * `score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
        '''
        if self.attn_type == 'Luong':
            hidden = self.linear_in(hidden)
            encoder_outputs = encoder_outputs.transpose(1, 2)
            scores = torch.bmm(hidden, encoder_outputs)
            
        if self.attn_type == 'Bahdanau':

            batch, tgt_len, hidden_size = hidden.size()
            src_len = encoder_outputs.size()[1]
            
            # query
            hidden = self.linear_query(hidden).unsqueeze(2)
            hidden = hidden.expand(tgt_batch, tgt_len, src_len, hidden_size)
            # context
            encoder_outputs = self.linear_context(encoder_outputs).unsqueeze(1)
            encoder_outputs = encoder_outputs.expand(batch, tgt_len, src_len, hidden_size)

            # (batch, t_len, s_len, d)
            scores = self.v(self.tanh(hidden + encoder_outputs))
            scores = outputs.squeeze(-1)

        return scores

