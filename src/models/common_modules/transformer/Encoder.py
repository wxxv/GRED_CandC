#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/07/29
import torch
import torch.nn as nn

from src.models.common_modules.transformer import Attention, Modules


class TransformerEncoderLayer(nn.Module):
    '''
        one layer for transformer
    '''
    def __init__(self, d_model, n_heads, dim_feedforward = 2048, dropout = 0.0):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = Attention.MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = Modules.PositionalWiseFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, inputs, attn_mask = None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class TransformerEncoder(nn.Module):
    def __init__(self, max_seq_len, 
               num_layers = 6,
               d_model = 512,
               n_heads = 8,
               dim_feedforward = 2048,
               dropout = 0.0):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in
           range(num_layers)])

        self.pos_embedding = Modules.PositionalEncoding(d_model, max_seq_len)

    def forward(self, inputs, inputs_len = None, pos = False, attn_mask = None):
        '''
            inputs: [batch, len, dim]
            inputs_len: list of true length
        '''
        if pos:
            output = inputs + self.pos_embedding(inputs_len)
        else:
            output = inputs
        

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, attn_mask)
            attentions.append(attention)
        return output, attentions


