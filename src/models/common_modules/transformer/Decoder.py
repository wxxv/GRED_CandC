#!/usr/bin/env python
# coding=utf-8
#@summerzhao: 2021/7/29

import torch
import torch.nn as nn
import numpy as np

from src.models.common_modules.transformer import Modules, Attention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads = 8, dim_feedforward = 2048, dropout = 0.0):
        super(TransformerDecoderLayer, self).__init__()

        self.attention1 = Attention.MultiHeadAttention(d_model, n_heads, dropout)
        self.attention2 = Attention.MultiHeadAttention(d_model, n_heads, dropout)

        self.feed_forward = Modules.PositionalWiseFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, decoder_inputs, encoder_outputs, self_attn_mask = None, context_attn_mask = None):
        
        #print('-------------1')
        decoder_output1, self_attention = self.attention1(decoder_inputs, decoder_inputs, decoder_inputs, self_attn_mask)
        #print('-------------2')
        decoder_output2, context_attention = self.attention2(encoder_outputs, encoder_outputs, decoder_output1, context_attn_mask)

        decoder_output = self.feed_forward(decoder_output2)
        #if decoder_inputs.size()[1] >= 3:
        #    print('--decoder_output1: ', decoder_output1[0, :3, :5])
        #    print('--decoder_output2: ', decoder_output2[0, :3, :5])

        return decoder_output, self_attention, context_attention


class TransformerDecoder(nn.Module):
    def __init__(self, vocab, max_seq_len, num_layers = 6, d_model = 512, 
                 n_heads = 8, dim_feedforward = 2048, dropout = 0.0, embed = None):

        super(TransformerDecoder, self).__init__()
        self.vocab = vocab
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        
        #self.seq_embedding = nn.Embedding(self.vocab.size, d_model, padding_idx = 0)
        self.seq_embedding = embed
        
        self.pos_embedding = Modules.PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, inputs_len, encoder_output, encoder_inputs):
        '''
            context_attn_mask for encoder_output and decoder_inputs
        '''
        output = self.seq_embedding(inputs)
        max_len = inputs.size()[1]
        #print('output', output.size(), self.pos_embedding(inputs_len, max_len).size())
        output += self.pos_embedding(inputs_len, max_len)
        output = self.dropout(output)
        self_attention_padding_mask = Modules.padding_mask(inputs, inputs)
        
        
        seq_mask = Modules.sequence_mask(inputs)
        seq_mask = seq_mask.cuda()
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)
        context_attn_mask = Modules.padding_mask(encoder_inputs, inputs)
        
        
        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, encoder_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        #if inputs.size(1) >=1:
        #    print('self_attn_mask: ', self_attn_mask[0, :5], self_attn_mask.size())
        #    print('self attention: ', self_attn[0, :5, :5])
        #    print('context_attn_mask: ', context_attn_mask[0][0], context_attn_mask.size())
        #    print('context attn: ', context_attn[0, :5, :5])
        #    print('output: ', output[0, :5, :5])
        return output, self_attentions, context_attentions
