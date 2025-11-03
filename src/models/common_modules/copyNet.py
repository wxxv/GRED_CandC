#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/08/09
import torch
import torch.nn as nn
import torch.nn.functional as F

#from src.models.transformer import Attention
from src.modules.globalAttention import GlobalAttention

class CopyNet(nn.Module):
    def __init__(self, d_model, linear_tgt):
        super(CopyNet, self).__init__()
        self.linear_tgt = linear_tgt
        self.linear_copy = nn.Linear(d_model, 1, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        #self.copy_attention = Attention.MultiHeadAttention(d_model, num_heads = 1)
        self.copy_attention = GlobalAttention(d_model)


    def forward(self, decoder_outputs, encoder_outputs, src_map, padding_mask = None):
        
        #_, copy_scores = self.copy_attention(encoder_outputs, encoder_outputs, decoder_outputs, padding_mask)
        copy_scores = self.copy_attention(decoder_outputs, encoder_outputs, padding_mask)
        #print('copy_scores', copy_scores)

        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(decoder_outputs))

        # Original probabilities
        logits = self.linear_tgt(decoder_outputs)
        preds_probs = self.softmax(logits)
        vocab_probs = torch.mul(preds_probs, p_copy.expand_as(preds_probs))


        # copy probs
        copy_scores = torch.mul(copy_scores, 1-p_copy.expand_as(copy_scores))
        copy_probs = torch.bmm(copy_scores, src_map)  
        probs = vocab_probs + copy_probs
        probs = probs.log()
        return probs

        

