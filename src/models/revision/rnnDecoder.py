#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/8/10
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.common_modules import globalAttention

class RNNDecoder(nn.Module):
    def __init__(self, vocab, hidden_size, n_layers = 1, dropout = 0.0, useAttention = True):
        super(RNNDecoder, self).__init__()
        self.vocab = vocab 
        self.vocab_size = self.vocab.size_out

        self.embed = nn.Embedding(self.vocab_size, hidden_size)

        self.useAttention = useAttention

        self.dropout = nn.Dropout(dropout)
        self.attention = globalAttention.GlobalAttention(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout = dropout, batch_first = True)
        self.tgt_linear = nn.Linear(hidden_size * 2, self.vocab_size)

    def forward(self, decoder_inputs, encoder_outputs = None, decoder_hidden = None, padding_mask = None):
        # Get the embedding of the current input word (last output word)
        decoder_inputs = self.embed(decoder_inputs)
        decoder_inputs = self.dropout(decoder_inputs)
  
        rnn_outputs, rnn_hiddens = self.gru(decoder_inputs, decoder_hidden.unsqueeze(0))
        hidden = rnn_hiddens[-1]

        if self.useAttention:
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attention(rnn_outputs, encoder_outputs, padding_mask)
            context = attn_weights.bmm(encoder_outputs)
            
            output = self.tgt_linear(torch.cat([rnn_outputs, context], -1))
            #output = F.log_softmax(output, dim=-1)
        else:
            output = rnn_outputs
            attn_weights = None
        return output, hidden, attn_weights

    def parse(self, batch, encoder_outputs, encoder_hidden, ifTest = False, beam_size = 3):
        targets = batch.targets
        targets_mask = batch.targets_mask
        tgts = targets[:, 1:]

        tgts_len = targets.size()[1]
        batch_size = targets.size()[0]
        preds_start = torch.tensor(np.array([self.vocab.go] * batch_size, dtype = np.int64)).view(-1, 1)
        preds = preds_start.cuda()
        probs = []
        for i in range(tgts_len - 1):
            outputs, encoder_hidden, _ = self.forward(preds, encoder_outputs, encoder_hidden)
            outputs = outputs[:, -1:]
            if targets_mask is not None:
                outputs = outputs.masked_fill(targets_mask, -1e9)
                #print(outputs)
            preds = torch.argmax(outputs, axis = -1)
            probs.append(outputs)
        probs = torch.stack(probs, axis = 1).squeeze(-2)
        preds = torch.argmax(probs, axis = -1).squeeze(-1)
        return preds, tgts
