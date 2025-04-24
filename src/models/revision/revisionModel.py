#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/11/3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import cfg, args
from src.utils import dataset_semQL
from src.models.common_modules import textEncoder, visEncoder
#from src.models.common_modules.transformer import Modules, Encoder
from src.models.transformer.models import Encoder, get_attn_pad_mask
from src.models.revision import rnnDecoder, SQLDecoder, transDecoder
from src.models.retrieval.retrievalModel import RetrievalNet

class RevisionNet(RetrievalNet):
    def __init__(self, vocab, grammar = None, useVis = True):
        super(RevisionNet, self).__init__(vocab)
        self.vocab = vocab
        self.return_embed = args.no_LSTM
        self.embed = nn.Sequential(nn.Embedding(self.vocab.size, cfg.embed_vocab, padding_idx = 0),
                                   nn.Linear(cfg.embed_vocab, cfg.d_model, bias = True))
        
        self.word_encoder = textEncoder.LSTMEncoder(cfg.Net.wordencoder_in, cfg.Net.wordencoder_hidden, cfg.Net.wordencoder_layers,
                                                   cfg.Net.wordencoder_bi, embed = self.embed, return_embed = self.return_embed)

        self.ifGCN = cfg.use_gcn
        self.vis_encoder = visEncoder.ASTEncoder(cfg.ASTEncoder.gcn_layers, cfg.ASTEncoder.gcn_in, cfg.ASTEncoder.gcn_out, cfg.max_seq_len,
                                                 cfg.ASTEncoder.transformer_layers, cfg.ASTEncoder.d_model, cfg.ASTEncoder.n_heads,
                                                 cfg.ASTEncoder.dim_feedforward, cfg.dropout, self.ifGCN)
        self.text_encoder = textEncoder.NLEncoder(cfg.Net.wordencoder_hidden, cfg.max_seq_len, cfg.NLEncoder.transformer_layers, cfg.NLEncoder.d_model,
                                                 cfg.NLEncoder.n_heads, cfg.NLEncoder.dim_feedforward, cfg.dropout)

        #self.encoder = Encoder.TransformerEncoder(cfg.max_seq_len, cfg.Encoder.transformer_layers, cfg.Encoder.d_model, cfg.Encoder.n_heads, 
        #                                          cfg.Encoder.dim_feedforward, cfg.dropout)
        self.encoder = Encoder(vocab, cfg.Encoder.transformer_layers, cfg.Encoder.d_model, cfg.Encoder.d_model, cfg.Encoder.d_model, cfg.Encoder.dim_feedforward, 
                              cfg.Encoder.n_heads, cfg.max_seq_len, dropout = cfg.dropout)
        
        self.decoder_type = cfg.Net.decoder_type
        if self.decoder_type == 'rnn':
            self.decoder = rnnDecoder.RNNDecoder(vocab, cfg.RNNDecoder.hidden_size, cfg.RNNDecoder.layers, cfg.dropout)
        elif self.decoder_type == 'transformer':
            self.decoder = transDecoder.TransDecoder(vocab, cfg.max_seq_len, cfg.TRANSDecoder.transformer_layers, cfg.TRANSDecoder.d_model,
                                                    cfg.TRANSDecoder.n_heads, cfg.TRANSDecoder.dim_feedforward, cfg.dropout, copy = False)
        else:
            self.decoder = SQLDecoder.SQLDecoder(grammar, self.vocab, self.word_encoder)

        self.saliency_linear = nn.Linear(cfg.ASTEncoder.d_model, cfg.NLEncoder.d_model, bias = False)

        self.bias = nn.Parameter(torch.Tensor(1, 1))

        self.vis_pooling = cfg.Net.pooling
        self.useVis = useVis


    def get_saliency(self, text_hidden, vis_hidden):
        text_hidden = text_hidden.unsqueeze(1).transpose(1, 2)
        vis_hidden = vis_hidden.unsqueeze(1)
        vis_hidden = self.saliency_linear(vis_hidden)
        saliency = torch.bmm(vis_hidden, text_hidden) + self.bias
        saliency = saliency.squeeze(1)
        saliency = torch.sigmoid(saliency)
        return saliency


    def get_decoder_inputs(self, batch):
        
        # encode
        vis_embedding, vis_hidden = self.get_embedding_vis(batch.node_seqs_embedding_pattern, batch.adjs_pattern, batch.node_seqs_len_pattern)
        text_embedding, text_hidden = self.get_embedding_text(batch)

        encoder_inputs = torch.cat([text_embedding, vis_embedding], axis = 1)

        text_len = text_embedding.size()[1]
        if (not self.useVis) and (self.decoder_type == 'rnn'):
            encoder_outputs = text_embedding
        elif self.useVis:
            encoder_lens = text_len + np.array(batch.node_seqs_len_pattern)
            attn_mask = batch.padding_mask()
            encoder_outputs, _ = self.encoder(encoder_inputs, encoder_lens, pos = True, attn_mask = attn_mask)
        else:
            encoder_lens = np.array(batch.text_seqs_len)
            attn_mask = get_attn_pad_mask(batch.text_seqs_input, batch.text_seqs_input)
            #attn_mask = Modules.padding_mask(batch.text_seqs_embedding, batch.text_seqs_embedding)
            encoder_outputs, _ = self.encoder(text_embedding, encoder_lens, pos = True, attn_mask = attn_mask)
        encoder_hidden = F.adaptive_avg_pool1d(encoder_outputs.transpose(1, 2), 1)
        encoder_hidden = encoder_hidden.squeeze(-1)
        # loss1
        saliency = self.get_saliency(text_hidden, vis_hidden) 
        text_embedding = encoder_outputs[:, :text_len]

        return encoder_outputs, encoder_hidden, saliency, text_embedding

    
    def forward(self, examples):
        # preprocess / word encoder
        batch = dataset_semQL.Batch(examples, self.vocab, self.word_encoder, self.return_embed, revision = True)
        # encoder
        encoder_outputs, encoder_hidden, saliency, text_embedding = self.get_decoder_inputs(batch)
        rouges = batch.rouges

        # decoder
        targets = batch.targets
        decoder_inputs = targets[:, :-1]
        tgts = targets[:, 1:]
        if self.decoder_type == 'rnn':
            outputs, _, _ = self.decoder(decoder_inputs, encoder_outputs, encoder_hidden)
            if batch.targets_mask is not None:
                outputs = outputs.masked_fill(batch.targets_mask, -1e9)
            outputs = F.log_softmax(outputs, dim = -1)
            preds = torch.argmax(outputs, axis = -1)
            return outputs, preds, tgts, saliency, rouges
        elif self.decoder_type == 'transformer':
            decoder_inputs_len = batch.sql_seqs_len
            encoder_inputs = batch.text_seqs_input
            encoder_inputs_len = batch.text_seqs_len
            outputs = self.decoder(decoder_inputs, decoder_inputs_len, text_embedding, encoder_inputs)
            #outputs = self.decoder(decoder_inputs, decoder_inputs_len, encoder_outputs, encoder_inputs)
            if batch.targets_mask is not None:
                outputs = outputs.masked_fill(batch.targets_mask, -1e9)
            outputs = F.log_softmax(outputs, dim = -1)
            preds = torch.argmax(outputs, axis = -1)
            return outputs, preds, tgts, saliency, rouges
        else:
            decoder_outputs = self.decoder(batch, encoder_outputs, text_embedding)
            return decoder_outputs, saliency, rouges
        

    def parse(self, examples, ifTest = False, beam_size = None):
        batch = dataset_semQL.Batch(examples, self.vocab, self.word_encoder, self.return_embed, revision = True)
        encoder_outputs, encoder_hidden, _, text_embedding = self.get_decoder_inputs(batch)

        if self.decoder_type  == 'rnn':
            preds, tgts = self.decoder.parse(batch, encoder_outputs, encoder_hidden, ifTest = ifTest, beam_size = beam_size)
            return preds, tgts
        elif self.decoder_type == 'transformer':
            preds, tgts = self.decoder.parse(batch, text_embedding, encoder_hidden, ifTest = ifTest, beam_size = beam_size)
            return preds, tgts
        else:
            decoder_outputs = self.decoder.parse(batch, encoder_outputs, text_embedding, beam_size)
            return decoder_outputs
