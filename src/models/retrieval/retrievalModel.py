#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/11/2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import cfg, args
from src.utils import dataset_semQL as dataset
from src.models.common_modules import textEncoder, visEncoder
from src.models.common_modules.transformer import Modules

class RetrievalNet(nn.Module):
    def __init__(self, vocab):
        super(RetrievalNet, self).__init__()
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

        self.vis_pooling = cfg.Net.pooling
        self.common_projection = nn.Linear(cfg.d_model, cfg.d_model)
        if self.vis_pooling == 'linear':
            self.pooling_linear = nn.Linear(cfg.d_model, 1)
        #self.__init_weights__()


    def __init_weights__(self):
        if self.vocab.embedding is None:
            self.embed[0].weight.data.uniform_(-0.1, 0.1)
        else:
            self.embed[0].weight.data.copy_(torch.from_numpy(self.vocab.embedding))
            self.embed[0].weight.requires_grad = True


    def get_embedding_vis(self, node_seqs_embedding, adjs, lens):
        ast_padding_mask = Modules.padding_mask(node_seqs_embedding, node_seqs_embedding)
        vis_embeddings = self.vis_encoder(node_seqs_embedding, adjs, lens, ast_padding_mask)
        if self.vis_pooling == 'max':
            vis_hidden = F.adaptive_max_pool1d(vis_embeddings.transpose(1, 2), 1)
        elif self.vis_pooling == 'avg':
            vis_hidden = F.adaptive_avg_pool1d(vis_embeddings.transpose(1, 2), 1)
        else:
            agg_weights = self.pooling_linear(vis_embeddings)
            agg_weights = F.softmax(agg_weights, dim = 1)
            vis_hidden = torch.bmm(vis_embeddings.transpose(1, 2), agg_weights)

        vis_hidden = vis_hidden.squeeze(-1)
        vis_hidden = self.common_projection(vis_hidden)
        
        return vis_embeddings, vis_hidden

    def get_embedding_text(self, batch):
        text_padding_mask = Modules.padding_mask(batch.text_seqs_embedding, batch.text_seqs_embedding)
        text_embeddings = self.text_encoder(batch, text_padding_mask)
        text_hidden = F.adaptive_avg_pool1d(text_embeddings.transpose(1, 2), 1)
        text_hidden = text_hidden.squeeze(-1)
        text_hidden = self.common_projection(text_hidden)
        return text_embeddings, text_hidden

    def forward(self, examples):
        # preprocess / word encoder
        batch = dataset.Batch(examples, self.vocab, self.word_encoder, self.return_embed)
        # encode
        _, vis_embedding_pos = self.get_embedding_vis(batch.node_seqs_embeddings[0], batch.adjs[0], batch.node_seqs_lens[0])
        _, vis_embedding_neg = self.get_embedding_vis(batch.node_seqs_embeddings[1], batch.adjs[1], batch.node_seqs_lens[1])
        _, text_embedding = self.get_embedding_text(batch)

        # similarity
        sims_pos = F.cosine_similarity(text_embedding, vis_embedding_pos, dim = -1)
        sims_neg = F.cosine_similarity(text_embedding, vis_embedding_neg, dim = -1)
        
        # loss
        loss = 1-sims_pos+sims_neg
        mask = loss >= 0
        loss = torch.mean(loss*mask)
        return loss

    def parse(self, examples):
        batch = dataset.Batch(examples, self.vocab, self.word_encoder, self.return_embed, test = True)
        # encode
        _, vis_embedding_pos = self.get_embedding_vis(batch.node_seqs_embeddings[0], batch.adjs[0], batch.node_seqs_lens[0])
        _, vis_embedding_neg = self.get_embedding_vis(batch.node_seqs_embeddings[1], batch.adjs[1], batch.node_seqs_lens[1])
        _, text_embedding = self.get_embedding_text(batch)

        # similarity
        sims_pos = F.cosine_similarity(text_embedding, vis_embedding_pos, dim = -1)
        sims_neg = F.cosine_similarity(text_embedding, vis_embedding_neg, dim = -1)

        sims = torch.stack([sims_pos, sims_neg], axis = -1)
        #print(sims[:10])
        preds = torch.argmax(sims, axis = -1)
        labels = batch.tags
        return preds, labels
