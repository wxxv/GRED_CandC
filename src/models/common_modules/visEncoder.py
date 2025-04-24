#!/usr/bin/env python
# coding=utf-8
#@summerzhao:2021/11/2

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import cfg
from src.models.common_modules.transformer import Encoder


def gen_adj(A):
    '''
        A = A + I_n
        adj = D^(-1/2)AD^(-1/2)
    '''
    #batch_size, node_size, _ = A.size()
    #A_eye = torch.eye(node_size)
    #A_eye = A_eye.reshape((1, node_size, node_size))
    #A_eye = A_eye.repeat(batch_size, 1, 1).cuda()
    #A = A + A_eye
    #A = A + torch.eye(A.size()[0])
    A_eye = torch.eye(A.size()[1])
    if cfg.cuda: A_eye = A_eye.cuda()
    A = A + A_eye
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag_embed(D, offset=0, dim1=-2, dim2=-1)
    adj = torch.matmul(torch.matmul(A, D).transpose(1,2), D)
    #D = torch.diag(D)
    #adj = torch.matmul(torch.matmul(A, D).t(), D)
    #adj = adj.cuda()
    return adj

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    input:
        inputs
        adj: Adjacent matrix
    """

    def __init__(self, in_features, out_features, dropout=0, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, inputs, adjs):
        inputs = self.dropout(inputs)
        outputs = torch.matmul(adjs, inputs)
        outputs = torch.matmul(outputs, self.weight)
        if self.bias:
            outputs = outputs + bias
        outputs = F.relu(outputs)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MultiLayerGCN(nn.Module):
    def __init__(self, num_layers, in_features, out_features, dropout = 0):
        super(MultiLayerGCN, self).__init__()
        self.gcns = nn.ModuleList(
          [GCN(in_features, out_features, dropout, bias = False) for _ in range(num_layers)])


    def forward(self, inputs, adjs):
        outputs = inputs
        adjs = gen_adj(adjs)

        for gcn in self.gcns:
            outputs = gcn(outputs, adjs)
        return outputs


class ASTEncoder(nn.Module):
    def __init__(self, gcn_layers, gcn_in, gcn_out, 
                max_seq_len, transformer_layers, d_model, n_heads, dim_feedforward, dropout = 0.0, ifGCN = True):
        super(ASTEncoder, self).__init__()
        
        #self.transformer_encoder = Encoder.TransformerEncoder(max_seq_len, transformer_layers, d_model, n_heads, dim_feedforward, dropout)
        self.ifGCN = ifGCN
        if self.ifGCN:
            self.gcn = MultiLayerGCN(gcn_layers, gcn_in, gcn_out, dropout)

    def forward(self, ast_nodes, ast_adjs, lens, attn_mask = None):
        if self.ifGCN:
            ast_embeddings = self.gcn(ast_nodes, ast_adjs)
        else:
            ast_embeddings = ast_nodes
        #ast_embeddings, _ = self.transformer_encoder(ast_embeddings, lens, pos = True, attn_mask = attn_mask)

        return ast_embeddings

if __name__ == '__main__':
    adj = np.ones([3, 4, 4])
    adj = torch.from_numpy(adj)
    adj_new = gen_adj(adj)
    print(adj_new.size())
