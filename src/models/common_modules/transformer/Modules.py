#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/13
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import cfg
from src.models.common_modules.transformer import Attention

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, d_model = 512, dim_feedforward = 2048, dropout = 0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(d_model, dim_feedforward, 1)
        self.w2 = nn.Conv1d(dim_feedforward, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len, dropout = 0.0):
        """
        Args:
            d_model: dimention of model
            max_seq_len: max length of text in the dataset
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)], dtype = np.float32)
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        
        # add row for padding 
        position_encoding = torch.from_numpy(position_encoding)
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        self.position_encoding = nn.Embedding(max_seq_len, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, inputs_len, max_len = None):
        """

        input:
          input_lens: a list of length for one batch

        Returns:
          positional_embedding with padding
        """
        inputs_len = torch.tensor(inputs_len).cuda()
        if max_len is None:
            max_len = torch.max(inputs_len)

        tensor = torch.cuda.LongTensor if inputs_len.is_cuda else torch.LongTensor
        input_pos = tensor(
          [list(range(1, input_len + 1)) + [0] * (max_len - input_len) for input_len in inputs_len])
        position_encoding = self.position_encoding(input_pos)
        position_encoding = self.dropout(position_encoding)
        return position_encoding


def padding_mask(seq_k, seq_q):
    # 'PAD' = 0
    if seq_k.dim() == 3:
        seq_k = seq_k[:, :, 0]
        #seq_q = seq_q[:, :, 0]
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask



