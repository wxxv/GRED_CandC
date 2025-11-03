# !/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/07/30
import os
import argparse
import os.path as osp
import numpy as np
from easydict import EasyDict as edict


parser = argparse.ArgumentParser(description='vis2text training')
parser.add_argument('--log_name', default='retrieval.log', type=str, help="training log file")
parser.add_argument('--model_save_path', default='vis2text', type=str, help="training model file")
parser.add_argument('--load_model', default=None, type=str, help="continuous training model file")
parser.add_argument('--GPU', default='5', type=str, help="GPU used")
parser.add_argument('--epoches', default=100, type=int, help="Epoch for training")
parser.add_argument('--batch_size', default=128, type=int, help="batch_size for training / dev")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate for training")
parser.add_argument('--test', action='store_true', help="test or train")
parser.add_argument('--generate', action='store_true', help="test or train")

parser.add_argument('--no_LSTM', action='store_true', help="test or train")
parser.add_argument('--copy', action='store_true', help="use copy mechanism")

parser.add_argument('--use_gcn', action='store_true', help="use gcn layer")
parser.add_argument('--gcn_layers', default=1, type=int, help="number or gcn layers")
parser.add_argument('--trans_layers', default=1, type=int, help="number or transformer layers")
parser.add_argument('--trans_heads', default=4, type=int, help="number or transformer heads")
parser.add_argument('--pooling', default='avg', type=str, help="pooling strategy for vis embedding")
parser.add_argument('--decoder_type', default='rnn', type=str, help="use which decoder , rnn based or schema_aware based")

parser.add_argument('--pretrain', action='store_true', help="if load pretrained weights")
parser.add_argument('--pretrained_model', default=None, type=str, help="pretrained model file")

parser.add_argument('--noleaf', action='store_true', help="if has leaf(c/t) for sql")
parser.add_argument('--usevis', action='store_true', help="if use vis for revision")
parser.add_argument('--data_path', default='./data/', help="path for save predict results")
parser.add_argument('--oracle', default = 0, type=int, help="if use oracle sql template")
parser.add_argument('--use_keywords', action='store_true', help="if use keywords retrieval for revision")
parser.add_argument('--top', default = 5, type=int, help="top k selected for revision training")


# args = parser.parse_args()
args = parser.parse_known_args()[0]

__C = edict()
cfg = __C

# pre-set parameters
cfg.load_model = args.load_model
cfg.num_workers = 4
cfg.cuda = True
cfg.use_gcn = args.use_gcn
cfg.pretrain = args.pretrain
cfg.pretrained_model = args.pretrained_model
cfg.noleaf = args.noleaf
cfg.useVis = args.usevis
cfg.oracle = args.oracle
cfg.data_path = args.data_path
cfg.top = args.top
cfg.use_keywords = args.use_keywords

# path
cfg.Path = edict()
data_path = cfg.data_path
if cfg.noleaf:
    cfg.Path.data_path = '%s/{}/{}_pattern_pair_noleaf1.json'%data_path
else:
    cfg.Path.data_path = '%s/{}/{}_pattern_pair.json'%data_path

if cfg.oracle:
    cfg.Path.data_revision_path = '%s/{}/{}_revision_oracle.json'%data_path
else:
    if cfg.use_keywords:
        cfg.Path.data_revision_path = '%s/{}/{}_revision_keywords.json'%data_path
    elif not cfg.use_gcn:
        cfg.Path.data_revision_path = '%s/{}/{}_revision_nogcn_notrans.json'%data_path
    else:
        cfg.Path.data_revision_path = '%s/{}/{}_revision_gcn%s_notrans.json'%(data_path,args.gcn_layers)

#cfg.Path.data_revision_path = '../RGVisNet.bak/data_new_pattern2/{}/{}_revision_nogcn_notrans.json'
#cfg.Path.data_revision_path = '../RGVisNet.bak/data_new_pattern2/{}/{}_revision_gcn%s_asr.json'%(args.gcn_layers)

cfg.Path.table_path = './data/tables.json'
# cfg.Path.vocab_path = './data/vocab_new.txt'
# cfg.Path.vocab_embedding_path = './data/vocab_embedding_new.txt'
cfg.Path.vocab_path = './data/vocab_new.txt'
cfg.Path.vocab_embedding_path = './data/vocab_embedding_new.txt'

cfg.Path.input_vocab_path = './data/vocab_new.txt'
cfg.Path.input_vocab_embedding_path = './data/vocab_embedding_input_merge.txt'
cfg.Path.output_vocab_path = './data/vocab_output_merge.txt'
cfg.Path.output_vocab_embedding_path = './data/vocab_embedding_output_merge.txt'

cfg.Path.glove_path = './GloVe/glove.840B.300d/glove.840B.300d.txt'
cfg.Path.checkpoint_path = './saved_model/{}/'.format(args.model_save_path)

cfg.Path.pred_path = './results/pred_%s_{}.json'%args.model_save_path
# Model Paramters
# shared parameters
cfg.max_seq_len = 500
cfg.embed_vocab = 300
cfg.embed_hidden = 512
cfg.d_model = 512
#cfg.transformer_layers = args.trans_layers
#cfg.transformer_head = args.trans_heads
cfg.transformer_layers = 1
cfg.transformer_head = 4
cfg.dim_feedforward = 300
cfg.dropout = 0.2
cfg.max_pred_len = 100

# optimizer
cfg.optimizer = 'Adam'
cfg.lr_scheduler = False
cfg.lr_scheduler_gammar = 0.5
cfg.clip_grad = 5
cfg.epoches = args.epoches
cfg.batch_size = args.batch_size
cfg.lr = args.lr
cfg.loss_epoch_threshold = 20
cfg.sketch_loss_coefficient = 0.2



# ast encoder
cfg.ASTEncoder = edict()
cfg.ASTEncoder.gcn_layers = args.gcn_layers
cfg.ASTEncoder.gcn_in = cfg.embed_hidden
cfg.ASTEncoder.gcn_out = 512
cfg.ASTEncoder.transformer_layers = cfg.transformer_layers
cfg.ASTEncoder.d_model = cfg.d_model
cfg.ASTEncoder.n_heads = cfg.transformer_head
cfg.ASTEncoder.dim_feedforward = cfg.dim_feedforward



# sbt encoder
cfg.NLEncoder = edict()
cfg.NLEncoder.transformer_layers = cfg.transformer_layers
cfg.NLEncoder.d_model = cfg.d_model
cfg.NLEncoder.n_heads = cfg.transformer_head
cfg.NLEncoder.dim_feedforward = cfg.dim_feedforward




# TextVisNet 
cfg.Net = edict()
cfg.Net.wordencoder_in = cfg.d_model
cfg.Net.wordencoder_hidden = cfg.embed_hidden
cfg.Net.wordencoder_layers = 1
cfg.Net.wordencoder_bi = False  # bidirectional
cfg.Net.pooling = args.pooling
cfg.Net.decoder_type = args.decoder_type


# transformer encoder
cfg.Encoder = edict()
cfg.Encoder.transformer_layers = args.trans_layers
cfg.Encoder.d_model = cfg.d_model
cfg.Encoder.n_heads = args.trans_heads
cfg.Encoder.dim_feedforward = cfg.dim_feedforward

# transformer decoder
cfg.TRANSDecoder = edict()
cfg.TRANSDecoder.transformer_layers = args.trans_layers
cfg.TRANSDecoder.d_model = cfg.d_model
cfg.TRANSDecoder.n_heads = args.trans_heads
cfg.TRANSDecoder.dim_feedforward = cfg.dim_feedforward



# RNNDecoder
cfg.RNNDecoder = edict()
cfg.RNNDecoder.hidden_size = 512
cfg.RNNDecoder.layers = 1



# SQLDecoder
cfg.SQLDecoder = edict()
cfg.SQLDecoder.beam_size = 3
cfg.SQLDecoder.column_pointer = True
cfg.SQLDecoder.action_embed_size = 128 # size of action embeddings
cfg.SQLDecoder.att_vec_size = 512 # size of attentional vector
cfg.SQLDecoder.type_embed_size = 128 # size of word embeddings
cfg.SQLDecoder.hidden_size = 512
cfg.SQLDecoder.readout = 'identity' # 'non_linear'
cfg.SQLDecoder.col_embed_size = 512
cfg.SQLDecoder.dropout = cfg.dropout
cfg.SQLDecoder.column_att = 'dot_prod' #'affine'
cfg.SQLDecoder.decode_max_time_step = 50


cfg.VOCAB = edict()
cfg.VOCAB.structure_tokens1 = ['visualize', 'select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'bin', 'by', 'distinct']
cfg.VOCAB.structure_tokens2 = ['join', 'on', 'as']
cfg.VOCAB.structure_tokens3 = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exist']
cfg.VOCAB.structure_tokens4 = ['-', '+', "*", '/']
cfg.VOCAB.structure_tokens5 = ['max', 'min', 'count', 'sum', 'avg']
cfg.VOCAB.structure_tokens6 = ['and', 'or', 'desc', 'asc']
cfg.VOCAB.structure_tokens7 = ['bar', 'pie', 'line', 'scatter', 'stack', 'group']
cfg.VOCAB.structure_tokens8 = ['minute', 'hour', 'day', 'weekday', 'month', 'quarter', 'year', 'udf']
cfg.VOCAB.structure_tokens9 = ['<eos>', '<pad>', '<unk>', '<go>', 't1', 't2', 't3', 't4', '1', '2', '*']

cfg.VOCAB.structure_tokens = cfg.VOCAB.structure_tokens1 + cfg.VOCAB.structure_tokens2 + cfg.VOCAB.structure_tokens3 + cfg.VOCAB.structure_tokens4 + \
        cfg.VOCAB.structure_tokens5 + cfg.VOCAB.structure_tokens6 + cfg.VOCAB.structure_tokens7 + cfg.VOCAB.structure_tokens8 + cfg.VOCAB.structure_tokens9

