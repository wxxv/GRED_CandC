#!/usr/bin/env python
# coding=utf-8
# @summerzhao: inherit from IRNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable

from src.config import cfg
from src.rule.beam import Beams, ActionInfo
from src.utils.dataset_semQL import Batch, gen_x_batch
from src.models.common_modules import nn_utils
from src.models.common_modules.pointer_net import PointerNet
from src.rule import semQL as define_rule

class SQLDecoder(nn.Module):
    def __init__(self, grammar, vocab = None, word_encoder = None):
        super(SQLDecoder, self).__init__()
        args = cfg
        self.args = args
        self.grammar = grammar
        self.vocab = vocab
        self.word_encoder = word_encoder
        self.use_column_pointer = args.SQLDecoder.column_pointer

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor


        input_dim = args.SQLDecoder.action_embed_size + \
                    args.SQLDecoder.att_vec_size  + \
                    args.SQLDecoder.type_embed_size
        # previous action (e.g. except R R)
        # input feeding, attention(previous_hidden)
        # pre type embedding (e.g. Root1)

        self.lf_decoder_lstm = nn.LSTMCell(input_dim, args.SQLDecoder.hidden_size)

        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.SQLDecoder.hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.SQLDecoder.hidden_size, args.SQLDecoder.hidden_size)

        self.att_sketch_linear = nn.Linear(args.SQLDecoder.hidden_size, args.SQLDecoder.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(args.SQLDecoder.hidden_size, args.SQLDecoder.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(args.SQLDecoder.hidden_size + args.SQLDecoder.hidden_size, args.SQLDecoder.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(args.SQLDecoder.hidden_size + args.SQLDecoder.hidden_size, args.SQLDecoder.att_vec_size, bias=False)

        self.prob_att = nn.Linear(args.SQLDecoder.att_vec_size, 1)
        self.prob_len = nn.Linear(1, 1)

        self.col_type = nn.Linear(4, args.SQLDecoder.col_embed_size)
        self.sketch_encoder = nn.LSTM(args.SQLDecoder.action_embed_size, args.SQLDecoder.action_embed_size // 2, bidirectional=True,
                                      batch_first=True)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.SQLDecoder.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.SQLDecoder.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        self.att_project = nn.Linear(args.SQLDecoder.hidden_size + args.SQLDecoder.type_embed_size, args.SQLDecoder.hidden_size)

        self.N_embed = nn.Embedding(len(define_rule.N._init_grammar()), args.SQLDecoder.action_embed_size)

        self.read_out_act = F.tanh if args.SQLDecoder.readout == 'non_linear' else nn_utils.identity

        self.query_vec_to_action_embed = nn.Linear(args.SQLDecoder.att_vec_size, args.SQLDecoder.action_embed_size,
                                                   bias=args.SQLDecoder.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        #self.q_att = nn.Linear(args.SQLDecoder.hidden_size, args.SQLDecoder.embed_size)

        self.column_rnn_input = nn.Linear(args.SQLDecoder.col_embed_size, args.SQLDecoder.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(args.SQLDecoder.col_embed_size, args.SQLDecoder.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.SQLDecoder.hidden_size, args.SQLDecoder.col_embed_size, attention_type=args.SQLDecoder.column_att)

        self.table_pointer_net = PointerNet(args.SQLDecoder.hidden_size, args.SQLDecoder.col_embed_size, attention_type=args.SQLDecoder.column_att)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)
        print('Use Column Pointer: ', True if self.use_column_pointer else False)

    def forward(self, batch, encoder_outputs, text_embedding = None):
        args = self.args
        examples = batch.examples
        table_appear_mask = batch.table_appear_mask
        
        src_encodings = encoder_outputs
        src_encodings = self.dropout(src_encodings)
        #text_embedding = self.dropout(text_embedding)
        #text_embedding = src_encodings[:, :text_len]

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings) # FC
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings) # FC

        init_cell = F.adaptive_avg_pool1d(encoder_outputs.transpose(1, 2), 1)
        init_cell = init_cell.squeeze(-1)
        dec_init_vec = self.init_decoder_state(init_cell)
        h_tm1 = dec_init_vec

        action_probs = [[] for _ in range(encoder_outputs.size()[0])]

        zero_action_embed = Variable(self.new_tensor(args.SQLDecoder.action_embed_size).zero_())
        zero_type_embed = Variable(self.new_tensor(args.SQLDecoder.type_embed_size).zero_())

        sketch_attention_history = list()

        for t in range(batch.max_sketch_num): # max prediction_step, predict step by step
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_(),
                             requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, example in enumerate(examples):

                    if t < len(example.sketch):
                        # get the last action
                        # This is the action embedding
                        action_tm1 = example.sketch[t - 1]
                        if type(action_tm1) in [define_rule.Root0,
                                                define_rule.Root1,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.Bin,
                                                define_rule.N,
                                                define_rule.Order,
                                                define_rule.Vis,
                                                define_rule.Group]:
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        else:
                            print(action_tm1, 'only for sketch')
                            quit()
                            a_tm1_embed = zero_action_embed
                            pass
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, example in enumerate(examples):
                    if t < len(example.sketch):
                        action_tm = example.sketch[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)
            src_mask = batch.src_token_mask
            #src_mask = None
            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_sketch_linear, self.sketch_decoder_lstm,
                                                 self.sketch_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)
            sketch_attention_history.append(att_t)

            # get the Root possibility
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_prob_var = torch.stack(
            [(torch.stack(action_probs_i, dim=0)+1e-8).log().sum() for action_probs_i in action_probs], dim=0)

        table_embedding, _ = gen_x_batch(batch.table_sents, self.vocab.word2idx, self.word_encoder) # average word embedding
        src_embedding, _ = gen_x_batch(batch.src_sents, self.vocab.word2idx, self.word_encoder)
        schema_embedding, _ = gen_x_batch(batch.table_names, self.vocab.word2idx, self.word_encoder)

        # get emb differ
        embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                 table_unk_mask=batch.table_unk_mask)

        schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                              table_unk_mask=batch.schema_token_mask)

        tab_ctx = (text_embedding.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (text_embedding.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)



        table_embedding = table_embedding + tab_ctx
        schema_embedding = schema_embedding + schema_ctx

        #print('table_embedding', table_embedding.size(), schema_embedding.size())
        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        table_embedding = table_embedding + col_type_var #initial_embedding + context_vector + type_embedding

        batch_table_dict = batch.col_table_dict
        table_enable = np.zeros(shape=(len(examples)))
        action_probs = [[] for _ in examples]

        h_tm1 = dec_init_vec

        for t in range(batch.max_action_num):
            if t == 0:
                # x = self.lf_begin_vec.unsqueeze(0).repeat(len(batch), 1)
                x = Variable(self.new_tensor(len(batch), self.lf_decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                pre_types = []

                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm1 = example.tgt_actions[t - 1]
                        if type(action_tm1) in [define_rule.Root1,
                                                define_rule.Root0,
                                                define_rule.Root,
                                                define_rule.Sel,
                                                define_rule.Filter,
                                                define_rule.Sup,
                                                define_rule.N,
                                                define_rule.Order,
                                                define_rule.Bin,
                                                define_rule.Vis, 
                                                define_rule.Group
                                                ]:

                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]

                        else:
                            if isinstance(action_tm1, define_rule.C):
                                a_tm1_embed = self.column_rnn_input(table_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.T):
                                a_tm1_embed = self.column_rnn_input(schema_embedding[e_id, action_tm1.id_c])
                            elif isinstance(action_tm1, define_rule.A):
                                a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                            else:
                                print(action_tm1, 'not implement')
                                quit()
                                a_tm1_embed = zero_action_embed
                                pass

                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                # tgt t-1 action type
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_tm = example.tgt_actions[t - 1]
                        pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    else:
                        pre_type = zero_type_embed
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)

                inputs.append(pre_types)

                x = torch.cat(inputs, dim=-1)

            src_mask = batch.src_token_mask
            #src_mask = None

            (h_t, cell_t), att_t, aw = self.step(x, h_tm1, src_encodings,
                                                 utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                                 self.lf_att_vec_linear,
                                                 src_token_mask=src_mask, return_att_weight=True)

            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            table_appear_mask_val = torch.from_numpy(table_appear_mask)
            if self.cuda:
                table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
            else:
                weights = self.column_pointer_net(src_encodings=table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)

            weights.data.masked_fill_(batch.table_token_mask.bool(), -float('inf'))

            column_attention_weights = F.softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))
            table_dict = [batch_table_dict[x_id][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.softmax(table_weights, dim=-1)
            # now get the loss
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        table_appear_mask[e_id, action_t.id_c] = 1
                        table_enable[e_id] = action_t.id_c
                        act_prob_t_i = column_attention_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.T):
                        act_prob_t_i = table_weights[e_id, action_t.id_c]
                        action_probs[e_id].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                        action_probs[e_id].append(act_prob_t_i)
                    else:
                        pass
            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t
        lf_prob_var = torch.stack(
            [(torch.stack(action_probs_i, dim=0)+1e-8).log().sum() for action_probs_i in action_probs], dim=0)

        return [sketch_prob_var, lf_prob_var]

    def parse(self, batch, encoder_outputs, text_embedding = None, beam_size=5):
        """
        one example a time
        :param examples:
        :param beam_size:
        :return:
        """
        args = self.args
        examples = batch.examples
        table_appear_mask = batch.table_appear_mask
        
        src_encodings = encoder_outputs
        #src_encodings = self.dropout(src_encodings)
        #text_embedding = self.dropout(text_embedding)
        #text_embedding = src_encodings[:, :text_len]

        utterance_encodings_sketch_linear = self.att_sketch_linear(src_encodings) # FC
        utterance_encodings_lf_linear = self.att_lf_linear(src_encodings) # FC

        init_cell = F.adaptive_avg_pool1d(encoder_outputs.transpose(1, 2), 1)
        init_cell = init_cell.squeeze(-1)
        dec_init_vec = self.init_decoder_state(init_cell)
        h_tm1 = dec_init_vec

        #action_probs = [[] for _ in range(encoder_outputs.size()[0])]


        t = 0
        beams = [Beams(is_sketch=True)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.SQLDecoder.decode_max_time_step:
            hyp_num = len(beams)
            exp_src_enconding = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_sketch_linear = utterance_encodings_sketch_linear.expand(hyp_num,
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           1),
                                                                                       utterance_encodings_sketch_linear.size(
                                                                                           2))
            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.sketch_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order,
                                            define_rule.Root0,
                                            define_rule.Vis,
                                            define_rule.Group,
                                            define_rule.Bin]:
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_enconding,
                                             exp_src_encodings_sketch_linear, self.sketch_decoder_lstm,
                                             self.sketch_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                action_class = hyp.get_availableClass()
                #print(action_class)
                if action_class in [
                                    define_rule.Root1,
                                    define_rule.Root0,
                                    define_rule.Root,
                                    define_rule.Sel,
                                    define_rule.Filter,
                                    define_rule.Sup,
                                    define_rule.N,
                                    define_rule.Order,
                                    define_rule.Bin,
                                    define_rule.Vis, 
                                    define_rule.Group]:
                    possible_productions = self.grammar.get_production(action_class)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]
                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': action_class, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    print('wrong class', action_class)
                    raise RuntimeError('No right action class')

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]
                action_type_str = hyp_meta_entry['action_type']
                prod_id = hyp_meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']
                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                beams = new_beams
                t += 1
            else:
                break

        # now get the sketch result
        completed_beams.sort(key=lambda hyp: -hyp.score)
        if len(completed_beams) == 0:
            return [[], []]

        sketch_actions = completed_beams[0].actions
        # sketch_actions = examples.sketch

        padding_sketch = self.padding_sketch(sketch_actions)

        table_embedding, _ = gen_x_batch(batch.table_sents, self.vocab.word2idx, self.word_encoder)
        src_embedding, _ = gen_x_batch(batch.src_sents, self.vocab.word2idx, self.word_encoder)
        schema_embedding, _ = gen_x_batch(batch.table_names, self.vocab.word2idx, self.word_encoder)

        # get emb differ
        embedding_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=table_embedding,
                                                 table_unk_mask=batch.table_unk_mask)

        schema_differ = self.embedding_cosine(src_embedding=src_embedding, table_embedding=schema_embedding,
                                              table_unk_mask=batch.schema_token_mask)
        #print('----embedding_differ', embedding_differ.size(), src_embedding.size(), table_embedding.size())

        tab_ctx = (text_embedding.unsqueeze(1) * embedding_differ.unsqueeze(3)).sum(2)
        schema_ctx = (text_embedding.unsqueeze(1) * schema_differ.unsqueeze(3)).sum(2)


        table_embedding = table_embedding + tab_ctx

        schema_embedding = schema_embedding + schema_ctx

        col_type = self.input_type(batch.col_hot_type)

        col_type_var = self.col_type(col_type)

        table_embedding = table_embedding + col_type_var

        batch_table_dict = batch.col_table_dict

        h_tm1 = dec_init_vec

        t = 0
        beams = [Beams(is_sketch=False)]
        completed_beams = []

        while len(completed_beams) < beam_size and t < self.args.SQLDecoder.decode_max_time_step:
            hyp_num = len(beams)

            # expand value
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_utterance_encodings_lf_linear = utterance_encodings_lf_linear.expand(hyp_num,
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         1),
                                                                                     utterance_encodings_lf_linear.size(
                                                                                         2))
            exp_table_embedding = table_embedding.expand(hyp_num, table_embedding.size(1),
                                                         table_embedding.size(2))

            exp_schema_embedding = schema_embedding.expand(hyp_num, schema_embedding.size(1),
                                                           schema_embedding.size(2))


            table_appear_mask = batch.table_appear_mask
            table_appear_mask = np.zeros((hyp_num, table_appear_mask.shape[1]), dtype=np.float32)
            table_enable = np.zeros(shape=(hyp_num))
            for e_id, hyp in enumerate(beams):
                for act in hyp.actions:
                    if type(act) == define_rule.C:
                        table_appear_mask[e_id][act.id_c] = 1
                        table_enable[e_id] = act.id_c

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.lf_decoder_lstm.input_size).zero_())
            else:
                a_tm1_embeds = []
                pre_types = []
                for e_id, hyp in enumerate(beams):
                    action_tm1 = hyp.actions[-1]
                    if type(action_tm1) in [define_rule.Root0,
                                            define_rule.Root1,
                                            define_rule.Root,
                                            define_rule.Sel,
                                            define_rule.Filter,
                                            define_rule.Sup,
                                            define_rule.N,
                                            define_rule.Order,
                                            define_rule.Group,
                                            define_rule.Bin,
                                            define_rule.Vis]:

                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        hyp.sketch_step += 1
                    elif isinstance(action_tm1, define_rule.C):
                        a_tm1_embed = self.column_rnn_input(table_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.T):
                        a_tm1_embed = self.column_rnn_input(schema_embedding[0, action_tm1.id_c])
                    elif isinstance(action_tm1, define_rule.A): # or isinstance(action_tm1, define_rule.V):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    else:
                        raise ValueError('unknown action %s' % action_tm1)

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                for e_id, hyp in enumerate(beams):
                    action_tm = hyp.actions[-1]
                    pre_type = self.type_embed.weight[self.grammar.type2id[type(action_tm)]]
                    pre_types.append(pre_type)

                pre_types = torch.stack(pre_types)

                inputs.append(att_tm1)
                inputs.append(pre_types)
                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_utterance_encodings_lf_linear, self.lf_decoder_lstm,
                                             self.lf_att_vec_linear,
                                             src_token_mask=None)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            table_appear_mask_val = torch.from_numpy(table_appear_mask)

            if self.args.cuda: table_appear_mask_val = table_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = torch.sigmoid(self.prob_att(att_t))
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * table_appear_mask_val * gate + self.column_pointer_net(
                    src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                    src_token_mask=None) * (1 - table_appear_mask_val) * (1 - gate)
                # weights = weights + self.col_attention_out(exp_embedding_differ).squeeze()
            else:
                weights = self.column_pointer_net(src_encodings=exp_table_embedding, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=batch.table_token_mask)
            weights.data.masked_fill_(batch.table_token_mask.bool(), -float('inf'))
            # weights.data.masked_fill_(exp_col_pred_mask, -float('inf'))

            column_selection_log_prob = F.log_softmax(weights, dim=-1)

            table_weights = self.table_pointer_net(src_encodings=exp_schema_embedding, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            schema_token_mask = batch.schema_token_mask.expand_as(table_weights)
            table_weights.data.masked_fill_(schema_token_mask.bool(), -float('inf'))

            table_dict = [batch_table_dict[0][int(x)] for x_id, x in enumerate(table_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            table_weights = F.log_softmax(table_weights, dim=-1)

            new_hyp_meta = []
            for hyp_id, hyp in enumerate(beams):
                # TODO: should change this
                if type(padding_sketch[t]) == define_rule.A:
                    possible_productions = self.grammar.get_production(define_rule.A)
                    for possible_production in possible_productions:
                        prod_id = self.grammar.prod2id[possible_production]
                        prod_score = apply_rule_log_prob[hyp_id, prod_id]

                        new_hyp_score = hyp.score + prod_score.data.cpu()
                        meta_entry = {'action_type': define_rule.A, 'prod_id': prod_id,
                                      'score': prod_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)

                elif type(padding_sketch[t]) == define_rule.C:
                    for col_id, _ in enumerate(batch.table_sents[0]):
                        col_sel_score = column_selection_log_prob[hyp_id, col_id]
                        new_hyp_score = hyp.score + col_sel_score.data.cpu()
                        meta_entry = {'action_type': define_rule.C, 'col_id': col_id,
                                      'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                elif type(padding_sketch[t]) == define_rule.T:
                    for t_id, _ in enumerate(batch.table_names[0]):
                        t_sel_score = table_weights[hyp_id, t_id]
                        new_hyp_score = hyp.score + t_sel_score.data.cpu()

                        meta_entry = {'action_type': define_rule.T, 't_id': t_id,
                                      'score': t_sel_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                else:
                    prod_id = self.grammar.prod2id[padding_sketch[t].production]
                    new_hyp_score = hyp.score + torch.tensor(0.0)
                    meta_entry = {'action_type': type(padding_sketch[t]), 'prod_id': prod_id,
                                  'score': torch.tensor(0.0), 'new_hyp_score': new_hyp_score,
                                  'prev_hyp_id': hyp_id}
                    new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.stack([x['new_hyp_score'] for x in new_hyp_meta], dim=0)
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_beams)))

            live_hyp_ids = []
            new_beams = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = beams[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if 'prod_id' in hyp_meta_entry:
                    prod_id = hyp_meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = hyp_meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = hyp_meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = hyp_meta_entry['score']

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score
                new_hyp.inputs.extend(prev_hyp.inputs)

                if new_hyp.is_valid is False:
                    continue

                if new_hyp.completed:
                    completed_beams.append(new_hyp)
                else:
                    new_beams.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]

                beams = new_beams
                t += 1
            else:
                break

        completed_beams.sort(key=lambda hyp: -hyp.score)

        return [completed_beams, sketch_actions]

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, decoder, attention_func, src_token_mask=None,
             return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = decoder(x, h_tm1)
        #print('x', x.size(), h_tm1[0].size(), src_encodings.size(), src_encodings_att_linear.size(), src_token_mask.size())

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = torch.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def init_decoder_state(self, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def embedding_cosine(self, src_embedding, table_embedding, table_unk_mask):
        embedding_differ = []
        for i in range(table_embedding.size(1)):
            one_table_embedding = table_embedding[:, i, :]
            one_table_embedding = one_table_embedding.unsqueeze(1).expand(table_embedding.size(0),
                                                                          src_embedding.size(1),
                                                                          table_embedding.size(2))

            topk_val = F.cosine_similarity(one_table_embedding, src_embedding, dim=-1)

            embedding_differ.append(topk_val)
        embedding_differ = torch.stack(embedding_differ).transpose(1, 0)
        embedding_differ.data.masked_fill_(table_unk_mask.unsqueeze(2).expand(
            table_embedding.size(0),
            table_embedding.size(1),
            embedding_differ.size(2)
        ).bool(), 0)

        return embedding_differ

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def padding_sketch(self, sketch):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup  or type(action) == define_rule.Bin or (type(action) == define_rule.Group and action.id_c == 0):
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))

        return padding_result

