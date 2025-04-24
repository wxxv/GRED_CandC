#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/11/3

import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

from src.trainer import metrics
from src.config import cfg


class Trainer(object):
    def __init__(self, checkpoint_path, vocab):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.vocab = vocab
        self.metrics = metrics.Metrics()
        self.accuracy = metrics.Accuracy()

    def train_retrieval(self, model, opt, train_loader, dev_loader, num_epoches, clip_grad):
        logging.info('Training start')
        best_acc = 0
        for epoch in range(num_epoches):
            loss = self.train_retrieval_step(model, opt, train_loader, clip_grad)
            logging.info('Epoch {}: train loss {}'.format(epoch, loss))
            print('Epoch {}: train loss {}'.format(epoch, loss))

            if epoch % 5 == 0:
                with torch.no_grad():
                    dev_acc = self.test_retrieval_step(model, dev_loader, ifTest = False)
                if dev_acc >= best_acc:
                    torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'best_model.model'))
                    best_acc = dev_acc
                logging.info('Epoch {}. Dev acc: {}'.format(epoch, dev_acc))

            torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'new.model'))
        logging.info('Train complete')

    def test_retrieval(self, model, test_loader):
        logging.info('Test start')
        with torch.no_grad():
            test_acc = self.test_retrieval_step(model, test_loader, ifTest = True)
        print('test acc', test_acc)
        return test_acc

    def train_retrieval_step(self, model, opt, train_loader, clip_grad):
        model.train()
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        total_loss, total_num, start_iter = 0, 0, 0
        for i, (data) in enumerate(pbar, start=start_iter):
            examples = data
            opt.zero_grad()
            loss = model(examples)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            total_loss += loss.data.item() * len(examples)
            total_num += len(examples)
        loss_epoch = total_loss / total_num
        return loss_epoch

    def test_retrieval_step(self, model, dev_loader, ifTest = False):
        model.eval()
        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        start_iter, total_acc, total_num = 0, 0, 0

        for _, (data) in enumerate(pbar, start=start_iter):
            examples = data
            preds, labels = model.parse(examples)
            print("preds:\t", preds)
            print("labels:\t", labels)
            preds = preds.cpu()
            labels = torch.tensor(labels)
            acc = torch.sum(preds==labels) / len(examples)
            total_num += len(examples)

            total_acc += acc * len(examples)
        acc = total_acc / total_num
    
        return acc

    #------------------------------------------------------------- for revision ----------------------------------------------------------------------------- #

    def train_revision(self, model, opt, train_loader, dev_loader, num_epoches, clip_grad, decoder_type = 'rnn'):
        logging.info('Training start')
        best_loss = 0
        for epoch in range(num_epoches):
            if decoder_type in ['rnn', 'transformer']:
                loss = self.train_revision_step(model, opt, train_loader, clip_grad)
            else:
                loss = self.train_revision_step_SQL(model, opt, train_loader, clip_grad, epoch = epoch)
            logging.info('Epoch {}: train loss {}'.format(epoch, loss))
            print('Epoch {}: train loss {}'.format(epoch, loss))

            if epoch % 5 == 0:
                with torch.no_grad():
                    if decoder_type in ['rnn', 'transformer']:
                        scores, _, _ = self.test_revision_step(model, dev_loader, ifTest = False)
                        scores = np.round(scores, 4)
                        score = scores[0]
                        logging.info('Epoch {}. score_bleu: {}; score_rouge: {}; score_meteor: {}, acc: {}'.format(epoch, scores[0], scores[1:4], scores[4], scores[5]))
                        print('Epoch {}. score_bleu: {}; score_rouge: {}; score_meteor: {}, acc: {}'.format(epoch, scores[0], scores[1:4], scores[4], scores[5]))
                    else:
                        _, sketch_acc, acc = self.test_revision_step_SQL(model, dev_loader)
                        score = acc
                        logging.info('Epoch {}. sketch_acc: {:.4f}; acc: {:.4f}'.format(epoch, sketch_acc, acc))
                        print('Epoch {}. sketch_acc: {:.4f}; acc: {:.4f}'.format(epoch, sketch_acc, acc))

                if score >= best_loss:
                    torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'best_model.model'))
                    best_loss = score

            torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'new.model'))
        logging.info('Train complete')

    def test_revision(self, model, test_loader, decoder_type = 'rnn'):
        logging.info('Test start')
        json_datas = None
        with torch.no_grad():
            if decoder_type in ['rnn', 'transformer']:
                scores, preds, tgts = self.test_revision_step(model, test_loader, ifTest = True)
                json_datas = [{'pred': pred, 'tgt': tgt, 'acc_rnn': 1 if pred == tgt else 0} for (pred, tgt) in zip(preds, tgts)]
                acc = scores[5]
                logging.info('Test. score_bleu: {}; score_rouge: {}; score_meteor: {}, acc: {}'.format(scores[0], scores[1:4], scores[4], scores[5]))
                print('Test. score_bleu: {}; score_rouge: {}; score_meteor: {}, acc: {}'.format(scores[0], scores[1:4], scores[4], scores[5]))
            else:
                json_datas, sketch_acc, acc = self.test_revision_step_SQL(model, test_loader)
                logging.info('Test. sketch acc: {:.4f}, acc: {:.4f}'.format(sketch_acc, acc))
                print('Test. sketch acc: {:.4f}, acc: {:.4f}'.format(sketch_acc, acc))
        return json_datas

    def train_revision_step(self, model, opt, train_loader, clip_grad, ifeval = False):
        if ifeval:
            model.eval()
        else:
            model.train()
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        total_loss, total_num, start_iter = 0, 0, 0
        
        criterion = nn.NLLLoss(reduction = 'sum', ignore_index = self.vocab.pad)
        for i, (data) in enumerate(pbar, start=start_iter):
            examples = data
            opt.zero_grad()
            outputs, preds, tgts, saliency, rouges = model(examples)
            loss1 = -(rouges * torch.log(saliency+1e-8) + (1-rouges) * torch.log(1-saliency+1e-5))
            loss1 = loss1.mean()
            loss2 = criterion(outputs.view(-1, self.vocab.size_out) + 1e-8 , tgts.contiguous().view(-1))
            #loss = loss1 + loss2
            loss = loss2
            total_loss += loss.data.item() #* outputs.size()[0]
            total_num += outputs.size()[0]
            
            if not ifeval:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                opt.step()
            
        loss_epoch = total_loss / total_num
        return loss_epoch



    def test_revision_step(self, model, dev_loader, ifTest = True):
        model.eval()
        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        start_iter, total_num = 0, 0
        scores = []
        outputs = []
        preds_total = []
        tgts_total = []
        beam_size = cfg.SQLDecoder.beam_size

        for _, (data) in enumerate(pbar, start=start_iter):
            examples = data
            preds, tgts = model.parse(examples, ifTest = ifTest, beam_size = beam_size)
            total_num += tgts.size()[0]

            outputs += preds
            acc, score_bleu, score_rouge, score_meteor = self.get_metrics(preds, tgts)
            score = np.array([score_bleu] + score_rouge + [score_meteor, acc])
            scores.append(score * tgts.size()[0])
            
            self.print_results(preds, tgts, examples)
            
            preds_total.extend(preds)
            tgts_total.extend(tgts)

        if ifTest:
            preds_total = self.get_outputs(preds_total)
            tgts_total = self.get_outputs(tgts_total)
    
            acc_tree, acc_vis, acc_axis, acc_data = self.metrics.accuracy(preds_total, tgts_total, 'rnn')
            acc_com = (acc_vis + acc_axis + acc_data) / 3
            logging.info('Test.  acc_tree: {:.4f}, acc_vis: {:.4f}, acc_axis: {:.4f}, acc_data: {:.4f}, acc_com: {:.4f},'.format(
                acc_tree, acc_vis, acc_axis, acc_data, acc_com))
            print('Test.  acc_tree: {:.4f}, acc_vis: {:.4f}, acc_axis: {:.4f}, acc_data: {:.4f}, acc_com: {:.4f},'.format(
                acc_tree, acc_vis, acc_axis, acc_data, acc_com))
        scores = np.sum(np.array(scores), axis = 0) / total_num
        return scores, preds_total, tgts_total

    def train_revision_step_SQL(self, model, opt, train_loader, clip_grad, epoch = 0):
        loss_epoch_threshold = cfg.loss_epoch_threshold
        sketch_loss_coefficient = cfg.sketch_loss_coefficient

        model.train()
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        total_loss, total_num, start_iter = 0, 0, 0
        
        criterion = nn.NLLLoss(reduction = 'mean', ignore_index = self.vocab.pad)
        
        for i, (data) in enumerate(pbar, start=start_iter):
            examples = data
            opt.zero_grad()
            score, saliency, rouges = model(examples)

            loss_sketch = -score[0]
            loss_lf = -score[1]

            loss_sketch = torch.mean(loss_sketch)
            loss_lf = torch.mean(loss_lf)

            #loss1 = -(rouges * torch.log(saliency+1e-5) + (1-rouges) * torch.log(1-saliency+1e-5))
            #loss1 = loss1.mean()

            if epoch > loss_epoch_threshold:
                loss = loss_lf + sketch_loss_coefficient * loss_sketch #+ loss1
            else:
                loss = loss_lf + loss_sketch #+ loss1 * 5
            
            assert not torch.any(torch.isnan(loss) + torch.isinf(loss))
            #with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            
            print('sketch loss:', loss_sketch, 'lf loss:', loss_lf)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            for name, param in model.named_parameters():
                #assert torch.isnan(param.data).sum() == 0, print('parameter data before step: ', param.name, param.data)
                assert not torch.any(torch.isnan(param.data)), print('parameter data before step: ', param.name, param.data)

            
            opt.step()
            for name, param in model.named_parameters():
                assert not torch.any(torch.isnan(param.data)), print('parameter data after step: ', param.name, param.data)
                if param.grad is not None:
                    assert not torch.any(torch.isnan(param.grad)), print('parameter grad after step: ', param.name, param.grad)

            total_loss += loss.data.item() * len(examples)
            total_num += len(examples)
            pre_loss = loss_lf
        loss_epoch = total_loss / total_num
        return loss_epoch

    def test_revision_step_SQL(self, model, dev_loader, ifTest = False):
        model.eval()
        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        start_iter, total_num = 0, 0
        scores = []
        outputs = []
        json_datas = []
        preds = []
        tgts = []
        beam_size = cfg.SQLDecoder.beam_size
        sketch_correct, rule_label_correct, total = 0, 0, 0
        for _, (data) in enumerate(pbar, start=start_iter):
            examples = data
            for example in examples:
                results_all = model.parse([example], beam_size=beam_size)
                print('\n\n',results_all,'\n\n')
                results = results_all[0]
                list_preds = []
                try:
                    pred = " ".join([str(x) for x in results[0].actions])
                    for x in results:
                        list_preds.append(str(x.score) + " ".join([str(action) for action in x.actions]))
                except Exception as e:
                    print('error:', e, results_all)
                    pred = ""

                simple_json = example.sql_json['pre_sql']
                simple_json['sqls'] = simple_json['query']
                simple_json['sketch_result'] =  " ".join(str(x) for x in results_all[1])
                simple_json['model_result'] = pred
                simple_json['model_results'] = list_preds
                simple_json['sketch_true'] = 'sketch_false'
                simple_json['lf_true'] = 'lf_false'

                truth_sketch = " ".join([str(x) for x in example.sketch])
                truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

                if truth_sketch == simple_json['sketch_result']:
                    sketch_correct += 1
                    simple_json['sketch_true'] = 'sketch_trues'

                if truth_rule_label == simple_json['model_result']:
                    rule_label_correct += 1
                    simple_json['lf_true'] = 'lf_trues'
                
                preds.append(simple_json['model_result'])
                tgts.append(truth_rule_label)

                total += 1
                
                #keys = ['sqls', 'rule_label', 'sktech_result', 'model_result', 'model_results', 'sketch_true', 'lf_true']
                #simple_json = {key: value for key, value in simple_json.items() if key in keys}
                json_datas.append(simple_json)

        acc_tree, acc_vis, acc_axis, acc_data = self.metrics.accuracy(preds, tgts)
        acc_com = (acc_vis + acc_axis + acc_data) / 3
        logging.info('Test.  acc_tree: {:.4f}, acc_vis: {:.4f}, acc_axis: {:.4f}, acc_data: {:.4f}, acc_com: {:.4f},'.format(
            acc_tree, acc_vis, acc_axis, acc_data, acc_com))
        print('Test.  acc_tree: {:.4f}, acc_vis: {:.4f}, acc_axis: {:.4f}, acc_data: {:.4f}, acc_com: {:.4f},'.format(
            acc_tree, acc_vis, acc_axis, acc_data, acc_com))
        return json_datas, float(sketch_correct)/float(total), float(rule_label_correct)/float(total)

    def get_outputs(self, preds):
        if not isinstance(preds, list):
            preds = preds.cpu().numpy()
        preds = [[self.vocab.idx2word_out[i] for i in sentence] for sentence in preds]
        preds = self.remove_pad(preds)
        preds = [' '.join(x) for x in preds]
        return preds

    def print_results(self, preds, tgts, examples = None):
        preds = self.get_outputs(preds)
        tgts = self.get_outputs(tgts)
        for pred, tgt, example in zip(preds[:2], tgts[:2], examples[:2]):
            print('------------')
            print('pred:', pred)
            print('tgt:', tgt, '|| record_name: ', example.record_name)#, example.text_seqs)
        
        data = pd.DataFrame({'pred': preds, 'tgt': tgts})
        data.to_json('./results/predict.json', orient = 'records', indent = 1)
    
    def get_metrics(self, preds, targets):
        if not isinstance(preds, list):
            preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        preds = [[self.vocab.idx2word_out[i] for i in sentence] for sentence in preds]
        targets = [[self.vocab.idx2word_out[i] for i in sentence] for sentence in targets]

        preds = self.remove_pad(preds)
        targets = self.remove_pad(targets)
        score_bleu, score_rouge, score_meteor = self.metrics.score(preds, targets, 4)
        accuracy = self.accuracy.accuracy_split(preds, targets)
        return accuracy, score_bleu, score_rouge, score_meteor

    def remove_pad(self, preds):
        preds = [' '.join(x) for x in preds]
        preds = [x.split('<eos>')[0].split() for x in preds]
        preds = [pred if len(pred) > 0 else ['<pad>'] for pred in preds]
        return preds



