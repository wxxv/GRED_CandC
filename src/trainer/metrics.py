#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/7/20

import nltk
import nltk.translate.bleu_score as bleu
from nltk.translate import meteor_score
from rouge import Rouge

from src.trainer import sem_accuracy, sql_accuracy

class Metrics(object):
    def __init__(self):
        self.metric_rouge = Rouge()

    def bleu(self, preds, targets, n = 4):
        scores = []
        for pred, target in zip(preds, targets):
            #print(pred, target)
            if n == 1:
                score = bleu.sentence_bleu([pred], target, weights=(1,0,0,0))
            elif n == 2:
                score = bleu.sentence_bleu([pred], target, weights=(0.5,0.5,0,0))
            elif n == 3:
                score = bleu.sentence_bleu([pred], target, weights=(0.33, 0.33, 0.33, 0))
            else:
                score = bleu.sentence_bleu([pred], target, weights=(0.25, 0.25, 0.25, 0.25))
            scores.append(score)
        return sum(scores) / len(scores)

    def rouge(self, preds, targets):
        #f:F1值  p：查准率  R：召回率
        scores = self.metric_rouge.get_scores(targets, preds, avg = True)
        return [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]

    def meteor(self, preds, targets):
        scores = []
        for pred, target in zip(preds, targets):
            score = meteor_score.single_meteor_score(pred, target)
            scores.append(score)
        return sum(scores) / len(scores)
            

    def score(self, preds, targets, n):
        bleu_score = self.bleu(preds, targets, n)
        preds = [' '.join(x) for x in preds]
        targets = [' '.join(x) for x in targets]

        rouge_score = self.rouge(preds, targets)

        meteor_score_ = self.meteor(preds, targets)
        return bleu_score, rouge_score, meteor_score_   

    def accuracy(self, preds, targets, final_dvq, sql_type = ''):
        if sql_type == 'sql':
            acc_tree, acc_vis, acc_axis, acc_data = sem_accuracy.tree_accuracy(preds, targets)
        else:
            acc_tree, acc_vis, acc_axis, acc_data = sql_accuracy.tree_accuracy(preds, targets, final_dvq)
        return acc_tree, acc_vis, acc_axis, acc_data


class Accuracy(object):
    def __init__(self):
        pass

    @staticmethod
    def accuracy_split(preds, targets):
        acc = [1 if pred == target else 0 for pred, target in zip(preds, targets)]
        return sum(acc) / len(acc)

