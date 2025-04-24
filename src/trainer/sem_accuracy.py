#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/12/1
import numpy as np
from src.rule.semQL import Root, Root0, Root1, Vis, Sel, N, A, C, T, Group, Bin, Sup, Order, Filter

'''
    once obtain vis text, transfer text to VQL, and split into part
    measure tree_accuracy / tree component accurcy
'''

def tree_component_accuracy(pred, target):
    pass

def to_VQL(seq):
    # print("seq:\t", seq)
    seqs = seq.split()
    seqs = [eval(x) for x in seqs]
    vql_dict = {}
    # vql_dict['vis'] = ''
    if len(seq) == 0:
        vql_dict['vis'] = ''
        vql_dict['axis'] = ''
        vql_dict['data'] = ''
        vql_dict['vql'] = ''
        return vql_dict
    while True:
        component = seqs.pop(0)
        if isinstance(component, Vis):
            vql_dict['vis'] = str(component)
        if isinstance(component, Sel):
            seqs.pop(0)
            vql_dict['axis'] = []
            while isinstance(seqs[0], A) or isinstance(seqs[0], C) or isinstance(seqs[0], T) or isinstance(seqs[0], N):
                component = seqs.pop(0)
                vql_dict['axis'].append(component)
                if len(seqs) == 0:
                    break
            vql_dict['axis'] = ' '.join([str(x) for x in vql_dict['axis']])
            break
    vql_dict['data'] = ' '.join([str(x) for x in seqs])
    vql_dict['vql'] = seq
    return vql_dict


def tree_accuracy(preds, targets):
    num_tree, num_vis, num_axis, num_data = 0, 0, 0, 0
    for pred, target in zip(preds, targets):
        # print('--------------------------------')
        # print("before\n", pred)
        try:
            pred_dict = to_VQL(pred)
            target_dict = to_VQL(target)
        except:
            continue
        # print('--------------------------------')
        # print('pred_dict', pred_dict)
        # print('target_dict', target_dict)

        # print("pred_dict['axis']",  pred_dict['axis'])
        # print("pred_dict['data']",  pred_dict['data'])

        # print("\n")
        # print("target_dict['axis']",  target_dict['axis'])
        # print("target_dict['data']",  target_dict['data'])
        if pred_dict['vql'] == target_dict['vql']:
            num_tree += 1
        if pred_dict['vis'] == target_dict['vis']:
            num_vis += 1
        if pred_dict['axis'] == target_dict['axis']:
            num_axis += 1
        if pred_dict['data'] == target_dict['data']:
            num_data += 1
    acc_tree, acc_vis, acc_axis, acc_data = np.array([num_tree, num_vis, num_axis, num_data]) / len(preds)
    return acc_tree, acc_vis, acc_axis, acc_data





if __name__ == '__main__':
    #text = 'visualize bar select date of enrolment count date of enrolment from student as t1 join employee as t2 on t1 emp id = t2 emp id where t2 project name = 1 bin date by weekday'
    text = "Root0(1) Vis(0) Root1(3) Root(10) Sel(0) N(1) A(0) C(8) T(1) A(3) C(8) T(1) Group(0) A(0) C(8) T(1) Order(1) A(3) C(8) T(1)"
    to_VQL(text) 
