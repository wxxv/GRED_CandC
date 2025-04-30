#!/usr/bin/env python
# coding=utf-8
# @summerzhao: 2021/09/02
from nltk import word_tokenize
'''
    once obtain vis text, transfer text to VQL, and split intp part
    measure tree_accuracy / tree component accurcy
'''
structure_tokens1 = ['visualize', 'select', 'from', 'where', 'group by', 'order by', 'limit', 'intersect', 'union', 'except', 'bin', 'by']
structure_tokens2 = ['join', 'on', 'as']
structure_tokens3 = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists']
structure_tokens4 = ['-', '+', "*", '/']
structure_tokens5 = ['max', 'min', 'count', 'sum', 'avg']
structure_tokens6 = ['and', 'or', 'desc', 'asc']
structure_tokens7 = ['bar', 'pie', 'line', 'scatter', 'stacked bar', 'grouping line', 'grouping scatter']
structure_tokens = structure_tokens1 + structure_tokens2 + structure_tokens3 + structure_tokens4 + structure_tokens5 \
        + structure_tokens6 + structure_tokens7

key_tokens = ['visualize', 'select', 'from', 'where', 'group', 'order', 'bin']

def to_VQL(text):
    '''
        text to VQL
    '''
    # print(text)
    text = text.lower()
    text=text.replace("."," . ").replace("(", " ( ").replace(")", " ) ").replace("IS NOT NULL", "!= \"null\"")
    text = word_tokenize(text)
    VQL = []
    binning = True
    keywords_dict = {'group': [], 'bin': [], 'order': [], 'where': [], 'select': [], 'from': [], 'visualize': []}
    keyword = ''

    for i, token in enumerate(text):
        
        if token in key_tokens:
            keyword = token
            continue

        keywords_dict[keyword] = keywords_dict.get(keyword, []) + [token]
    return keywords_dict



def tree_accuracy(preds, targets, final_dvq):
    num_tree, num_vis, num_axis, num_data = 0, 0, 0, 0
    for idx, (pred, target) in enumerate(zip(preds, targets)):
        pred_dict = to_VQL(pred)
        target_dict = to_VQL(target)
        final_dvq_dict = to_VQL(final_dvq[idx])
        print(pred)
        print(target)
        print(final_dvq[idx]
        if final_dvq_dict == target_dict:
            print("1"*50)
        if pred_dict == target_dict:
            num_tree += 1
            print("2"*50)
        print("-"*50)
        # else:
        #     print("pred:\t{}".format(pred))
        #     print("tgt:\t{}\n".format(target))
        if pred_dict['visualize'] == target_dict['visualize']:
            num_vis += 1
        if (pred_dict['select'] == target_dict['select']) and (pred_dict['from'] == target_dict['from']):
            num_axis += 1
        data_part = (target_dict['where'] == []) and (target_dict['group'] == []) and (target_dict['bin'] == []) and (target_dict['order'] == [])
        if not data_part and ((target_dict['where'] == pred_dict['where']) and 
                              (target_dict['group'] == pred_dict['group']) and 
                              (target_dict['bin'] == pred_dict['bin']) and 
                              (target_dict['order'] == pred_dict['order'])):
            num_data += 1

    acc_tree = num_tree / len(preds)
    acc_vis = num_vis / len(preds)
    acc_axis = num_axis / len(preds)
    acc_data = num_data / len(preds)
    return acc_tree, acc_vis, acc_axis, acc_data





if __name__ == '__main__':
    text = 'visualize bar select date of enrolment count date of enrolment from student as t1 join employee as t2 on t1 emp id = t2 emp id where t2 project name = 1 bin date by weekday'
    to_VQL(text)
