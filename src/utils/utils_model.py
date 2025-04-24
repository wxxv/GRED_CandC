# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# @summerzhao:2021/07/30
import os
import json
import time
import numpy as np
import pandas as pd
import torch


def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    return ret



def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

if __name__ == '__main__':
    sql_data, speech_data, table_data, _, _, _ = load_dataset(True)
    perm=np.random.permutation(len(sql_data))
    st, ed = 0, 4
    examples = to_batch_seq(sql_data, speech_data, table_data, perm, st, ed, is_train=True)

