import json

import numpy as np

data_path = "../nvBench-Rob/dev_nlq_schema/result_rebuttal/dev_nlq_schema_confidence.json"

with open(data_path, 'r') as f:
    data = json.load(f)

data_new = []
sum_diff = 0
len_diff = 0
for d in data:
    record_name = d['record_name']
    db_id = d['db_id']
    target = d['target']
    nlq = d['nlq']
    predict_rag_nlq = d['predict_rag_nlq']
    predict_dvq_set = d['predict_dvq_set']
    target_prob = {index+1 : float(value) for index, value in enumerate(predict_dvq_set.values())}
    # print(target_prob)

    # 统计predict_rag_nlq各个数字出现的频率，计算概率
    frequency = {}
    for num in predict_rag_nlq:
        num = int(float(num))
        if num not in frequency:
            frequency[num] = 0
        frequency[num] += 1
    probability = {int(num): frequency[num] / len(predict_rag_nlq) for num in frequency}
    # print(probability)
    # exit()
    
    
    for k, v in target_prob.items():
        len_diff += len(target_prob)
        if k in probability:
            diff = np.abs(target_prob[k] - probability[k])
        else:
            diff = np.abs(target_prob[k] - 0)
        sum_diff += diff
print(sum_diff)
print(sum_diff / len_diff)