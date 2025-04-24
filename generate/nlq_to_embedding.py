import openai
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import os
import time, json

openai.api_base = "https://openkey.cloud/v1"

# 替换为你的OpenAI API密钥
openai.api_key = "sk-u4QlUZW8nLtaiqyv40338f916f5a4d6c81B42aB83b6fE567"

# Embedding_file_path = "./nlq_embedding.pkl"
Embedding_file_path = "../data/nlq_embedding.pkl"

# 获取问题和文档中每个句子的嵌入向量
def get_embedding(text, model="text-embedding-ada-002"):
    model = "text-embedding-3-large"
    response = openai.Embedding.create(input=text, engine=model)
    # 确保我们正确地处理响应数据
    embedding = response['data'][0]['embedding']
    # 将嵌入向量转换为numpy数组
    return np.array(embedding)

# 输入问题和文档
# document = pd.concat([pd.read_csv("./data/dev_nvBench/dev_nvBench.csv"), pd.read_csv("./data/test_nvBench/test_nvBench.csv")])
document = pd.read_csv("../data/train_nvBench/train_nvBench.csv").dropna(subset=['nl_queries'])
# document = sum(json.load(open("./dvqs_weak.json", 'r')).values(), [])
# pprint(document)

nlq_embedding = []
if os.path.exists(Embedding_file_path):
    with open(Embedding_file_path, "rb") as f:
        nlq_embedding = pickle.load(f)
    # pprint(nlq_embedding)
# l = len(nlq_embedding)

for index, row in tqdm(document.iterrows(), total=len(document)):
    if index < len(nlq_embedding):
        continue
    # print(nlq)
    nlq = row['nl_queries']
    vql = eval(row['vis_query'])['VQL']
    db_id = row['db_id']

    while True:
        try:
            embedding = get_embedding(nlq)
            break
        except Exception as ex:
            print("NLQ:\t{}".format(nlq))
            print(ex)
            print("api errr... wait for 3s...")
            time.sleep(3)

    nlq_embedding.append({"NLQ":nlq, "Embedding":embedding, "VQL":vql, "db_id":db_id})

    with open(Embedding_file_path, "wb") as f:
        pickle.dump(nlq_embedding, f)
