import openai
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import os
import time, json

openai.api_base = "https://openkey.cloud/v1"

# 替换为你的OpenAI API密钥
openai.api_key = "sk-u4QlUZW8nLtaiqyv40338f916f5a4d6c81B42aB83b6fE567"

# Embedding_file_path = "./vql_embedding.pkl"
Embedding_file_path = "../data//vql_embedding.pkl"

# 获取问题和文档中每个句子的嵌入向量
def get_embedding(text, model="text-embedding-ada-002"):
    model = "text-embedding-3-large"
    response = openai.Embedding.create(input=text, engine=model)
    # 确保我们正确地处理响应数据
    embedding = response['data'][0]['embedding']
    # 将嵌入向量转换为numpy数组
    return np.array(embedding)

# 输入问题和文档
document = pd.DataFrame(pd.concat([pd.read_json("./data/dev_nlq/dev_nlq.json"), pd.read_json("./data/dev_schema/dev_schema.json")])["vis_query"].to_list())['VQL'].to_list()
# document = sum(json.load(open("./dvqs_weak.json", 'r')).values(), [])
# pprint(document)
vql_embedding = []
if os.path.exists(Embedding_file_path):
    with open(Embedding_file_path, "rb") as f:
        vql_embedding = pickle.load(f)
    # pprint(vql_embedding)
# l = len(vql_embedding)

for vql in tqdm(document[len(vql_embedding):], initial=len(vql_embedding), total=len(document)):
    # print(vql)
    while True:
        try:
            embedding = get_embedding(vql)
            break
        except Exception as ex:
            print(ex)
            print("api errr... wait for 3s...")
            time.sleep(3)

    vql_embedding.append({"VQL":vql, "Embedding":embedding})

    with open(Embedding_file_path, "wb") as f:
        pickle.dump(vql_embedding, f)
