from openai import OpenAI
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import os
import time

client = OpenAI(
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
)

Embedding_file_path = "./nvBench-Rob/nlq_embedding_test.pkl"

# 获取问题和文档中每个句子的嵌入向量
def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=text, model=model)
    # 确保我们正确地处理响应数据
    embedding = response.data[0].embedding
    # 将嵌入向量转换为numpy数组
    return np.array(embedding)

# 输入问题和文档
document = pd.concat([
    pd.read_csv("./nvBench-Rob/dev_nlq/dev_nlq.csv"),
    pd.read_csv("./nvBench-Rob/dev_schema/dev_schema.csv"),
    pd.read_csv("./nvBench-Rob/dev_nlq_schema/dev_nlq_schema.csv")
])
document = document.drop_duplicates(subset=['nl_queries'])

nlq_embedding = []
if os.path.exists(Embedding_file_path):
    with open(Embedding_file_path, "rb") as f:
        nlq_embedding = pickle.load(f)

processed_count = len(nlq_embedding)

for index, row in tqdm(document.iterrows(), total=len(document)):
    if index < processed_count:
        continue
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

    new_entry = {"NLQ": nlq, "Embedding": embedding, "VQL": vql, "db_id": db_id}
    nlq_embedding.append(new_entry)

    with open(Embedding_file_path, "wb") as f:
        pickle.dump(nlq_embedding, f)
