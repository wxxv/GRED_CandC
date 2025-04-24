import openai
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import time
import os

openai.api_base = "https://openkey.cloud/v1"

# 替换为你的OpenAI API密钥
openai.api_key = "sk-u4QlUZW8nLtaiqyv40338f916f5a4d6c81B42aB83b6fE567"

embedding_file_path = "../data/vql_embedding.pkl"
data_path = "../data/{}/{}_result_nlq_rag.json"
result_save_path = "../data/{}/{}_result_dvq_rag.json"

with open(embedding_file_path, 'rb') as f:
        vql_embedding_all = pickle.load(f)

# 获取问题和文档中每个句子的嵌入向量
def get_embedding(text, model="text-embedding-3-large"):
    # model = "text-embedding-3-large"
    response = openai.Embedding.create(input=text, engine=model)
    # 确保我们正确地处理响应数据
    embedding = response['data'][0]['embedding']
    # 将嵌入向量转换为numpy数组
    return np.array(embedding)


def get_dvqs_by_dvq_list(dvq:str, k=5):
    
    document_embeddings = pd.DataFrame(vql_embedding_all)['Embedding'].to_list()
    document = pd.DataFrame(vql_embedding_all)['VQL'].to_list()

    question_embedding = get_embedding(dvq)
    # 计算问题向量与文档中每个句子向量的相似度
    similarities = [1 - cosine(question_embedding, doc_embedding) for doc_embedding in document_embeddings]

    # 选择top-k个最相似的句子
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_sentences = [document[i] for i in top_k_indices]
    return top_k_sentences

# def remove_as(dvq:str):
#     tokens = dvq.replace(",", " , ").replace(".", " . ").replace("(", " ( ").replace(")", " ) ").split()
#     tokens_new = []
#     ref = {}
#     for id, token in enumerate(tokens):
#         if token.lower() == "as" and "T" not in tokens[id + 1]:
#             ref[tokens[id + 1]] = tokens[id - 1]
#             continue
#         if id > 0 and tokens[id - 1].lower() == "as" and "T" not in token:
#             continue

#         tokens_new.append(token)

#     dvq_new = " ".join(tokens_new).replace(" . ", ".").replace(" ( ", "(").replace(" ) ", ") ")

#     for name, c in ref.items():
#         # print("{} -> {}".format(name, c))
#         dvq_new = dvq_new.replace(name, c)

#     while "  " in dvq_new:
#         dvq_new = dvq_new.replace("  ", " ")

#     return dvq_new

# def remove_having(text:str):
#     structure_tokens1 = ['visualize', 'select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'having', 'bin']
#     text = text.split()
#     text_new = []
#     flag = ""
#     for token in text:
#         if token.lower() in structure_tokens1:
#             flag = token.lower()
#         if flag == "having":
#             continue
#         text_new.append(token)
#     text = text_new
#     return " ".join(text)

if __name__ == '__main__':
    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
        data_new = []
        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f: 
                data_new = json.load(f)
        with open(data_path.format(mode, mode), 'r') as f:
            data = json.load(f)


        for index, example in tqdm(enumerate(data), total=len(data)):
            
            if index < len(data_new):
                continue

            # dvq = remove_having(remove_as(example['predict_rag_nlq']))
            dvq = example['predict_rag_nlq']
            while "  " in dvq:
                dvq = dvq.replace("  ", " ")

            dvqs = get_dvqs_by_dvq_list(dvq, 10)
            
            example_new = example.copy()


            example_new['rag_dvqs'] = dvqs
            data_new.append(example_new)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
            