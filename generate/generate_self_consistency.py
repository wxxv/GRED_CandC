from math import e
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time
from collections import Counter
from config import OPENAI_BASE_URL, OPENAI_API_KEY

client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

embedding_file_path = "./nvBench-Rob/nlq_embedding.pkl"
embedding_test_file_path = "./nvBench-Rob/nlq_embedding_test.pkl"
data_file_path = "./nvBench-Rob/{}/{}.csv"
result_save_path = "./nvBench-Rob/{}/result_rebuttal/{}_result_self_consistency.json"

# Self-consistency 配置
NUM_SAMPLES = 5  # 生成的候选答案数量
TEMPERATURE = 0.7

with open(embedding_file_path, 'rb') as f:
        nlq_embedding_all = pickle.load(f)

with open(embedding_test_file_path, 'rb') as f:
        nlq_test_embedding_all = pickle.load(f)

def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def reconstruct_schemas(db_name):
  db = {}
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  for name, group in df:
    db[name] = []
    for index, row in group.iterrows():
      db[name].append(row[" Field Name"])
  return db

def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "# Table " + name+ ', columns = [ '
    for index, row in group.iterrows():
      output += row[" Field Name"] + ' , '
    output = output[:-2] + ']\n'
  return output

def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "# Foreign_keys = [ "
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ' , '
  output= output[:-2] + "]"
  return output

def reconstruct_fk(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  db_fk = {}
  for index, row in df.iterrows():
    if row['First Table Name'] not in db_fk:
       db_fk[row['First Table Name']] = {row['Second Table Name']:(row['First Table Foreign Key'], row['Second Table Foreign Key'])}
    else:
       db_fk[row['First Table Name']][row['Second Table Name']] = (row['First Table Foreign Key'], row['Second Table Foreign Key'])

    if row['Second Table Name'] not in db_fk:
       db_fk[row['Second Table Name']] = {row['First Table Name']:(row['Second Table Foreign Key'], row['First Table Foreign Key'])}
    else:
       db_fk[row['First Table Name']][row['First Table Name']] = (row['Second Table Foreign Key'], row['First Table Foreign Key'])

  return db_fk

def generate_schema(db_name:str):
  schema = "### Database Schemas:\n" + find_fields_MYSQL_like(db_name) + find_foreign_keys_MYSQL_like(db_name)
  return schema

DATASET_SCHEMA = './nvBench-Rob/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)

def generate_reply_self_consistency(messages, n=5, temperature=0.7):
    """
    生成多个候选答案用于 self-consistency
    返回: (候选答案列表, token使用信息, 延迟时间)
    """
    start_time = time.time()
    
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4o-mini",
        messages=messages,
        n=n,
        stream=False,
        temperature=temperature,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    end_time = time.time()
    latency = end_time - start_time
    
    all_candidates = []
    for i in range(n):
        vql = completions.choices[i].message.content
        all_candidates.append(vql)
    
    # 获取 token 使用信息
    token_usage = {
        "prompt_tokens": completions.usage.prompt_tokens,
        "completion_tokens": completions.usage.completion_tokens,
        "total_tokens": completions.usage.total_tokens
    }
    
    return all_candidates, token_usage, latency

def normalize_vql(vql):
    """
    标准化 VQL 以便进行比较
    去除多余的空格、换行等
    """
    vql = vql.replace("\n", " ").strip()
    # 统一多个空格为一个空格
    vql = " ".join(vql.split())
    # 移除 "Visualize" 前缀以便比较
    if vql.startswith("Visualize "):
        vql = vql[10:]
    return vql

def majority_vote(candidates):
    """
    对候选答案进行多数投票
    返回出现次数最多的答案及其投票详情
    """
    # 标准化所有候选答案
    normalized_candidates = [normalize_vql(c) for c in candidates]
    
    # 统计每个答案的出现次数
    vote_counts = Counter(normalized_candidates)
    
    # 找出出现次数最多的答案
    most_common = vote_counts.most_common(1)[0]
    winner = most_common[0]
    winner_count = most_common[1]
    
    # 计算一致性得分（最高票数/总票数）
    consistency_score = winner_count / len(candidates)
    
    # 构建投票详情
    vote_details = {}
    for candidate, count in vote_counts.items():
        vote_details[candidate] = {
            "count": count,
            "percentage": count / len(candidates)
        }
    
    return winner, consistency_score, vote_details

def get_embedding_from_file(vql:str, embedding_file:list):
    for v in embedding_file:
        if v['NLQ'].lower() == vql.lower():
            return v['Embedding']
    raise RuntimeError("No such NLQ in Embedding file: {}".format(vql))

def rag_by_nlq(nlq:str, k=10):
    document_all = pd.DataFrame(nlq_embedding_all)
    document_embeddings = document_all['Embedding'].to_list()

    try:
        question_embedding = get_embedding_from_file(nlq, nlq_test_embedding_all)
    except:
        # 如果没有预存的 embedding，这里需要实现 get_embedding 函数
        raise RuntimeError("No embedding found for NLQ: {}".format(nlq))
    
    # 计算问题向量与文档中每个句子向量的相似度
    similarities = [1 - cosine(question_embedding, doc_embedding) for doc_embedding in document_embeddings]

    # 选择top-k个最相似的句子
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_row = document_all.loc[top_k_indices.tolist()][['NLQ', "VQL","db_id"]]

    examples = []
    for index, row in top_k_row.iterrows():
            example = {
                    "NLQ":row['NLQ'],
                    "VQL":row['VQL'],
                    "schema":generate_schema(row['db_id'])
            }
            examples.append(example)

    examples.reverse()
    return examples

def prompt_maker(rag_list:list, db_id:str, nlq:str):
    prompt="""#### Given Natural Language Questions, Generate DVQs based on their correspoding Database Schemas.

"""
    for example in rag_list:
        prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{}"
### Data Visualization Query:
A: {}

""".format(example['schema'], example['NLQ'], example['VQL'])
        

    prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{}"
### Data Visualization Query:
A: Visualize """.format(generate_schema(db_id), nlq)
    
    return prompt


if __name__ == '__main__':

    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:

        data = pd.read_csv(data_file_path.format(mode, mode))
        data_new = []

        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f:
                data_new = json.load(f)
        else:
            os.makedirs(os.path.dirname(result_save_path.format(mode, mode)), exist_ok=True)
            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
        
        for index, d in tqdm(data.iterrows(), total=len(data), desc=f"Processing {mode} with Self-Consistency"):
            if index < len(data_new):
                continue
            nlq = d['nl_queries']
            target = d['VQL']
            db_id = d['db_id']
            record_name = d['record_name']

            examples = rag_by_nlq(nlq, 10)
            prompt = prompt_maker(examples, db_id, nlq)

            message = [
                {
                    "role":"system",
                    "content":"Please follow the syntax in the examples instead of SQL syntax."
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]

            while True:
                try:
                    # 使用 self-consistency: 生成多个候选答案
                    candidates, token_usage, latency = generate_reply_self_consistency(message, n=NUM_SAMPLES, temperature=TEMPERATURE)
                    
                    # 为每个候选答案添加 "Visualize" 前缀
                    candidates_with_prefix = ["Visualize " + c.replace("\n", " ") for c in candidates]
                    
                    # 进行多数投票
                    winner, consistency_score, vote_details = majority_vote(candidates_with_prefix)
                    
                    # 最终答案添加 "Visualize" 前缀（如果还没有的话）
                    if not winner.startswith("Visualize "):
                        final_answer = "Visualize " + winner
                    else:
                        final_answer = winner
                    
                    break
                except Exception as ex:
                    print(ex)
                    print("api error, wait for 3s...")
                    if "maximum context length" in str(ex):
                        examples = rag_by_nlq(nlq, 8)
                        prompt = prompt_maker(examples, db_id, nlq)
                        message = [
                            {
                                "role":"system",
                                "content":"Please follow the syntax in the examples instead of SQL syntax."
                            },
                            {
                                "role":"user",
                                "content":prompt
                            }
                        ]
                    time.sleep(3)

            
            print("=" * 80)
            print(f"Question: {nlq}")
            print(f"\nAll Candidates ({NUM_SAMPLES} samples):")
            for i, candidate in enumerate(candidates_with_prefix, 1):
                print(f"  {i}. {candidate}")
            print(f"\nFinal Answer (Consistency Score: {consistency_score:.2f}):")
            print(f"  {final_answer}")
            print(f"\nTarget:")
            print(f"  {target}")
            print("=" * 80)
            print()

            example = {
                "record_name": record_name,
                "db_id": db_id,
                "target": target,
                "nlq": nlq,
                "predict_self_consistency": final_answer,
                "consistency_score": consistency_score,
                "all_candidates": candidates_with_prefix,
                "vote_details": vote_details,
                "num_samples": NUM_SAMPLES,
                "temperature": TEMPERATURE,
                "token_usage": token_usage,
                "latency_seconds": latency
            }

            data_new.append(example)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4, ensure_ascii=False)
            
