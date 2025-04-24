from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time

client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

embedding_file_path = "./nvBench-Rob/nlq_embedding.pkl"
embedding_test_file_path = "./nvBench-Rob/nlq_embedding_test.pkl"
data_file_path = "./nvBench-Rob/{}/{}.csv"
result_save_path = "./nvBench-Rob/{}/result/{}_result_nlq_rag.json"

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
  # output = output[:-1] + ']\n'
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
# print(find_fields_MYSQL_like("pets_1"))
# print(find_foreign_keys_MYSQL_like("pets_1"))
# print(generate_schema("network_1"))

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        frequency_penalty=-0.5, # 避免重复性(-2.0 ~ 2.0)
        presence_penalty=-0.5   # 生成新主题(-2.0 ~ 2.0)
    )
    # print(completions)
    mes = completions.choices[0].message.content
    if flag == "vql":
        all_p_vqls = []
        for i in range(n):
            vql = completions.choices[i].message.content
            # print(vql)
            all_p_vqls.append(vql)
    else:
        return completions.choices[0].message.content
    return all_p_vqls

def get_embedding(text, model="text-embedding-3-large"):
    # model = "text-embedding-3-large"
    response = openai.Embedding.create(input=text, engine=model)
    # 确保我们正确地处理响应数据
    embedding = response['data'][0]['embedding']
    # 将嵌入向量转换为numpy数组
    return np.array(embedding)

def get_embedding_from_file(vql:str, embedding_file:list):
    # print(vql)
    for v in embedding_file:
        # print(v)
        if v['NLQ'].lower() == vql.lower():
            return v['Embedding']
        
    raise RuntimeError("No such NLQ in Embedding file: {}".format(vql))


def rag_by_nlq(nlq:str, k=10):


    document_all = pd.DataFrame(nlq_embedding_all)
    document_embeddings = document_all['Embedding'].to_list()

    # question_embedding = get_embedding(nlq)
    try:
        question_embedding = get_embedding_from_file(nlq, nlq_test_embedding_all)
    except:
        question_embedding = get_embedding(nlq)
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
# “{}”
### Data Visualization Query:
A: {}

""".format(example['schema'], example['NLQ'], example['VQL'])
        

    prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# “{}”
### Data Visualization Query:
A: Visualize """.format(generate_schema(db_id), nlq)
    
    return prompt


if __name__ == '__main__':

    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
    # for mode in ['dev_nvBench']:

        data = pd.read_csv(data_file_path.format(mode, mode))
        data_new = []

        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f:
                data_new = json.load(f)
        
        for index, d in tqdm(data.iterrows(), total=len(data), desc=f"Processing {mode}"):
            if index < len(data_new):
                continue
            nlq = d['nl_queries']
            target = d['VQL']
            db_id = d['db_id']
            record_name = d['record_name']

            examples = rag_by_nlq(nlq, 10)
            prompt = prompt_maker(examples, db_id, nlq)

            # print(prompt)
            # exit()
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

            # times = 0
            while True:
                # times += 1
                # if times == 3:
                #     reply = generate_reply(message, 1, "nlq").replace("IS NOT NULL", "!= \"null\"").replace("\n", " ").replace("<>", "!=")
                #     break
                try:
                    reply = "Visualize " + generate_reply(message, 1, "nlq").replace("\n", " ")
                    break
                except Exception as ex:
                    print(ex)
                    print("api error, wait for 3s...")
                    if "maximum context length" in str(ex):
                        examples = rag_by_nlq(nlq, 8)
                        prompt = prompt_maker(examples, db_id, nlq)

                        # print(prompt)
                        # exit()
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

            
            print("Predict:\n{}\n".format(reply))
            print("Target:\n{}\n".format(target))

            example = {
                "record_name":record_name,
                "db_id":db_id,
                "target":target,
                "nlq":nlq,
                "predict_rag_nlq":reply
            }

            data_new.append(example)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)