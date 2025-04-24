import openai
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time





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

DATASET_SCHEMA = './data/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)
# print(find_fields_MYSQL_like("pets_1"))
# print(find_foreign_keys_MYSQL_like("pets_1"))
# print(generate_schema("network_1"))



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

def prompt_maker(rag_list:pd.DataFrame):

    prompt="""#### Given Natural Language Questions, Generate DVQs based on their correspoding Database Schemas.

"""
    for index , example in tqdm(rag_list.iterrows(), total=len(rag_list)):
        # print(example)
        prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# “{}”
### Data Visualization Query:
A: {}

""".format(generate_schema(example['db_id']), example['nl_queries'], example['VQL'])
    return prompt


if __name__ == '__main__':

    data_file_path = "./data/{}/{}.csv"
    result_save_path = "./store.txt"

    data = pd.read_csv(data_file_path.format("train_nvBench", "train_nvBench"))
    data_new = ""

    examples = data[['nl_queries', "VQL", "db_id"]]
    print(examples)

    prompt = prompt_maker(examples)



    with open(result_save_path, 'w', encoding='utf-8') as f:
        f.write(prompt)