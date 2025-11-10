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
from config import OPENAI_BASE_URL, OPENAI_API_KEY
import random

client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

embedding_file_path = "../nvBench-Rob/nlq_embedding.pkl"
embedding_test_file_path = "../nvBench-Rob/nlq_embedding_test.pkl"
data_file_path = "../nvBench-Rob/{}/result_multi-turn/rebuttal/{}_result_multi-turn_gpt-3.5-turbo_0.json"
result_save_path = "../nvBench-Rob/{}/result_rebuttal/{}_confidence_2.json"

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

DATASET_SCHEMA = '../nvBench-Rob/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4o-mini",
        messages=messages,
        n = n,
        stream = False,
        temperature=1.4,
        frequency_penalty=0, # 避免重复性(-2.0 ~ 2.0)
        presence_penalty=0   # 生成新主题(-2.0 ~ 2.0)
    )
    # print(completions)
    output = completions.choices[0].message.content
    if flag == "multiple":
        output = []
        for i in range(n):
            o = completions.choices[i].message.content
            # print(vql)
            output.append(o)
    else:
        return completions.choices[0].message.content
    return output


def prompt_maker(db_id:str, nlq:str, dvqs:list):

    prompt="""#### Given Natural Language Questions, Database Schemas, and multiple Generate DVQs. Please select one of them to answer the question.

{}

### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# “{}”
### Data Visualization Query (DVQ):""".format(generate_schema(db_id), nlq)

    for index, dvq in enumerate(dvqs):
        prompt += """
{} : {}""".format(index+1, dvq)

    prompt += "NOTE: ONLY return a numeric index of the selected DVQ. Do not return any other text.\n\nindex: "
    return prompt


if __name__ == '__main__':
    

    # for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
    for mode in ['dev_nlq_schema']:

        data = json.load(open(data_file_path.format(mode, mode), 'r'))
        # 生成100个不同的随机整数
        random_numbers = random.sample(range(len(data)), 100)
        data_new = []

        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f:
                data_new = json.load(f)
        else:
            os.makedirs(os.path.dirname(result_save_path.format(mode, mode)), exist_ok=True)
            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
        
        for index, d in tqdm(enumerate(data), total=len(data), desc=f"Processing {mode}"):
            if index not in random_numbers:
                continue
            nlq = d['nlq']
            target = d['target']
            db_id = d['db_id']
            record_name = d['record_name']
            dvqs = d['predict_dvq_set']

            prompt = prompt_maker(db_id, nlq, dvqs)

            message = [
                {
                    "role":"system",
                    "content":""
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]

            while True:
                try:
                    reply = generate_reply(message, 10, "multiple")
                    break
                except Exception as ex:
                    print(ex)
                    print("api error, wait for 3s...")
                    if "maximum context length" in str(ex):
                        prompt = prompt_maker(db_id, nlq)

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
                "predict_dvq_set":dvqs,
                "predict_rag_nlq":reply
            }

            data_new.append(example)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
            
