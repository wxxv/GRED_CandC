from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import time
import os
from prompt import ask_gen_candidate_set, answer_gen_candidate_set


client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

db_ann_file_path = "./nvBench-Rob/database_anno.json"
data_path = "./nvBench-Rob/{}/{}_result_debugged_by_db_ann.json"
result_save_path = "./nvBench-Rob/{}/result_multi-turn/{}_result_gen_candidate_set_gpt4o.json"
DATASET_SCHEMA = './nvBench-Rob/tables.json'

message = [
   {
      "role":"system",
      "content":"""You are a helpful data visualization expert that generates possible DVQs(a new Programming Language abstracted from Vega-Zero)."""
   }
#    {
#       "role":"user",
#       "content":ask_gen_candidate_set
#    },
#    {
#       "role":"assistant",
#       "content":answer_gen_candidate_set
#    }
]


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
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)


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


def generate_schema(db_name:str):
  schema = "### Database Schemas:\n" + find_fields_MYSQL_like(db_name) + find_foreign_keys_MYSQL_like(db_name)
  return schema


def prompt_maker(db_id:str, nlq:str, rag_dvqs:list, final_dvq:str):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]

    db = generate_schema(db_id)
    
    prompt = db + "\n\n" + """### Natural Language Question (NLQ): 
# {}

#### Given a Database Schema, Natural Language Question, and Original Data Visualization Query(DVQ, a new Programming Language abstracted from Vega-Zero), please generate a set of candidate DVQs with their probabilities that you think are correct. 
# Step-by-step Instructions:
# 1. Copy the Original DVQ as the first candidate without any modification.
# 2. Then for each of other candidate DVQs, only modify a content part of the Original DVQ, not structure or keywords.
# 3. Generate the probability that you think each of the candidate DVQs is correct.
# 4. Return format - JSON dictionary: {{candidate_dvq: probability}}
#### NOTE: Remember use '\"' to escape the double quotes in the candidate DVQs. Ensure the sum of probabilities is 1. Ensure the first candidate is the original DVQ.
### Original DVQ: 
# {}
A: Let's think step by step!""".format(nlq, final_dvq)

    return prompt

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4o-mini",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

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

def get_dvqs(dvqs:list):
    prompt = ""
    for i, s in enumerate(dvqs[:2]):
        s = s.replace(" ) ", ") ")
        prompt += f"{i+1} - " + s + "\n"
    return prompt


if __name__ == '__main__':
    # for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
    for mode in ['dev_nlq_schema']:
        data_new = []
        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f: 
                data_new = json.load(f)
        else:
            os.makedirs(os.path.dirname(result_save_path.format(mode, mode)), exist_ok=True)
            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)

        with open(data_path.format(mode, mode), 'r') as f:
            data = json.load(f)

        for index, example in tqdm(enumerate(data), total=len(data), desc=f"Processing {mode}"):
            if index<len(data_new):
                continue
            nlq = example['nlq']
            db_id = example['db_id']
            target = example['target']
            record_name = example['record_name']
            dvq = example['predict_debugged_ref_dvqs']
            rag_dvqs = example['rag_dvqs']
            predict_debugged_db_ann = example['predict_debugged_db_ann']
            ref_dvqs = get_dvqs(rag_dvqs)

            if "  " in predict_debugged_db_ann:
                predict_debugged_db_ann = predict_debugged_db_ann.replace("  ", " ")
            
            if True:
                prompt = prompt_maker(db_id, nlq, ref_dvqs, predict_debugged_db_ann)

                # print(prompt)
                # exit()
                messages = message.copy()
                messages.append(
                    {
                        "role":"user",
                        "content":prompt
                    }
                )
                
                err_count = 0
                while True:
                    try:
                        reply = generate_reply(messages, 1, "nlq")
                        if "```" in reply:
                            reply = reply.split("```", 2)[1].split("json", 1)[-1]
                            reply = json.loads(reply)
                        else:
                            reply = json.loads(reply)
                        if err_count > 0:
                            err_count = 0
                        break
                    except Exception as ex:
                        print(target)
                        print(predict_debugged_db_ann)
                        print(reply)
                        print(ex)
                        print("api error, wait for 3s...")
                        time.sleep(3)
                        err_count += 1
                        # if err_count > 3:
                        # reply = json.dumps({f"{final_dvq}": 1.0})
                        # reply = json.loads(str(reply))
                        exit()
                        break
                
                # print(reply)
                rag_dvqs_new = []
                
                example_new = example.copy()
                example_new['predict_dvq_set'] = reply
                data_new.append(example_new)

                with open(result_save_path.format(mode, mode), 'w') as f:
                    json.dump(data_new, f, indent=4)

                # if index == 19:
                #     exit()
