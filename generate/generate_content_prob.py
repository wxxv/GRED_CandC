from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import time
import os
from prompt import ask_gen_prob, answer_gen_prob


client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

db_ann_file_path = "../nvBench-Rob/database_anno.json"
data_path = "../nvBench-Rob/{}/result_multi-turn/{}_result_gen_candidate_set_gpt4o.json"
result_save_path = "../nvBench-Rob/{}/result_multi-turn/{}_result_gen_candidate_set_with_content_prob_gpt4o.json"
DATASET_SCHEMA = '../nvBench-Rob/tables.json'

message = [
   {
      "role":"system",
      "content":"""You are an expert in programming language analysis."""
   },
   {
      "role":"user",
      "content":ask_gen_prob
   },
   {
      "role":"assistant",
      "content":answer_gen_prob
   }
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


def prompt_maker(db_id:str, nlq:str, predict_dvq_set:str):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]

    db = generate_schema('browser_web_robust')
    
#     prompt = """#### Given the candidate set of Data Visualization Queries (DVQs, a new Programming Language abstracted from Vega-Zero) with their corresponding probabilities, please compute the probability mass function (PMF) of the contents under each DVQ keyword (e.g., VISUALIZE, SELECT, JOIN, WHERE) by treating the contents as discrete random variables.
# # Step-by-step Instructions:
# # 1. Content Identification per Keyword:
# #   - For each DVQ keyword that appears in DVQs, list all unique content variants found in DVQs.
# #   - For each keyword, if a content appears in multiple DVQs, sum the probabilities of all DVQs in which it appears.
# #   - If a keyword is not present in given DVQs, treat its content as "None" with the corresponding probability sum of those DVQs.
# # 2. Output Format:
# #   - Return the result as a JSON-formatted nested dictionary:
# #     - Outer keys: DVQ keywords (excluding ones that are completely missing across all DVQs)
# #     - Inner keys: Content strings corresponding to each keyword
# #     - Inner values: Their associated probabilities (rounded to two decimal places)
# #### Note: Verify that the sum of probabilities of the contents for each keyword equals 1. When indicating that a field is not empty, you should also use the form "!= \\"null\\"", use the double quotes instead of the single quotes to indicate the string.
# ### Candidate Data Visualization Query (DVQs) with their probabilities: 
# {}
# A: Let's think step by step!""".format(nlq, predict_dvq_set)
    prompt = """### Given a set of candidate DVQs and their corresponding probabilities, please extract the contents associated with each DVQ keyword (e.g., VISUALIZE, SELECT, JOIN, WHERE, GROUP BY, etc.) from the DVQs and compute the probability mass function (PMF) for each keyword's content.

### The calculation of the PMF: 
# For each DVQ keyword that appears in candidate DVQs:
# 1 - Extract the content associated with the keyword from each candidate DVQ.
# 2 - If candidate DVQs do not contain the keyword, record the content as "None".
# 3 - Group identical contents together under the same keyword.
# 4 - Sum the probabilities of all candidates that contain each unique content.
#### Note: 
# 1. Ensure that the sum of probabilities of the contents for each keyword equals 1.0. Use double quotes for all strings in the JSON. 
# 2. Return the result in JSON format: each key is a DVQ keyword, and the value is a dictionary of contents and their normalized PMF values. Like this:
# {{
#     "Visualize": {{
#         "BAR": 1.00
#     }},
#     "SELECT": {{
#         "name , identification": 1.00,
#     }},
# }}
### Here is the candidate DVQs with their probabilities: 
{}
A: Let's think step by step!""".format(predict_dvq_set)
    return prompt

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini",
        messages=messages,
        n = n,
        stream = False,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
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
    prompt = "{\n"
    for i, (k, v) in enumerate(dvqs.items()):
        k = k.replace(" ) ", ") ")
        prompt += k + " : " + str(v) + "\n"
    prompt += "}"
    return prompt


if __name__ == '__main__':
    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
    # for mode in ['dev_schema']:
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
            final_dvq = example['predict_debugged_db_ann']
            predict_dvq_set = example['predict_dvq_set']
            possible_dvqs = get_dvqs(predict_dvq_set)

            if "  " in final_dvq:
                final_dvq = final_dvq.replace("  ", " ")
            
            if True:
                prompt = prompt_maker(db_id, nlq, possible_dvqs)

                # print(prompt)
                # exit()
                messages = message.copy()
                messages.append(
                    {
                        "role":"user",
                        "content":prompt
                    }
                )
                
                while True:
                    try:
                        reply = generate_reply(messages, 1, "nlq")
                        if "```" in reply:
                            content_prob = reply.split("```", 2)[1].split("json", 1)[-1]
                            content_prob = json.loads(content_prob)
                            # print(content_prob)
                            break
                        else:
                            content_prob = json.loads(reply)
                        # print(content_prob)
                        break

                    except Exception as ex:
                        print(content_prob)
                        print(ex)
                        print("api error, wait for 3s...")
                        time.sleep(3)
                        exit()
                        break
                
                # print(content_prob)
                # exit()
                example_new = example.copy()
                example_new['content_prob'] = content_prob
                data_new.append(example_new)
                with open(result_save_path.format(mode, mode), 'w') as f:
                    json.dump(data_new, f, indent=4)

