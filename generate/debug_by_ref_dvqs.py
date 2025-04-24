import openai
import json
import pandas as pd
from tqdm import tqdm
import time
import re
import os

openai.api_base = "https://openkey.cloud/v1"

# 替换为你的OpenAI API密钥
openai.api_key = "sk-u4QlUZW8nLtaiqyv40338f916f5a4d6c81B42aB83b6fE567"

DATASET_SCHEMA = '../data/tables.json'
db_ann_file_path = "../data/database_anno.json"
data_path = "../data/{}/{}_result_dvq_rag.json"
result_save_path = "../data/{}/{}_result_debugged_by_ref_dvqs.json"


ask1 = """### Database Schemas:
# Table countries, columns = [ * , COUNTRY_ID , COUNTRY_NAME , REGION_ID ]
# Table departments, columns = [ * , DEPARTMENT_ID , DEPARTMENT_NAME , MANAGER_ID , LOCATION_ID ]
# Table employees, columns = [ * , EMPLOYEE_ID , FIRST_NAME , LAST_NAME , EMAIL , PHONE_NUMBER , HIRE_DATE , JOB_ID , SALARY , COMMISSION_PCT , MANAGER_ID , DEPARTMENT_ID ]
# Table job_history, columns = [ * , EMPLOYEE_ID , START_DATE , END_DATE , JOB_ID , DEPARTMENT_ID ]
# Table jobs, columns = [ * , JOB_ID , JOB_TITLE , MIN_SALARY , MAX_SALARY ]
# Table locations, columns = [ * , LOCATION_ID , STREET_ADDRESS , POSTAL_CODE , CITY , STATE_PROVINCE , COUNTRY_ID ]
# Table regions, columns = [ * , REGION_ID , REGION_NAME ]
# Foreign_keys = [ countries.REGION_ID = regions.REGION_ID , employees.JOB_ID = jobs.JOB_ID , employees.DEPARTMENT_ID = departments.DEPARTMENT_ID , job_history.JOB_ID = jobs.JOB_ID , job_history.DEPARTMENT_ID = departments.DEPARTMENT_ID , job_history.EMPLOYEE_ID = employees.EMPLOYEE_ID , locations.COUNTRY_ID = countries.COUNTRY_ID ]

### Reference DVQs:
1 - Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID
2 - Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID ORDER BY JOB_ID ASC
3 - Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID ORDER BY COUNT(JOB_ID) ASC
4 - Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID ORDER BY JOB_ID DESC
5 - Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID ORDER BY COUNT(JOB_ID) DESC

#### Given the Reference DVQs, please modify the Original DVQ to mimic the style of the Reference DVQs.
#### NOTE: Do not Replace the column name in Original DVQ, especially those in the ORDER clause!
### Original DVQ:
# Visualize BAR SELECT JOB_ID , COUNT(DISTINCT JOB_ID) FROM employees WHERE DEPARTMENT_ID = (SELECT DEPARTMENT_ID FROM departments WHERE DEPARTMENT_NAME = Finance)
A: Let’s think step by step! """

answer1 = """### Modified DVQ:
# Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY JOB_ID"""

message = [
    {
        "role":"system",
        "content":"""The Reference Data Visualization Queries(DVQs) all comply with the syntax of DVQ. Please follow the syntax of the referenced DVQ to modify the Original DVQ."""
    },
    {
        "role":"user",
        "content":ask1
    },
    {
        "role":"assistant",
        "content":answer1
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


spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)

def get_dvqs(dvqs:list):
    prompt = "### Reference DVQs:\n"
    for i, s in enumerate(dvqs):
        s = s.replace(" ) ", ") ")
        prompt += f"{i+1} - " + s + "\n"

    return prompt


def prompt_maker(db_id:str, nlq:str, dvq:str, dvqs_ref:list):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]

    dvqs = get_dvqs(dvqs_ref)

    # dvq = dvq.replace("(", " ").replace(")", " ").strip()
    while "  " in dvq:
        dvq = dvq.replace("  ", " ")

    db = generate_schema(db_id)
    
    prompt = db + "\n\n" + dvqs + "\n" + """#### Given the Reference DVQs, please modify the Original DVQ to mimic the style of the Reference DVQs.
#### NOTE: Do not Modify the column name in Original DVQ. Especially do not Modify the column names in the ORDER clause!
### Original DVQ:
# {}
A: Let’s think step by step! """.format(dvq)
    return prompt

def generate_reply(messages, n=1):
    # print("generate...")
    completions = openai.ChatCompletion.create(
        # model="gpt-4-0125-preview",
        model="gpt-3.5-turbo-0125",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        frequency_penalty=-0.5,
        presence_penalty=-0.5
    )
    # print(completions)
    reply = completions.choices[0].message.content.replace("\n", " ")
    while "  " in reply:
       reply = reply.replace("  ", " ")

    return reply

def remove_as(dvq:str):
    tokens = dvq.replace(",", " , ").replace(".", " . ").replace("(", " ( ").replace(")", " ) ").split()
    tokens_new = []
    ref = {}
    for id, token in enumerate(tokens):
        if token.lower() == "as" and "T" not in tokens[id + 1]:
            ref[tokens[id + 1]] = tokens[id - 1]
            continue
        if id > 0 and tokens[id - 1].lower() == "as" and "T" not in token:
            continue

        tokens_new.append(token)

    dvq_new = " ".join(tokens_new).replace(" . ", ".").replace(" ( ", "(").replace(" ) ", ") ")

    for name, c in ref.items():
        # print("{} -> {}".format(name, c))
        dvq_new = dvq_new.replace(name, c)

    while "  " in dvq_new:
        dvq_new = dvq_new.replace("  ", " ")

    return dvq_new

def remove_having(text:str):
    structure_tokens1 = ['visualize', 'select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'having', 'bin']
    text = text.split()
    text_new = []
    flag = ""
    for token in text:
        if token.lower() in structure_tokens1:
            flag = token.lower()
        if flag == "having":
            continue
        text_new.append(token)
    text = text_new
    return " ".join(text)

if __name__ == '__main__':
    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
        data_new = []
        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f: 
                data_new = json.load(f)
        with open(data_path.format(mode, mode), 'r') as f:
            data = json.load(f)

        for index, example in tqdm(enumerate(data), total=len(data)):
            if index<len(data_new):
               continue
            nlq = example['nlq']
            db_id = example['db_id']
            target = example['target']
            record_name = example['record_name']
            dvq = remove_having(remove_as(example['predict_rag_nlq']))
            rag_dvqs = example['rag_dvqs']

            while "  " in dvq:
                dvq = dvq.replace("  ", " ")

            prompt = prompt_maker(db_id, nlq, dvq, rag_dvqs)

            messages = message.copy()

            messages.append(
                {
                    "role":"user",
                    "content":prompt
                }
            )

            times = 0
            while True:
                times += 1
                if times == 3:
                  reply = generate_reply(messages, 1).strip()
                  break
                try:
                    reply = generate_reply(messages, 1).strip()
                    print(reply)
                    if "```" in reply:
                        reply = reply.split("```", 2)[1]
                    reply = "Visualize " + reply.strip()
                    reply = "Visualize " + reply.rsplit("Visualize ", 1)[1]
                    break
                except Exception as ex:
                    # greply = generate_reply(messages, 1)
                    print(reply)
                    print(ex)
                    print("api error, wait for 3s...")
                    time.sleep(3)
            dvq_debugged = remove_as(reply)

            example_new = example.copy()
            example_new['predict_debugged_ref_dvqs'] = dvq_debugged
            data_new.append(example_new)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
            