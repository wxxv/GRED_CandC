from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import time
import os


client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

db_ann_file_path = "../nvBench-Rob/database_anno.json"
data_path = "../nvBench-Rob/{}/{}_result_debugged_by_ref_dvqs.json"
result_save_path = "../nvBench-Rob/{}/{}_result_debugged_by_db_ann_xwx.json"


ask1 = """### Database Schemas:
# Table countries, columns = [ * , COUNTRY_ID , COUNTRY_NAME , REGION_ID ]
# Table departments, columns = [ * , DEPARTMENT_ID , DEPARTMENT_NAME , MANAGER_ID , LOCATION_ID ]
# Table employees, columns = [ * , EMPLOYEE_ID , FIRST_NAME , LAST_NAME , EMAIL , PHONE_NUMBER , HIRE_DATE , JOB_ID , SALARY , COMMISSION_PCT , MANAGER_ID , DEPARTMENT_ID ]
# Table job_history, columns = [ * , EMPLOYEE_ID , START_DATE , END_DATE , JOB_ID , DEPARTMENT_ID ]
# Table jobs, columns = [ * , JOB_ID , JOB_TITLE , MIN_SALARY , MAX_SALARY ]
# Table locations, columns = [ * , LOCATION_ID , STREET_ADDRESS , POSTAL_CODE , CITY , STATE_PROVINCE , COUNTRY_ID ]
# Table regions, columns = [ * , REGION_ID , REGION_NAME ]
# Foreign_keys = [ countries.REGION_ID = regions.REGION_ID , employees.JOB_ID = jobs.JOB_ID , employees.DEPARTMENT_ID = departments.DEPARTMENT_ID , job_history.JOB_ID = jobs.JOB_ID , job_history.DEPARTMENT_ID = departments.DEPARTMENT_ID , job_history.EMPLOYEE_ID = employees.EMPLOYEE_ID , locations.COUNTRY_ID = countries.COUNTRY_ID ]

### Natural Language Annotations of Database Schemas:
Table countries:
- Contains information about different countries.
- Columns:
  - COUNTRY_ID: Unique identifier for each country.
  - COUNTRY_NAME: Name of the country.
  - REGION_ID: Foreign key referencing the id of the region in the regions table.

Table departments:
- Stores data related to different departments within an organization.
- Columns:
  - DEPARTMENT_ID: Unique identifier for each department.
  - DEPARTMENT_NAME: Name of the department.
  - MANAGER_ID: Identifier of the manager of the department.
  - LOCATION_ID: Identifier of the location where the department is situated.

Table employees:
- Contains details about employees working in the organization.
- Columns:
  - EMPLOYEE_ID: Unique identifier for each employee.
  - FIRST_NAME: First name of the employee.
  - LAST_NAME: Last name of the employee.
  - EMAIL: Email address of the employee.
  - PHONE_NUMBER: Phone number of the employee.
  - HIRE_DATE: Date when the employee was hired.
  - JOB_ID: Identifier of the job role of the employee.
  - SALARY: Salary of the employee.
  - COMMISSION_PCT: Commission percentage for the employee.
  - MANAGER_ID: Identifier of the manager of the employee.
  - DEPARTMENT_ID: Identifier of the department to which the employee belongs.

Table job_history:
- Stores historical data of job changes for employees.
- Columns:
  - EMPLOYEE_ID: Identifier of the employee.
  - START_DATE: Start date of the job role.
  - END_DATE: End date of the job role.
  - JOB_ID: Identifier of the job role.
  - DEPARTMENT_ID: Identifier of the department during the job role.

Table jobs:
- Contains information about different job roles.
- Columns:
  - JOB_ID: Unique identifier for each job role.
  - JOB_TITLE: Title of the job role.
  - MIN_SALARY: Minimum salary for the job role.
  - MAX_SALARY: Maximum salary for the job role.

Table locations:
- Stores details about different locations.
- Columns:
  - LOCATION_ID: Unique identifier for each location.
  - STREET_ADDRESS: Street address of the location.
  - POSTAL_CODE: Postal code of the location.
  - CITY: City where the location is situated.
  - STATE_PROVINCE: State or province of the location.
  - COUNTRY_ID: Foreign key referencing the id of the country in the countries table.

Table regions:
- Contains information about different regions.
- Columns:
  - REGION_ID: Unique identifier for each region.
  - REGION_NAME: Name of the region.

Foreign Keys:
- countries.REGION_ID references regions.REGION_ID, establishing a relationship between countries and regions.
- employees.JOB_ID references jobs.JOB_ID, linking employees to specific job roles.
- employees.DEPARTMENT_ID references departments.DEPARTMENT_ID, connecting employees to their respective departments.
- job_history.JOB_ID references jobs.JOB_ID, linking job history to specific job roles.
- job_history.DEPARTMENT_ID references departments.DEPARTMENT_ID, connecting job history to departments.
- job_history.EMPLOYEE_ID references employees.EMPLOYEE_ID, establishing a relationship between job history and employees.
- locations.COUNTRY_ID references countries.COUNTRY_ID, linking locations to specific countries.

#### Given Database Schemas and their corresponding Natural Language Annotations, Please replace the column names in the Data Visualization Query(DVQ, a new Programming Language abstracted from Vega-Zero) that do not exist in the database.
#### NOTE: Don’t replace column names in Original DVQ that already exist in the database schemas, especially column names in GROUP BY Clause!
### Original DVQ:
# Visualize BAR SELECT jobid , COUNT(jobid) FROM employees AS T1 JOIN departments AS T2 ON T1.dept_id = T2.dept_id WHERE T2.dept_name = 'Finance' GROUP BY FIRST_NAME
A: Let’s think step by step! """

answer1="""### Revised DVQ:
# Visualize BAR SELECT JOB_ID , COUNT(JOB_ID) FROM employees AS T1 JOIN departments AS T2 ON T1.DEPARTMENT_ID = T2.DEPARTMENT_ID WHERE T2.DEPARTMENT_NAME = 'Finance' GROUP BY FIRST_NAME"""

message = [
   {
      "role":"system",
      "content":"""#### NOTE: Don’t replace column names in Original DVQ that already exist in the database schemas, especially column names in GROUP BY Clause!"""
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

DATASET_SCHEMA = '../nvBench-Rob/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)


def prompt_maker(db_id:str, dvq:str):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]

    db = generate_schema(db_id)
    
    prompt = db + "\n\n### Natural Language Annotations of Database Schemas:\n" + db_ann + "\n\n" + """#### Given Database Schemas and their corresponding Natural Language Annotations, Please replace the column names in the Data Visualization Query(DVQ, a new Programming Language abstracted from Vega-Zero) that do not exist in the database.
#### NOTE: Don’t replace column names in Original DVQ that already exist in the database schemas, especially column names in GROUP BY Clause!
### Original DVQ:
# {}
A: Let’s think step by step! """.format(dvq)
    return prompt

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        frequency_penalty=-0.5,
        presence_penalty=-0.5
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
            dvq = example['predict_debugged_ref_dvqs']
            rag_dvqs = example['rag_dvqs']

            while "  " in dvq:
                dvq = dvq.replace("  ", " ")

            prompt = prompt_maker(db_id, dvq)

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
                    reply = "Visualize " + generate_reply(messages, 1, "nlq").replace("\n", " ")
                    # print(reply)
                    if "```" in reply:
                        reply = reply.split("```", 2)[1].replace("\n", "")
                    reply = "Visualize " + reply.rsplit("Visualize ", 1)[1].strip()
                    break
                except Exception as ex:
                  #  print(reply)
                   print(ex)
                   print("api error, wait for 3s...")
                   time.sleep(3)
            dvq_debugged_db_ann = reply

            rag_dvqs_new = []
            
            example_new = example.copy()
            example_new['predict_debugged_db_ann'] = dvq_debugged_db_ann
            data_new.append(example_new)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4)
            