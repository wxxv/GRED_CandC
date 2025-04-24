import openai
import json
import pandas as pd
from tqdm import tqdm
import time

openai.api_base = "https://openkey.cloud/v1"

# 替换为你的OpenAI API密钥
openai.api_key = "sk-u4QlUZW8nLtaiqyv40338f916f5a4d6c81B42aB83b6fE567"


ask_schema = """#### Please generate detailed natural language annotations to the following database schemas.
### Database Schemas:
# Table countries, columns = [*,COUNTRYID,COUNTRYNAME,REGIONID]
# Table departments, columns = [*,Dept_ID,Dept_NAME,Manager_ID,Lpcation_ID]
# Table employees, columns = [*,employee_id,Fname,Lname,Email_address,phone_number,date_of_hire,JOB_ID,wage,COMMISSION_PCT,Manager_ID,Dept_ID]
# Table job_history, columns = [*,employee_id,START_DATE,END_DATE,JOB_ID,Dept_ID]
# Table jobs, columns = [*,JOB_ID,JOB_TITLE,minimum_salary,maximum_salary]
# Table locations, columns = [*,Lpcation_ID,STREET_ADDRESS,POSTAL_CODE,CITY,STATE_PROVINCE,COUNTRYID]
# Table regions, columns = [*,REGIONID,REGIONNAME]
# Foreign_keys = [countries.REGIONID = regions.REGIONID,employees.JOB_ID = jobs.JOB_ID,employees.Dept_ID = departments.Dept_ID,job_history.JOB_ID = jobs.JOB_ID,job_history.Dept_ID = departments.Dept_ID,job_history.employee_id = employees.employee_id,locations.COUNTRYID = countries.COUNTRYID]

### Natural Language Annotations:
A:
Table countries:
- This table contains information about different countries.
- Columns:
  - COUNTRYID: Unique identifier for each country.
  - COUNTRYNAME: Name of the country.
  - REGIONID: Foreign key referencing the id of the region in the regions table.

Table departments:
- Stores data related to different departments within an organization.
- Columns:
  - Dept_ID: Unique identifier for each department.
  - Dept_NAME: Name of the department.
  - Manager_ID: Identifier of the manager of the department.
  - Location_ID: Identifier of the location where the department is situated.

Table employees:
- Contains details about employees working in the organization.
- Columns:
  - employee_id: Unique identifier for each employee.
  - Fname: First name of the employee.
  - Lname: Last name of the employee.
  - Email_address: Email address of the employee.
  - phone_number: Phone number of the employee.
  - date_of_hire: Date when the employee was hired.
  - JOB_ID: Identifier of the job role of the employee.
  - wage: Salary of the employee.
  - COMMISSION_PCT: Commission percentage for the employee.
  - Manager_ID: Identifier of the manager of the employee.
  - Dept_ID: Identifier of the department to which the employee belongs.

Table job_history:
- Stores historical data of job changes for employees.
- Columns:
  - employee_id: Identifier of the employee.
  - START_DATE: Start date of the job role.
  - END_DATE: End date of the job role.
  - JOB_ID: Identifier of the job role.
  - Dept_ID: Identifier of the department during the job role.

Table jobs:
- Contains information about different job roles.
- Columns:
  - JOB_ID: Unique identifier for each job role.
  - JOB_TITLE: Title of the job role.
  - minimum_salary: Minimum salary for the job role.
  - maximum_salary: Maximum salary for the job role.

Table locations:
- Stores details about different locations.
- Columns:
  - Location_ID: Unique identifier for each location.
  - STREET_ADDRESS: Street address of the location.
  - POSTAL_CODE: Postal code of the location.
  - CITY: City where the location is situated.
  - STATE_PROVINCE: State or province of the location.
  - COUNTRYID: Foreign key referencing the id of the country in the countries table.

Table regions:
- Contains information about different regions.
- Columns:
  - REGIONID: Unique identifier for each region.
  - REGIONNAME: Name of the region.

Foreign Keys:
- countries.REGIONID references regions.REGIONID, establishing a relationship between countries and regions.
- employees.JOB_ID references jobs.JOB_ID, linking employees to specific job roles.
- employees.Dept_ID references departments.Dept_ID, connecting employees to their respective departments.
- job_history.JOB_ID references jobs.JOB_ID, linking job history to specific job roles.
- job_history.Dept_ID references departments.Dept_ID, connecting job history to departments.
- job_history.employee_id references employees.employee_id, establishing a relationship between job history and employees.
- locations.COUNTRYID references countries.COUNTRYID, linking locations to specific countries.

{}

### Natural Language Annotations:
A:"""

def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = [] # [[database, table, '*', 'text'] * num_table, [database, table, col_name, col_type] * num_col]
    f_keys = [] # [[database, table, col_name] * num_pri]
    p_keys = [] # [[database, table1_f, table2_p, f_key, p_key] * num_foreign]
    # for each database
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        # for each col in tables
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
    output += "# Table " + name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"] + ','
    output = output[:-1] + ']\n'
  output = output[:-2] + ']\n'
  return output


def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "# Foreign_keys = ["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
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

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
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

DATASET_SCHEMA = '../data/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)

with open(DATASET_SCHEMA, 'r') as f:
   data = json.load(f)

db_ann = {}
for example in tqdm(data):
    db_id = example['db_id']
    if db_id in db_ann:
       continue
    schema = generate_schema(db_id)

    prompt = ask_schema.format(schema)

    message = [
        {
            "role":"system",
            "content":"""You are a data mining engineer with ten years of experience in data visualization."""
        },
        {
            "role":"user",
            "content":prompt
        }
    ]
    while True:
       try:
          reply = generate_reply(message, 1, "nlq")
          break
       except Exception as ex:
          print(ex)
          print("api error... wait for 3s")
          time.sleep(3)

    print(reply)
    exit()
    db_ann[db_id] = reply

with open("../data/database_anno.json", 'w') as f:
   json.dump(db_ann, f, indent=4)
    