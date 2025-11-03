import openai
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time

from config import OPENAI_BASE_URL, OPENAI_API_KEY
from openai import OpenAI

data_file_path = "../nvBench-Rob/{}/{}.csv"
result_save_path = "../nvBench-Rob/{}/result_rebuttal/{}_result_few_shot_gpt3.5.json"

client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )


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


DATASET_SCHEMA = '../nvBench-Rob/tables.json'
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


examples = [
    {
        "NLQ":"For those employees whose salary is in the range of 8000 and 12000 and commission is not null or department number does not equal to 40, for  department_id,  hire_date, visualize the trend.",
        "VQL":"Visualize LINE SELECT HIRE_DATE , DEPARTMENT_ID FROM employees WHERE salary BETWEEN 8000 AND 12000 AND commission_pct != \"null\" OR department_id != 40",
        "schema":generate_schema("hr_1")
    },
    {
       "NLQ":"How many documents for each document type description? Visualize by a bar chart, and could you show Y-axis from high to low order?",
       "VQL":"Visualize BAR SELECT Document_Type_Description , COUNT(Document_Type_Description) FROM Ref_document_types AS T1 JOIN Documents AS T2 ON T1.document_type_code = T2.document_type_code GROUP BY Document_Type_Description ORDER BY COUNT(Document_Type_Description) DESC",
       "schema":generate_schema("cre_Docs_and_Epenses")
    },
    {
       "NLQ":"A stacked bar chart that computes the total number of wines with a price is bigger than 100 of each year, and group by grape. Next, Bin the year into the weekday interval. ",
       "VQL":"Visualize BAR SELECT Year , COUNT(Year) FROM WINE WHERE Price > 100 GROUP BY Grape ORDER BY YEAR BIN Year BY WEEKDAY",
       "schema":generate_schema("wine_1")
    },
    {
       "NLQ":"Show me about the distribution of  Start_from and the amount of Start_from bin start_from by weekday in a bar chart.",
       "VQL":"Visualize BAR SELECT Start_from , COUNT(Start_from) FROM hiring BIN Start_from BY WEEKDAY",
       "schema":generate_schema("employee_hire_evaluation")
    },
    {
       "NLQ":"What is the total cloud cover rates of the dates (bin into year interval) that had the top 5 cloud cover rates? You can draw me a bar chart for this purpose.",
       "VQL":"Visualize BAR SELECT date , SUM(cloud_cover) FROM weather BIN date BY YEAR",
       "schema":generate_schema("bike_1")
    },
    {
       "NLQ":"For the transaction dates if share count is smaller than 10, bin the dates into the year interval, and count them using a line chart, could you sort X from high to low order?",
       "VQL":"Visualize LINE SELECT date_of_transaction , COUNT(date_of_transaction) FROM TRANSACTIONS WHERE share_count < 10  ORDER BY date_of_transaction DESC BIN date_of_transaction BY YEAR",
       "schema":generate_schema("tracking_share_transactions")
    },
    {
       "NLQ":"For all employees in the same department and with the first name Clara, please give me a bar chart that bins hire date into the day of week interval, and count how many employees in each day.",
       "VQL":"Visualize BAR SELECT HIRE_DATE , COUNT(HIRE_DATE) FROM employees WHERE department_id = (SELECT department_id FROM employees WHERE first_name = \"Clara\") BIN HIRE_DATE BY WEEKDAY",
       "schema":generate_schema("hr_1")
    },
    {
       "NLQ":"Give me the comparison about All_Games_Percent over the All_Games , rank Y-axis in desc order.",
       "VQL":"Visualize BAR SELECT All_Games , All_Games_Percent FROM basketball_match ORDER BY All_Games_Percent DESC",
       "schema":generate_schema("university_basketball")
    },
    {
       "NLQ":"Visualize the name and their component amounts with a bar chart for all furnitures that have more than 10 components, order by the X in desc please.",
       "VQL":"Visualize BAR SELECT Name , Num_of_Component FROM furniture WHERE Num_of_Component > 10 ORDER BY Name DESC",
       "schema":generate_schema("manufacturer")
    },
    {
       "NLQ":"What is the number of the faculty members for each rank? Visualize in bar chart, and I want to order in asc by the Y.",
       "VQL":"Visualize BAR SELECT Rank , COUNT(Rank) FROM Faculty GROUP BY Rank ORDER BY COUNT(Rank) ASC",
       "schema":generate_schema("activity_1")
    }
]


def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    completions = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        # model="o3-2025-04-16",
        model="gpt-4o-mini",
        messages=messages,
        n = n,
        stream = False,
        temperature=0.0,
        # frequency_penalty=-0.5, # 避免重复性(-2.0 ~ 2.0)
        # presence_penalty=-0.5   # 生成新主题(-2.0 ~ 2.0)
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

    # for mode in ['dev_nlq_schema']:
    for mode in ['dev_nlq', 'dev_schema', 'dev_nlq_schema']:
    # for mode in ['dev_nlq']:

        data = pd.read_csv(data_file_path.format(mode, mode))
        data_new = []

        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r') as f:
                data_new = json.load(f)
        
        for index, d in tqdm(data.iterrows(), total=len(data)):
            if index < len(data_new):
                continue
            nlq = d['nl_queries']
            target = d['VQL']
            db_id = d['db_id']
            record_name = d['record_name']

            examples = examples.copy()
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
                try:
                    reply = generate_reply(message, 1, "nlq").replace("\n", " ")
                    # print(reply)
                    if reply.startswith("Visualize"):
                        reply = reply
                    elif reply.startswith("A: "):
                        reply = reply.split("A: ", 1)[1]
                    else:
                        # reply = "Visualize " + generate_reply(message, 1, "nlq").replace("\n", " ")
                        reply = "Visualize " + reply
                    break
                except Exception as ex:
                    print(ex)
                    print("api error, wait for 3s...")
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
            
            # exit()