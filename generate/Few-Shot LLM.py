import openai
import numpy as np
from scipy.spatial.distance import cosine
import json
import pandas as pd
from tqdm import tqdm
import pickle
import os
import time

from utils import generate_reply, generate_schema

data_file_path = "../data/{}/{}.csv"
result_save_path = "../data/{}/{}_result_prompt_chatgpt_0125.json"


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
    for mode in ['dev_nlq', 'dev_schema']:
    # for mode in ['dev_nvBench']:

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
                    reply = "Visualize " + generate_reply(message, 1, "nlq").replace("\n", " ")
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