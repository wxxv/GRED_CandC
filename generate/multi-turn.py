import json
import os
import time
import math
from openai import OpenAI
from tqdm import tqdm
from prompt import ask_gen_question, answer_gen_question
import pandas as pd
import math

# 初始化OpenAI客户端
client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

db_ann_file_path = "./nvBench-Rob/database_anno.json"
data_path = "./nvBench-Rob/{}/result_multi-turn/{}_result_gen_candidate_set_with_content_prob_gpt4o.json"
result_save_path = "./nvBench-Rob/{}/result_multi-turn/{}_result_multi-turn_gpt4o.json"
DATASET_SCHEMA = './nvBench-Rob/tables.json'

# System messages for different models
message = [
    {
        "role": "system",
        "content": """You are a helpful assistant that can generate a question based on uncertain information."""
    },
    {
        "role": "user",
        "content": ask_gen_question
    },
    {
        "role": "assistant",
        "content": answer_gen_question
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

def prompt_maker(db_id:str, nlq:str, predict_dvq_set:str, content_prob:str, keyword:str):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]

    db = generate_schema('browser_web_robust')
    
    prompt = db + "\n\n" + """### Natural Language Question (NLQ): 
# {}

### Possible Data Visualization Query (DVQs):
{}
### Uncertain Data Visualization Query Information: 
{}

### Given Database Schemas, a Natural Language Question (NLQ), Possible Data Visualization Query (DVQs) and the Uncertain Data Visualization Query Information, please generate clear and concise questions you want to ask based on the content of \"{}\" of the Uncertain Data Visualization Query Information.""".format(nlq, predict_dvq_set, content_prob, keyword)
    return prompt

def generate_reply(messages, n=1, flag="vql"):
    # print("generate...")
    try:
        completions = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-mini",
            messages=messages,
            n = n,
            stream = False,
        temperature=0.0,
            frequency_penalty=0.0,
            presence_penalty=-0.0,
        )
    except Exception as ex:
        print("Exception in generate_reply")
        print(f"API error: {ex}")
        print("Waiting for 3s...")
        exit()
        time.sleep(3)

    mes = completions.choices[0].message.content
    if flag == "vql":
        all_p_vqls = []
        for i in range(n):
            vql = completions.choices[i].message.content
            # print(vql)
            all_p_vqls.append(vql)
    else:
        return completions.choices[0].message.content.replace("\n", "")
    return all_p_vqls

def get_dvqs(dvqs:list):
    prompt = ""
    for i, s in enumerate(dvqs):
        s = s.replace(" ) ", ") ")
        prompt += f"{i+1} - " + s + "\n"
    return prompt

def calculate_entropy(probabilities):
    """Calculate entropy of a probability distribution."""
    # Normalize probabilities
    total = sum(probabilities)
    if total == 0:
        return 0
    
    normalized_probs = [p/total for p in probabilities]
    
    # Calculate entropy
    entropy = 0
    for prob in normalized_probs:
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

def select_keyword(content_prob):
    """Select keyword based on probability of non-None content and entropy."""
    max_score = -float('inf')
    selected_keyword = None
    
    for keyword, content_dict in content_prob.items():
        # Calculate probability of content not being None
        non_none_prob = 1 - content_dict.get("None", 0)
        
        # Get probabilities of non-None contents
        non_none_probs = [prob for content, prob in content_dict.items() if content != "None"]
        
        if non_none_probs:
            # Calculate entropy of non-None parts
            entropy = calculate_entropy(non_none_probs)
            
            # Calculate score as product of non-None probability and entropy
            score = non_none_prob * entropy
            # print(f"keyword: {keyword}, Score: {score}")
            # print("-"*100)
            
            if score >= max_score:
                max_score = score
                selected_keyword = keyword
    
    return selected_keyword

def get_answer(question, db_id, nlq, target):
    """Get answer from the ground truth model."""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that can answer the questions based on the content of the Data Visualization Query."""
        },
        {
            "role": "user",
            "content": f"""Given the Database Schema, the Natural Language Question and its Ground Truth (Correct Data Visualization Query):

{generate_schema(db_id)}

### Natural Language Question:
# {nlq}

### Ground Truth (Correct Data Visualization Query): 
# {target}

### Suppose you can access the Ground Truth (Correct Data Visualization Query) to the Natural Language Question. Please reply to the Follow-up Questions based on the Ground Truth. You must strictly follow the content of Ground Truth (Correct Data Visualization Query) in the answer.
# Follow-up Question:
# {question}"""
        }
    ]
    
    while True:
        try:
            response = client.chat.completions.create(
                # model="gpt-3.5-turbo-0125",
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response.choices[0].message.content
        except Exception as ex:
            print("Exception in get_answer")
            print(f"API error: {ex}")
            print("Waiting for 3s...")
            exit()
            time.sleep(3)

def select_correct_dvq(question, answer, predict_dvq_set):
    """Select the correct DVQ based on the ground truth answer."""
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that can select the most appropriate data visualization query based on the given answer."""
        },
        {
            "role": "user",
            "content": f"""Given the following information, please select the most appropriate data visualization query:

Possible DVQs:
{predict_dvq_set}

Question: {question}

Answer: {answer}

Please only reply the most appropriate DVQ without the number prefix from the Possible DVQs that matches the given answer."""
        }
    ]

    while True:
        try:
            response = client.chat.completions.create(
                # model="gpt-3.5-turbo-0125",
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response.choices[0].message.content
        except Exception as ex:
            print("Exception in select_correct_dvq")
            print(f"API error: {ex}")
            print("Waiting for 3s...")
            exit()
            time.sleep(3)

if __name__ == "__main__":
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
            final_dvq = example['predict_debugged_db_ann']
            predict_dvq_set = example['predict_dvq_set']
            content_prob = example['content_prob']
            possible_dvqs = get_dvqs(predict_dvq_set)

            keyword = select_keyword(content_prob)
            if True:
                prompt = prompt_maker(db_id, nlq, possible_dvqs, content_prob, keyword)
                # print(prompt)
                # print("-"*100)
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
                        question = generate_reply(messages, 1, "nlq")
                        # print(f"Generated question: {question}")
                        # print("-"*100)
                        
                        # Get answer from ground truth model with target information
                        answer = get_answer(question, db_id, nlq, target)
                        # print(f"Answer: {answer}")
                        # print("-"*100)
                        
                        # Select correct DVQ based on answer
                        final_dvq = select_correct_dvq(question, answer, possible_dvqs)
                        # print(f"Final DVQ: {final_dvq}")
                        # print("-"*100)
                        
                        break
                    except Exception as ex:
                        print(f"Error: {ex}")
                        print("Waiting for 3s...")
                        exit()
                        time.sleep(3)
                
                # exit()                
                example_new = example.copy()
                # example_new['prompt'] = messages[-1]['content']
                example_new['generated_question'] = question
                example_new['answer'] = answer
                example_new['final_dvq'] = final_dvq
                data_new.append(example_new)
                with open(result_save_path.format(mode, mode), 'w') as f:
                    json.dump(data_new, f, indent=4)
                # exit()
                if index == 19:
                    exit()

        
