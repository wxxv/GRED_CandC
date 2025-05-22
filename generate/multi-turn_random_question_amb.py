import json
import os
import time
import math
from openai import OpenAI
from tqdm import tqdm
from prompt import ask_gen_question, answer_gen_question
import pandas as pd
import math
import sys
import os
import random

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trainer.sql_accuracy import to_VQL

# 初始化OpenAI客户端
client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-dBlXWGiSrbjB6RO9AcMSFfjGhS6O5unK1TWs0ul2tD6g2WT8',
    )

db_ann_file_path = "./nvBench-Rob/database_anno.json"
data_path = "./nvBench-Rob/{}/result_multi-turn/{}_result_gen_candidate_set_with_content_prob_gpt4o.json"
result_save_path = "./nvBench-Rob/{}/result_multi-turn/{}_result_multi-turn_gpt4o_random_question_amb.json"
DATASET_SCHEMA = './nvBench-Rob/tables.json'


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

def prompt_maker_gen_question(db_id:str, nlq:str, candidate_dvqs:str, content_prob:str, keywords:list, binary_keywords:list, max_score:float):
    with open(db_ann_file_path, 'r') as f:
        db_ann = json.load(f)
    db_ann = db_ann[db_id]
    db = generate_schema(db_id)
    # if max_score > 0:
    #     sub_prompt = f"""Given Database Schemas, a Natural Language Question (NLQ), Candidate DVQs, please generate a clear and concise question you want to ask based on the content of \"{keyword}\" of the Uncertain DVQ Information."""
    # else:
    #     sub_prompt = f"""Given Database Schemas, a Natural Language Question (NLQ), Candidate DVQs, please generate a clear and concise question you want to ask whether the \"{binary_keywords}\" is necessary in DVQ when answer the NLQ."""
    keyword = keywords[random.randint(0, len(keywords)-1)]
    sub_prompt = f"""Given Database Schemas, a Natural Language Question (NLQ), Candidate DVQs, please generate a clear and concise question you want to ask based on the content of \"{keyword}\" of the Uncertain DVQ Information.."""
    
    prompt = db + "\n\n" + """### Natural Language Question (NLQ): 
# {}

### Candidate Data Visualization Queries (DVQs, a new Programming Language abstracted from Vega-Zero):
{}
#### {}
### Uncertain DVQ Information: 
{}""".format(nlq, predict_dvq_set, sub_prompt, content_prob)
    return prompt


def prompt_maker_multi_turn(db_id:str, nlq:str):

    prompt="""#### Given Natural Language Questions, Generate DVQs based on correspoding Database Schemas.

"""
    prompt += """{}

### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{}"
### Data Visualization Query:
A: Visualize """.format(generate_schema(db_id), nlq)
    
    return prompt

def generate_question(messages, n=1, flag="vql"):
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
        print("Exception in generate_question")
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
    # selected_keyword = [None]
    selected_keyword = []

    # 存储只有两个键值对且包含None的关键词
    binary_keywords = []
    
    for keyword, content_dict in content_prob.items():
        # 检查是否只有两个键值对且包含None
        if len(content_dict) == 2 and "None" in content_dict:
            binary_keywords.append(keyword)
            selected_keyword.append(keyword)
            continue
            
        # Calculate probability of content not being None
        non_none_prob = 1 - content_dict.get("None", 0)
        # Get probabilities of non-None contents
        non_none_probs = [prob for content, prob in content_dict.items() if content != "None"]
        
        if non_none_probs:
            # Calculate entropy of non-None parts
            entropy = calculate_entropy(non_none_probs)

            if entropy > 0:
            
            # # Calculate score as product of non-None probability and entropy
            # score = non_none_prob * entropy
            # # print(f"keyword: {keyword}, Score: {score}")
            # # print("-"*100)
            
            # if score >= max_score:
            #     max_score = score
                selected_keyword.append(keyword)
    if selected_keyword == []:
        selected_keyword = ['None']
    return selected_keyword, binary_keywords, max_score

def get_answer(question, db_id, nlq, target):
    """Get answer from the ground truth model."""
    messages = [
        {
            "role": "system",
            "content": """You are an expert in data visualization."""
        },
        {
            "role": "user",
            "content": f"""Given the Database Schema, the Natural Language Question(NLQ):

{generate_schema(db_id)}

### NLQ:
# "{nlq}"

#### Suppose you can access the Correct Data Visualization Query(DVQ, a new Programming Language abstracted from Vega-Zero) of the NLQ. Please reply to the Follow-up Questions referring to the Correct DVQ.
### Correct DVQ: 
# {target}
### Follow-up Question:
# "{question}"
#### Note: Please answer the follow-up question by strictly referring to the Correct DVQ above but only return the conclusion without using "Correct DVQ" as subject. Your answer should only contain information that is explicitly present in the Correct DVQ. If the Correct DVQ doesn't contain information needed to answer the question, state that clearly. Remember, do not mention the details of the Correct DVQ in your description.
A: Let's think step by step!"""
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

def select_correct_dvq(messages):
    """Select the correct DVQ based on the ground truth answer."""

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
            debugged_db_ann_dvq = example['predict_debugged_db_ann']
            predict_dvq_set = example['predict_dvq_set']
            content_prob = example['content_prob']
            candidate_dvqs = get_dvqs(predict_dvq_set)

            debugged_db_ann_dvq_dict = to_VQL(debugged_db_ann_dvq)
            target_dict = to_VQL(target)
            if debugged_db_ann_dvq_dict != target_dict:
                keyword, binary_keywords, max_score = select_keyword(content_prob)
                prompt_multi_turn = prompt_maker_multi_turn(db_id, nlq)
                messages_multi_turn = [
                    {
                        "role":"system",
                        "content":"You are a data visualization expert."
                    },
                    {
                        "role":"user",
                        "content":prompt_multi_turn
                    },
                    {
                        "role":"assistant",
                        "content":debugged_db_ann_dvq + "\n\n" + "But I also have some other uncertain DVQs that may be correct:\n" + candidate_dvqs
                    }
                ]

                messages_gen_question = [
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
                prompt_gen_question = prompt_maker_gen_question(db_id, nlq, candidate_dvqs, content_prob, keyword, binary_keywords, max_score)
                messages_gen_question.append(
                    {
                        "role":"user",
                        "content": prompt_gen_question
                    }
                )

                while True:
                    try:
                        question = generate_question(messages_gen_question, 1, "nlq")
                        # print(f"Generated question: {question}")
                        # print("-"*100)
                        messages_multi_turn[-1]['content'] += "\nTo avoid the confusion of the content. " + question

                        answer = get_answer(question, db_id, nlq, target)
                        # print(f"Answer: {answer}")
                        # print("-"*100)

                        messages_multi_turn.append(
                            {
                                "role":"user",
                                "content":answer + "\n\n" + """According to the above history conversation, please reply only the most appropriate DVQ from the Possible DVQs.
                                A: Let's think step by step!"""
                            }
                        )

                        # for i in messages_multi_turn:
                        #     print(i['role'])
                        #     print(i['content'])
                        #     print("-"*100)

                        final_dvq = select_correct_dvq(messages_multi_turn)
                        if final_dvq.startswith("A: "):
                            final_dvq = "Visualize " + final_dvq.split("A: ")[-1]
                            final_dvq = "Visualize " + final_dvq.split("Visualize ")[-1]
                        else:
                            final_dvq = "Visualize " + final_dvq.split("Visualize ")[1]
                        
                        # print(final_dvq)
                        # exit()
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


            else:
                example_new = example.copy()
                example_new['generated_question'] = ""
                example_new['answer'] = ""
                example_new['final_dvq'] = debugged_db_ann_dvq
                data_new.append(example_new)
                with open(result_save_path.format(mode, mode), 'w') as f:
                    json.dump(data_new, f, indent=4)
            # if index == 19:
            #     exit()

        
