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
from collections import Counter
from config import OPENAI_BASE_URL, OPENAI_API_KEY

client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

embedding_file_path = "./nvBench-Rob/nlq_embedding.pkl"
embedding_test_file_path = "./nvBench-Rob/nlq_embedding_test.pkl"
data_file_path = "./nvBench-Rob/{}/{}.csv"
result_save_path = "./nvBench-Rob/{}/result_rebuttal/{}_result_multi_agent_debate.json"

# Multi-Agent Debate 配置
NUM_AGENTS = 3  # Agent 数量
NUM_ROUNDS = 1  # 辩论轮数
TEMPERATURE = 0.7

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

def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "# Table " + name+ ', columns = [ '
    for index, row in group.iterrows():
      output += row[" Field Name"] + ' , '
    output = output[:-2] + ']\n'
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

DATASET_SCHEMA = './nvBench-Rob/tables.json'
spider_schema,spider_primary,spider_foreign = creating_schema(DATASET_SCHEMA)

def generate_initial_answer(messages, agent_id, temperature=0.7):
    """
    生成 agent 的初始答案
    """
    start_time = time.time()
    
    # 给每个 agent 不同的 system prompt，增加多样性
    system_prompts = [
        "You are an expert in data visualization. Please follow the syntax in the examples instead of SQL syntax. Focus on accuracy and correctness.",
        "You are a careful data analyst. Please follow the syntax in the examples instead of SQL syntax. Think step by step and pay attention to details.",
        "You are a creative problem solver. Please follow the syntax in the examples instead of SQL syntax. Consider alternative approaches."
    ]
    
    # 使用不同的 system prompt
    agent_messages = [
        {
            "role": "system",
            "content": system_prompts[agent_id % len(system_prompts)]
        },
        messages[1]  # user message
    ]
    
    completions = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini",
        messages=agent_messages,
        stream=False,
        temperature=temperature,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    end_time = time.time()
    latency = end_time - start_time
    
    answer = completions.choices[0].message.content
    
    # 获取 token 使用信息
    token_usage = {
        "prompt_tokens": completions.usage.prompt_tokens,
        "completion_tokens": completions.usage.completion_tokens,
        "total_tokens": completions.usage.total_tokens
    }
    
    return answer, token_usage, latency

def generate_debate_answer(schema, nlq, agent_id, other_answers, round_num, temperature=0.7):
    """
    生成辩论轮次中的答案
    agent_id: 当前 agent 的 ID
    other_answers: 其他 agent 的答案列表 [(agent_id, answer), ...]
    round_num: 当前辩论轮数
    """
    start_time = time.time()
    
    # 构建辩论 prompt
    debate_prompt = f"""{schema}

### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{nlq}"

### Debate Round {round_num}

You are Agent {agent_id + 1}. Here are the answers from other agents:

"""
    
    for other_agent_id, other_answer in other_answers:
        debate_prompt += f"""Agent {other_agent_id + 1}'s answer:
{other_answer}

"""
    
    debate_prompt += f"""Now, based on the question, database schema, and other agents' answers:
1. Analyze whether other agents' answers are correct or have issues
2. Improve your own answer if you find better approaches from others
3. Explain your reasoning briefly

Please provide your improved Data Visualization Query:
A: Visualize """
    
    system_prompt = f"You are Agent {agent_id + 1} in a multi-agent debate. Carefully consider other agents' answers and provide the best possible DVQ. Follow the DVQ syntax, not SQL syntax."
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": debate_prompt
        }
    ]
    
    completions = client.chat.completions.create(
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini",
        messages=messages,
        stream=False,
        temperature=temperature,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    end_time = time.time()
    latency = end_time - start_time
    
    answer = completions.choices[0].message.content
    
    # 获取 token 使用信息
    token_usage = {
        "prompt_tokens": completions.usage.prompt_tokens,
        "completion_tokens": completions.usage.completion_tokens,
        "total_tokens": completions.usage.total_tokens
    }
    
    return answer, token_usage, latency

def normalize_vql(vql):
    """
    标准化 VQL 以便进行比较
    """
    vql = vql.replace("\n", " ").strip()
    vql = " ".join(vql.split())
    
    # 如果回答包含解释，尝试提取 DVQ 部分
    if "Visualize" in vql:
        # 找到第一个 Visualize
        idx = vql.find("Visualize")
        vql = vql[idx:]
        # 如果有多行，只取第一行（DVQ 通常在第一行）
        lines = vql.split('.')
        if lines:
            vql = lines[0].strip()
    
    # 移除 "Visualize" 前缀以便比较
    if vql.startswith("Visualize "):
        vql = vql[10:]
    
    return vql

def judge_discriminative(agent_answers, temperature=0.3):
    """
    Judge - Discriminative Mode
    判断当前轮次的 agent 答案是否已经达成共识（正确解决方案）
    
    返回: (是否达成共识, 共识答案, 判断说明)
    """
    # 标准化答案
    normalized_answers = [normalize_vql(a) for a in agent_answers]
    vote_counts = Counter(normalized_answers)
    
    # 检查是否所有答案都一致
    if len(vote_counts) == 1:
        # 完全一致，认为达成共识
        return True, list(vote_counts.keys())[0], "All agents agree on the same answer.", {}, 0
    
    # 检查是否有绝对多数（超过 2/3）
    most_common = vote_counts.most_common(1)[0]
    winner = most_common[0]
    winner_count = most_common[1]
    consensus_ratio = winner_count / len(agent_answers)
    
    if consensus_ratio >= 0.67:  # 2/3 多数
        # 使用 LLM 作为 judge 来验证这个答案
        judge_prompt = f"""You are a judge evaluating the consensus in a multi-agent debate about generating Data Visualization Queries (DVQs).

The agents have provided the following answers:
"""
        for i, ans in enumerate(agent_answers, 1):
            judge_prompt += f"\nAgent {i}: {ans}\n"
        
        judge_prompt += f"""
The majority answer is: {winner}
Consensus ratio: {consensus_ratio:.2f} ({winner_count}/{len(agent_answers)} agents)

Question: Has the debate reached a reliable consensus? Should we accept this answer and stop the debate?

Consider:
1. Are the majority of agents agreeing on the same answer?
2. Is the answer syntactically correct for DVQ?
3. Are the differences in minority answers significant or minor variations?

Answer with "YES" or "NO" and provide a brief explanation.
"""
        
        try:
            start_time = time.time()
            completions = client.chat.completions.create(
                # model="gpt-3.5-turbo-0125",
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert judge evaluating consensus in multi-agent debates. Be strict but fair."
                    },
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ],
                stream=False,
                temperature=temperature,
                frequency_penalty=0,
                presence_penalty=0
            )
            end_time = time.time()
            
            judge_response = completions.choices[0].message.content
            judge_token_usage = {
                "prompt_tokens": completions.usage.prompt_tokens,
                "completion_tokens": completions.usage.completion_tokens,
                "total_tokens": completions.usage.total_tokens
            }
            judge_latency = end_time - start_time
            
            # 判断 judge 的决定
            if "YES" in judge_response.upper()[:50]:  # 检查开头是否有 YES
                return True, winner, judge_response, judge_token_usage, judge_latency
            else:
                return False, None, judge_response, judge_token_usage, judge_latency
                
        except Exception as ex:
            print(f"Judge error: {ex}, using simple majority vote")
            # 如果 judge 失败，回退到简单规则
            return consensus_ratio >= 0.67, winner if consensus_ratio >= 0.67 else None, f"Simple rule: consensus ratio {consensus_ratio:.2f}", {}, 0
    
    return False, None, f"No consensus: highest agreement is only {consensus_ratio:.2f}", {}, 0

def judge_extractive(debate_history, schema, nlq, temperature=0.3):
    """
    Judge - Extractive Mode
    从整个辩论历史中提取最终答案
    当辩论达到最大轮次限制时使用
    """
    # 构建辩论历史摘要
    history_summary = f"""### Database Schema:
{schema}

### Natural Language Question:
{nlq}

### Debate History:
"""
    
    for round_info in debate_history:
        history_summary += f"\nRound {round_info['round']}:\n"
        for agent_info in round_info['agents']:
            history_summary += f"  Agent {agent_info['agent_id'] + 1}: {agent_info['answer']}\n"
    
    judge_prompt = f"""{history_summary}

You are a judge tasked with extracting the best Data Visualization Query (DVQ) from the above debate history.

Instructions:
1. Review all the answers from different agents across all rounds
2. Consider how the answers evolved through the debate
3. Identify the most reasonable and correct DVQ
4. Pay attention to:
   - Syntax correctness
   - Semantic correctness (matching the question)
   - Majority opinion (but not blindly follow it if there are better alternatives)
   - Improvements made in later rounds

Please provide the final DVQ answer. Start your response with "Visualize" and provide only the DVQ, followed by a brief explanation.

Final DVQ:
"""
    
    try:
        start_time = time.time()
        completions = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert judge extracting the best answer from a multi-agent debate. Follow DVQ syntax, not SQL syntax."
                },
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ],
            stream=False,
            temperature=temperature,
            frequency_penalty=0,
            presence_penalty=0
        )
        end_time = time.time()
        
        judge_response = completions.choices[0].message.content
        judge_token_usage = {
            "prompt_tokens": completions.usage.prompt_tokens,
            "completion_tokens": completions.usage.completion_tokens,
            "total_tokens": completions.usage.total_tokens
        }
        judge_latency = end_time - start_time
        
        # 提取 DVQ 答案（第一行通常是答案）
        lines = judge_response.strip().split('\n')
        extracted_answer = lines[0].strip()
        
        # 确保以 Visualize 开头
        if not extracted_answer.startswith("Visualize"):
            # 尝试在响应中找到 Visualize
            for line in lines:
                if "Visualize" in line:
                    extracted_answer = line.strip()
                    break
        
        return extracted_answer, judge_response, judge_token_usage, judge_latency
        
    except Exception as ex:
        print(f"Extractive judge error: {ex}, using majority vote as fallback")
        # 回退到多数投票
        final_answers = debate_history[-1]['agents']
        agent_answers = [agent['answer'] for agent in final_answers]
        normalized_answers = [normalize_vql(a) for a in agent_answers]
        vote_counts = Counter(normalized_answers)
        winner = vote_counts.most_common(1)[0][0]
        
        if not winner.startswith("Visualize "):
            winner = "Visualize " + winner
        
        return winner, "Fallback to majority vote due to error", {}, 0

def calculate_consensus_score(agent_answers):
    """
    计算共识得分
    """
    normalized_answers = [normalize_vql(a) for a in agent_answers]
    vote_counts = Counter(normalized_answers)
    most_common = vote_counts.most_common(1)[0]
    winner_count = most_common[1]
    consensus_score = winner_count / len(agent_answers)
    
    # 构建投票详情
    vote_details = {}
    for answer, count in vote_counts.items():
        vote_details[answer] = {
            "count": count,
            "percentage": count / len(agent_answers)
        }
    
    return consensus_score, vote_details

def get_embedding_from_file(vql:str, embedding_file:list):
    for v in embedding_file:
        if v['NLQ'].lower() == vql.lower():
            return v['Embedding']
    raise RuntimeError("No such NLQ in Embedding file: {}".format(vql))

def rag_by_nlq(nlq:str, k=10):
    document_all = pd.DataFrame(nlq_embedding_all)
    document_embeddings = document_all['Embedding'].to_list()

    try:
        question_embedding = get_embedding_from_file(nlq, nlq_test_embedding_all)
    except:
        raise RuntimeError("No embedding found for NLQ: {}".format(nlq))
    
    similarities = [1 - cosine(question_embedding, doc_embedding) for doc_embedding in document_embeddings]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_row = document_all.loc[top_k_indices.tolist()][['NLQ', "VQL","db_id"]]

    examples = []
    for index, row in top_k_row.iterrows():
            example = {
                    "NLQ":row['NLQ'],
                    "VQL":row['VQL'],
                    "schema":generate_schema(row['db_id'])
            }
            examples.append(example)

    examples.reverse()
    return examples

def prompt_maker(rag_list:list, db_id:str, nlq:str):
    prompt="""#### Given Natural Language Questions, Generate DVQs based on their correspoding Database Schemas.

"""
    for example in rag_list:
        prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{}"
### Data Visualization Query:
A: {}

""".format(example['schema'], example['NLQ'], example['VQL'])
        
    prompt += """{}
#
### Chart Type: [ BAR , PIE , LINE , SCATTER ]
### Natural Language Question:
# "{}"
### Data Visualization Query:
A: Visualize """.format(generate_schema(db_id), nlq)
    
    return prompt


if __name__ == '__main__':

    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:

        data = pd.read_csv(data_file_path.format(mode, mode))
        data_new = []

        if os.path.exists(result_save_path.format(mode, mode)):
            with open(result_save_path.format(mode, mode), 'r', encoding='gbk') as f:
                data_new = json.load(f)
        else:
            os.makedirs(os.path.dirname(result_save_path.format(mode, mode)), exist_ok=True)
            with open(result_save_path.format(mode, mode), 'w', encoding='gbk') as f:
                json.dump(data_new, f, indent=4)
        
        for index, d in tqdm(data.iterrows(), total=len(data), desc=f"Processing {mode} with Multi-Agent Debate"):
            if index < len(data_new):
                continue
            nlq = d['nl_queries']
            target = d['VQL']
            db_id = d['db_id']
            record_name = d['record_name']

            examples = rag_by_nlq(nlq, 10)
            prompt = prompt_maker(examples, db_id, nlq)

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

            # 记录所有轮次的信息
            debate_history = []
            total_token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            total_latency = 0.0
            judge_decisions = []  # 记录 judge 的决策
            early_stop = False
            final_answer = None

            while True:
                try:
                    # Round 0: 每个 agent 生成初始答案
                    agent_answers = []
                    round_0_info = {
                        "round": 0,
                        "agents": []
                    }
                    
                    for agent_id in range(NUM_AGENTS):
                        answer, token_usage, latency = generate_initial_answer(
                            message, agent_id, temperature=TEMPERATURE
                        )
                        answer_with_prefix = "Visualize " + answer.replace("\n", " ")
                        agent_answers.append(answer_with_prefix)
                        
                        total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                        total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                        total_token_usage["total_tokens"] += token_usage["total_tokens"]
                        total_latency += latency
                        
                        round_0_info["agents"].append({
                            "agent_id": agent_id,
                            "answer": answer_with_prefix,
                            "token_usage": token_usage,
                            "latency": latency
                        })
                    
                    debate_history.append(round_0_info)
                    
                    # Round 0 后检查是否已达成共识
                    judge_result = judge_discriminative(agent_answers, temperature=0)
                    has_consensus = judge_result[0]
                    consensus_answer = judge_result[1]
                    judge_explanation = judge_result[2]
                    judge_token = judge_result[3] if len(judge_result) > 3 else {}
                    judge_lat = judge_result[4] if len(judge_result) > 4 else 0
                    
                    if judge_token:
                        total_token_usage["prompt_tokens"] += judge_token.get("prompt_tokens", 0)
                        total_token_usage["completion_tokens"] += judge_token.get("completion_tokens", 0)
                        total_token_usage["total_tokens"] += judge_token.get("total_tokens", 0)
                        total_latency += judge_lat
                    
                    judge_decisions.append({
                        "round": 0,
                        "has_consensus": has_consensus,
                        "consensus_answer": consensus_answer,
                        "explanation": judge_explanation,
                        "token_usage": judge_token,
                        "latency": judge_lat
                    })
                    
                    if has_consensus:
                        # 提前终止
                        final_answer = consensus_answer
                        if not final_answer.startswith("Visualize "):
                            final_answer = "Visualize " + final_answer
                        early_stop = True
                        # 计算共识得分
                        consensus_score, vote_details = calculate_consensus_score(agent_answers)
                        print(f"  [Judge] Consensus reached at Round 0! Stopping debate.")
                        break
                    
                    # 进行多轮辩论
                    for round_num in range(1, NUM_ROUNDS + 1):
                        new_agent_answers = []
                        round_info = {
                            "round": round_num,
                            "agents": []
                        }
                        
                        for agent_id in range(NUM_AGENTS):
                            # 获取其他 agent 的答案
                            other_answers = [(i, agent_answers[i]) for i in range(NUM_AGENTS) if i != agent_id]
                            
                            # 生成改进的答案
                            answer, token_usage, latency = generate_debate_answer(
                                generate_schema(db_id), nlq, agent_id, 
                                other_answers, round_num, temperature=TEMPERATURE
                            )
                            
                            # 处理答案
                            answer_with_prefix = answer.replace("\n", " ")
                            if not answer_with_prefix.startswith("Visualize "):
                                answer_with_prefix = "Visualize " + answer_with_prefix
                            
                            new_agent_answers.append(answer_with_prefix)
                            
                            total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                            total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                            total_token_usage["total_tokens"] += token_usage["total_tokens"]
                            total_latency += latency
                            
                            round_info["agents"].append({
                                "agent_id": agent_id,
                                "answer": answer_with_prefix,
                                "token_usage": token_usage,
                                "latency": latency
                            })
                        
                        agent_answers = new_agent_answers
                        debate_history.append(round_info)
                        
                        # 每轮后使用 Judge - Discriminative Mode 检查
                        judge_result = judge_discriminative(agent_answers, temperature=0)
                        has_consensus = judge_result[0]
                        consensus_answer = judge_result[1]
                        judge_explanation = judge_result[2]
                        judge_token = judge_result[3] if len(judge_result) > 3 else {}
                        judge_lat = judge_result[4] if len(judge_result) > 4 else 0
                        
                        if judge_token:
                            total_token_usage["prompt_tokens"] += judge_token.get("prompt_tokens", 0)
                            total_token_usage["completion_tokens"] += judge_token.get("completion_tokens", 0)
                            total_token_usage["total_tokens"] += judge_token.get("total_tokens", 0)
                            total_latency += judge_lat
                        
                        judge_decisions.append({
                            "round": round_num,
                            "has_consensus": has_consensus,
                            "consensus_answer": consensus_answer,
                            "explanation": judge_explanation,
                            "token_usage": judge_token,
                            "latency": judge_lat
                        })
                        
                        if has_consensus:
                            # 提前终止辩论
                            final_answer = consensus_answer
                            if not final_answer.startswith("Visualize "):
                                final_answer = "Visualize " + final_answer
                            early_stop = True
                            # 计算共识得分
                            consensus_score, vote_details = calculate_consensus_score(agent_answers)
                            print(f"  [Judge] Consensus reached at Round {round_num}! Stopping debate.")
                            break
                    
                    # 如果达到最大轮次仍未达成共识，使用 Judge - Extractive Mode
                    if not early_stop:
                        print(f"  [Judge] Max rounds reached. Using Extractive Mode...")
                        extracted_answer, judge_explanation, judge_token, judge_lat = judge_extractive(
                            debate_history, generate_schema(db_id), nlq, temperature=0
                        )
                        
                        if judge_token:
                            total_token_usage["prompt_tokens"] += judge_token.get("prompt_tokens", 0)
                            total_token_usage["completion_tokens"] += judge_token.get("completion_tokens", 0)
                            total_token_usage["total_tokens"] += judge_token.get("total_tokens", 0)
                            total_latency += judge_lat
                        
                        judge_decisions.append({
                            "round": "extractive",
                            "extracted_answer": extracted_answer,
                            "explanation": judge_explanation,
                            "token_usage": judge_token,
                            "latency": judge_lat
                        })
                        
                        final_answer = extracted_answer
                        
                        # 计算最终的共识得分
                        consensus_score, vote_details = calculate_consensus_score(agent_answers)
                    
                    # 确保所有情况下都有 consensus_score 和 vote_details
                    if 'consensus_score' not in locals():
                        consensus_score, vote_details = calculate_consensus_score(agent_answers)
                    
                    break
                    
                except Exception as ex:
                    print(ex)
                    print("api error, wait for 3s...")
                    if "maximum context length" in str(ex):
                        examples = rag_by_nlq(nlq, 8)
                        prompt = prompt_maker(examples, db_id, nlq)
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

            
            print("=" * 80)
            print(f"Question: {nlq}")
            print(f"\nMulti-Agent Debate ({NUM_AGENTS} agents, max {NUM_ROUNDS} rounds):")
            
            for round_info in debate_history:
                print(f"\n  Round {round_info['round']}:")
                for agent_info in round_info['agents']:
                    print(f"    Agent {agent_info['agent_id'] + 1}: {agent_info['answer'][:100]}...")
            
            print(f"\nJudge Decisions:")
            for judge_dec in judge_decisions:
                if judge_dec.get('round') == "extractive":
                    print(f"  [Extractive Mode] Final answer extracted from debate history")
                    print(f"    Explanation: {judge_dec.get('explanation', '')[:100]}...")
                else:
                    round_num = judge_dec['round']
                    has_consensus = judge_dec['has_consensus']
                    print(f"  Round {round_num}: {'✓ Consensus' if has_consensus else '✗ No consensus'}")
                    if has_consensus:
                        print(f"    Answer: {judge_dec.get('consensus_answer', '')[:80]}...")
            
            print(f"\nFinal Answer (Consensus Score: {consensus_score:.2f}, Early Stop: {early_stop}):")
            print(f"  {final_answer}")
            print(f"\nTarget:")
            print(f"  {target}")
            print(f"\nTotal Token Usage: {total_token_usage['total_tokens']} tokens")
            print(f"Total Latency: {total_latency:.2f} seconds")
            print("=" * 80)
            print()

            example = {
                "record_name": record_name,
                "db_id": db_id,
                "target": target,
                "nlq": nlq,
                "predict_multi_agent_debate": final_answer,
                "consensus_score": consensus_score,
                "final_round_answers": agent_answers,
                "vote_details": vote_details,
                "debate_history": debate_history,
                "judge_decisions": judge_decisions,
                "early_stop": early_stop,
                "actual_rounds": len(debate_history),
                "num_agents": NUM_AGENTS,
                "max_rounds": NUM_ROUNDS,
                "temperature": TEMPERATURE,
                "total_token_usage": total_token_usage,
                "total_latency_seconds": total_latency
            }

            data_new.append(example)

            with open(result_save_path.format(mode, mode), 'w') as f:
                json.dump(data_new, f, indent=4, ensure_ascii=False)
            # exit()

