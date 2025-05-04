# import math


# def entropy_calculation(probabilities):
#     probabilities = [probability / sum(probabilities) for probability in probabilities]
#     # print(probabilities)
#     entropy = 0
#     for probability in probabilities:
#         if probability > 0:
#             entropy -= probability * math.log10(probability)
#     return entropy


# # 示例概率列表
# # probabilities = [0,0.25,0.4,0] 
# # entropy = entropy_calculation(probabilities)
# # print(f"给定概率的熵为: {entropy}")
    

# # a = entropy_calculation([0.1,0.1,0.4,0.4]) - 0.1*entropy_calculation([1,0,0,0]) - 0.1*entropy_calculation([0,1,0,0]) - 0.8*entropy_calculation([0,0,0.8,0.8])    # RQ1
# # a = entropy_calculation([0.1,0.1,0.4,0.4]) - 0.5*entropy_calculation([0.1, 0.1, 0.8, 0]) - 0.5*entropy_calculation([0.1, 0.1, 0.8, 0])     # RQ2

# a = entropy_calculation([0.4,0.15,0.2,0.25]) - 0.15/0.35*entropy_calculation([0.4,0.35,0,0.25]) - 0.2/0.35*entropy_calculation([0.4,0.35,0,0.25])
# print(a)


# # b = entropy_calculation([0.1, 0.1 ,0.8])
# # b = 0.8 * entropy_calculation([0.5, 0.5])

# b = 0.35*entropy_calculation([0.5, 0.5])
# print(b)

import json
from collections import defaultdict

def compute_pmf(candidates):
    # Step 1: Initialize the structure for storing keyword contents and their corresponding probabilities
    keyword_contents = defaultdict(lambda: defaultdict(float))
    keyword_probabilities = defaultdict(float)

    # Step 2: Accumulate the content and its associated probability for each keyword
    for candidate, prob in candidates.items():
        # Split the candidate into its individual keywords and contents
        keywords = ['Visualize', 'SELECT', 'FROM', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'WHERE']
        content_dict = {
            'Visualize': 'BAR',
            'SELECT': 'name , identification',
            'FROM': 'Web_client_accelerator AS T1',
            'JOIN': 'accelerator_compatible_browser AS T2',
            'ON': 'T2.accelerator_identification = T1.identification',
            'GROUP BY': 'name, identification',
            'ORDER BY': 'T1.identification ASC',
            'HAVING': 'COUNT(DISTINCT T2.browser_identification) >= 2',
            'WHERE': 'COUNT(DISTINCT T2.browser_identification) >= 2'
        }
        
        # Special case for extracting variable contents
        if 'GROUP BY' in candidate:
            content_dict['GROUP BY'] = 'name, identification' if 'GROUP BY name, identification' in candidate else 'None'
        
        if 'ORDER BY' in candidate:
            if 'T1.identification ASC' in candidate:
                content_dict['ORDER BY'] = 'T1.identification ASC'
            else:
                content_dict['ORDER BY'] = 'identification ASC'
        
        if 'HAVING' in candidate:
            content_dict['HAVING'] = 'COUNT(DISTINCT T2.browser_identification) >= 2'
        
        if 'WHERE' in candidate:
            content_dict['WHERE'] = 'COUNT(DISTINCT T2.browser_identification) >= 2'

        # Step 3: Accumulate the probabilities for each keyword and its content
        for keyword in keywords:
            content = content_dict[keyword]
            keyword_contents[keyword][content] += prob
            keyword_probabilities[keyword] += prob

    # Step 4: Normalize the probabilities for each content under every keyword
    pmf_result = {}
    for keyword, contents in keyword_contents.items():
        total_prob = keyword_probabilities[keyword]
        pmf_result[keyword] = {content: prob / total_prob for content, prob in contents.items()}

    # Return the result in JSON format
    return json.dumps(pmf_result, indent=4)


# Example candidate queries with their associated probabilities
candidates = {
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC": 0.4,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC": 0.3,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.2,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.1
}

# Compute PMF and output the result
pmf_result = compute_pmf(candidates)
print(pmf_result)
