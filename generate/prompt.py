ask_gen_candidate_set = """### Database Schemas:
# Table basketball_match, columns = [ * , teamID , schoolID , team_Name , ACC_regular_season , percentage_of_ACC , ACC_home , ACC_Street , Total_Games , percentage_of_all_games , all_home , total_street , total_neutral ]
# Table university, columns = [ * , schoolID , school , POSITION , established , association , registration , Nick_Name , basic_conference ]
# Foreign_keys = [ basketball_match.schoolID = university.schoolID ]

### Natural Language Question (NLQ): 
# Determine the cumulative count of students enrolled in colleges established after the year 1850 for each type of affiliation. Display the results in a bar chart.

#### Given a Database Schema, Natural Language Question, and Original Data Visualization Query(DVQ, a new Programming Language abstracted from Vega-Zero), please generate a set of candidate DVQs with their probabilities that you think are correct. 
# Step-by-step Instructions:
# 1. Copy the Original DVQ as the first candidate without any modification.
# 2. Then for each of other candidate DVQs, only modify a content part of the Original DVQ, not structure or keywords.
# 3. Generate the probability that you think each of the candidate DVQs is correct.
# 4. Return format - JSON dictionary: {{candidate_dvq: probability}}
#### NOTE: Remember use '\"' to escape the double quotes in the candidate DVQs. Ensure the sum of probabilities is 1. Ensure the first candidate is the original DVQ.
### Original DVQ: 
# Visualize BAR SELECT basic_conference , SUM(Enrollment) FROM university WHERE established > 1850 GROUP BY basic_conference
A: Let's think step by step!"""


answer_gen_candidate_set="""{
    "Visualize BAR SELECT basic_conference , SUM(Enrollment) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.6,
    "Visualize BAR SELECT basic_conference , SUM(Enrollment) FROM university WHERE established >= 1850 GROUP BY basic_conference": 0.3,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.1
}"""



















ask_gen_prob = """### Given a set of candidate DVQs and their corresponding probabilities, please extract the contents associated with each DVQ keyword (e.g., VISUALIZE, SELECT, JOIN, WHERE, GROUP BY, etc.) from the DVQs and compute the probability mass function (PMF) for each keyword's content.

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
{
"Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC" : 0.4,
"Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC" : 0.3,
"Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC" : 0.2,
"Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC" : 0.1
}
A: Let's think step by step!"""

answer_gen_prob="""{
    "Visualize": {
        "BAR": 1.00
    },
    "SELECT": {
        "name , identification": 1.00,
    },
    "FROM": {
        "Web_client_accelerator AS T1": 1.00,
    },
    "JOIN": {
        "accelerator_compatible_browser AS T2": 1.00,
    },
    "ON": {
        "T2.accelerator_identification = T1.identification": 1.00,
    },
    "GROUP BY": {
        "name, identification": 0.30,
        "None": 0.70
    },
    "ORDER BY": {
        "T1.identification ASC": 0.40,
        "identification ASC": 0.60,
    },
    "HAVING": {
        "None": 0.80,
        "COUNT(DISTINCT T2.browser_identification) >= 2": 0.20
    },
    "WHERE": {
        "None": 0.90,
        "COUNT(DISTINCT T2.browser_identification) >= 2": 0.10
    }
}"""
















ask_gen_question = """### Database Schemas:
# Table Web_client_accelerator, columns = [ * , identification , name , Operating_system , user , link ]
# Table accelerator_compatible_browser, columns = [ * , accelerator_identification , browser_identification , compatible_since_year ]
# Table browser, columns = [ * , identification , name , market_share ]
# Foreign_keys = [ accelerator_compatible_browser.browser_identification = browser.identification , accelerator_compatible_browser.accelerator_identification = Web_client_accelerator.identification ]

### Natural Language Question (NLQ): 
# Present a bar graph representing the IDs and names of web accelerators that are compatible with two or more browsers, and kindly sort the y-axis in ascending order.

### Candidate Data Visualization Queries (DVQs, a new Programming Language abstracted from Vega-Zero):
1 - Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC
2 - Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC
3 - Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC
4 - Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC

### Uncertain DVQ Information: 
{
    "Visualize": {
        "BAR": 1.00
    },
    "SELECT": {
        "name , identification": 1.00,
    },
    "FROM": {
        "Web_client_accelerator AS T1": 1.00,
    },
    "JOIN": {
        "accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification": 1.00,
    },
    "GROUP BY": {
        "name, identification": 0.30,
        "None": 0.70
    },
    "ORDER BY": {
        "T1.identification ASC": 0.40,
        "identification ASC": 0.60,
    },
    "HAVING": {
        "None": 0.80,
        "COUNT(DISTINCT T2.browser_identification) >= 2": 0.20
    },
    "WHERE": {
        "None": 0.90,
        "COUNT(DISTINCT T2.browser_identification) >= 2": 0.10
    }
}

### Given Database Schemas, a Natural Language Question (NLQ), Candidate DVQs and the Uncertain DVQ Information, please generate a clear and concise questions you want to ask based on the content of "ORDER BY" of the Uncertain DVQ Information."""

answer_gen_question = """Would you like the web accelerators to be sorted by their IDs in ascending or descending order on the y-axis? Also, would you prefer to prefix the column names in the ORDER BY clause with their table aliases?"""
