ask_gen_candidate_set = """### Database Schemas:
# Table basketball_match, columns = [ * , teamID , schoolID , team_Name , ACC_regular_season , percentage_of_ACC , ACC_home , ACC_Street , Total_Games , percentage_of_all_games , all_home , total_street , total_neutral ]
# Table university, columns = [ * , schoolID , school , POSITION , established , association , registration , Nick_Name , basic_conference ]
# Foreign_keys = [ basketball_match.schoolID = university.schoolID ]

### Natural Language Question (NLQ): 
# Determine the cumulative count of students enrolled in colleges established after the year 1850 for each type of affiliation. Display the results in a bar chart.

### Given a database schema, natural language question, and original DVQ, generate a set of candidate DVQs with their probabilities.
# Rules:
# 1. Only modify content, not structure or keywords
# 2. Include original DVQ as first candidate
# 3. Return JSON dictionary: {candidate_dvq: probability}
# 4. Probabilities must sum to 1.0

### Original DVQ:
# Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference
A: Let's think step by step!"""


answer_gen_candidate_set="""{
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.6,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established >= 1850 GROUP BY basic_conference": 0.3,
    "Visualize BAR SELECT association , COUNT(*) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.1
}"""



















ask_gen_prob = """### Database Schemas:
# Table Web_client_accelerator, columns = [ * , identification , name , Operating_system , user , link ]
# Table accelerator_compatible_browser, columns = [ * , accelerator_identification , browser_identification , compatible_since_year ]
# Table browser, columns = [ * , identification , name , market_share ]
# Foreign_keys = [ accelerator_compatible_browser.browser_identification = browser.identification , accelerator_compatible_browser.accelerator_identification = Web_client_accelerator.identification ]

### Natural Language Question (NLQ): 
# Present a bar graph representing the IDs and names of web accelerators that are compatible with two or more browsers, and kindly sort the y-axis in ascending order.

### Possible Data Visualization Query (DVQs) with their probabilities: 
# "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC" : 0.4,
# "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC" : 0.3,
# "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC" : 0.2,
# "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC" : 0.1

#### Given a set of database schemas, a natural language question (NLQ), and a list of candidate Data Visualization Queries (DVQs) with their associated probabilities, please compute the probability mass function (PMF) of the contents under each SQL keyword (e.g., SELECT, JOIN, WHERE) by treating the contents as discrete random variables.

# Step-by-step Instructions:
# 1. Content Identification per Keyword:
#   - For each SQL keyword that appears in the DVQs listed above, list all unique content variants found in the DVQs
#   - For each keyword, if a content appears in multiple DVQs, sum the probabilities of all DVQs in which it appears
#   - If a keyword is not present in given DVQs, treat its content as "None" with the corresponding probability sum of those DVQs
# 2. Normalization:
#   - For each SQL keyword, calculate the total probability of all its variants
#   - If the total is not 1.0, normalize by dividing each variant's probability by the total
#   - Round each probability to two decimal places
#   - Verify that the sum of probabilities for each keyword equals 1.0
# 3. Output Format:
#   Return the result as a JSON-formatted nested dictionary:
#   - Outer keys: SQL keywords (excluding ones that are completely missing across all DVQs)
#   - Inner keys: Content strings corresponding to each keyword
#   - Inner values: Their associated probabilities (rounded to two decimal places)
#   - Note: The sum of probabilities for each keyword must equal 1.0
#   - Example: If SELECT has variants with probabilities [0.70, 0.10], total is 0.80,
#     the normalized probabilities should be [0.88, 0.12] (rounded to two decimal places)

# Note: Verify that the sum of probabilities of the contents for each keyword equals 1. When indicating that a field is not empty, you should also use the form "!= \\"null\\"", use the double quotes instead of the single quotes to indicate the string

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

### Possible Data Visualization Query (DVQs) with their probabilities:
{
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC": 0.4,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC": 0.3,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.2,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.1
}

### Uncertain Data Visualization Query Information: 
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

### Given Database Schemas, a Natural Language Question (NLQ), Possible Data Visualization Query (DVQs) and the Uncertain Data Visualization Query Information, please generate clear and concise questions you want to ask based on the content of "ORDER BY" of the Uncertain Data Visualization Query Information."""

answer_gen_question = """Would you like the web accelerators to be sorted by their IDs in ascending or descending order on the y-axis? Also, would you prefer to prefix the column names in the ORDER BY clause with their table aliases?"""
