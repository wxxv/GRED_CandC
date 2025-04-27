ask_gen_candidate_set = """### Database Schemas:
# Table basketball_match, columns = [ * , teamID , schoolID , team_Name , ACC_regular_season , percentage_of_ACC , ACC_home , ACC_Street , Total_Games , percentage_of_all_games , all_home , total_street , total_neutral ]
# Table university, columns = [ * , schoolID , school , POSITION , established , association , registration , Nick_Name , basic_conference ]
# Foreign_keys = [ basketball_match.schoolID = university.schoolID ]

### Natural Language Question (NLQ): 
# Determine the cumulative count of students enrolled in colleges established after the year 1850 for each type of affiliation. Display the results in a bar chart.

### Possible Data Visualization Query (DVQ): 
# Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference

#### Given Natural Language Question (NLQ), reference the above Possible Data Visualization Query, please generate DVQs based on their correspoding Database Schemas. Follow these instructions:
# 1. Please consider all possible content of each keyword, such as what follows "SELECT", "WHERE", "GROUP BY" or "ORDER BY". 
# 2. Output the above Possible Data Visualization Query and all other possible correct DVQs with their probabilities in the form of a dictionary in JSON format.
# 3. Please note that when indicating that a field is not empty, you should also use the form "!= \\"null\\"", instead of "IS NOT NULL", although they are exactly the same. Note that use the double quotes instead of the single quotes to indicate the string. Do not generate the column names that do not exist in the database schemas.

A: Let's think step by step!"""


answer_gen_candidate_set="""{
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference": 0.35,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established >= 1850 GROUP BY basic_conference": 0.25,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference ORDER BY basic_conference ASC": 0.15,
    "Visualize BAR SELECT basic_conference , SUM(registration) FROM university WHERE established > 1850 GROUP BY basic_conference ORDER BY basic_conference DESC": 0.15,
    "Visualize BAR SELECT association , COUNT(*) FROM university WHERE established > 1850 GROUP BY association": 0.1
}"""



















ask_gen_prob = """### Database Schemas:
# Table Web_client_accelerator, columns = [ * , identification , name , Operating_system , user , link ]
# Table accelerator_compatible_browser, columns = [ * , accelerator_identification , browser_identification , compatible_since_year ]
# Table browser, columns = [ * , identification , name , market_share ]
# Foreign_keys = [ accelerator_compatible_browser.browser_identification = browser.identification , accelerator_compatible_browser.accelerator_identification = Web_client_accelerator.identification ]

### Natural Language Question (NLQ): 
For employees with salaries ranging from 8000 to 12000, and with either non-null commission or department number not equal to 40, provide a comparison of the total employee_id sum grouped by hire_date bins over time using a bar chart. Please display the results in descending order by the total number count.

### Candidate Set of Data Visualization Query (DVQs) with their probabilities: 
# Visualize BAR SELECT date_of_hire , SUM(employee_id) FROM employees WHERE wage BETWEEN 8000 AND 12000 AND COMMISSION_PCT != \"null\" OR Dept_ID != 40 ORDER BY SUM(employee_id) DESC BIN date_of_hire BY MONTH : 0.4
# Visualize BAR SELECT date_of_hire , SUM(employee_id) FROM employees WHERE wage >= 8000 AND wage <= 12000 AND (COMMISSION_PCT != \"null\" OR Dept_ID != 40) GROUP BY date_of_hire ORDER BY SUM(employee_id) DESC BIN date_of_hire BY MONTH : 0.3
# Visualize BAR SELECT TO_CHAR(date_of_hire, 'YYYY-MM') AS hire_month , COUNT(employee_id) FROM employees WHERE wage BETWEEN 8000 AND 12000 AND (COMMISSION_PCT != \"null\" OR Dept_ID != 40) GROUP BY hire_month ORDER BY COUNT(employee_id) DESC : 0.2
# Visualize BAR SELECT EXTRACT(MONTH FROM date_of_hire) AS hire_month , COUNT(employee_id) FROM employees WHERE wage BETWEEN 8000 AND 12000 AND (COMMISSION_PCT != \"null\" OR Dept_ID != 40) GROUP BY hire_month ORDER BY COUNT(employee_id) DESC : 0.1

#### Given a set of database schemas, a natural language question (NLQ), and a list of candidate Data Visualization Queries (DVQs) with their associated probabilities, please compute the probability mass function (PMF) of the contents under each SQL keyword (e.g., SELECT, JOIN, WHERE) by treating the contents as discrete random variables.

# Step-by-step Instructions:
# 1. Content Identification per Keyword:
#   - For each SQL keyword that appears in the DVQs listed above, list all unique content variants found in the DVQs
#   - For each keyword, if a content appears in multiple DVQs, sum the probabilities of all DVQs in which it appears
#   - If a keyword is not present in given DVQs, treat its content as "None" with the corresponding probability sum of those DVQs
# 2. Normalization:
#   - For each SQL keyword, ensure the sum of probabilities of all its variants equals 1.0
#   - Round each probability to two decimal places
# 3. Output Format:
#   Return the result as a JSON-formatted nested dictionary:
#   - Outer keys: SQL keywords (excluding ones that are completely missing across all DVQs)
#   - Inner keys: Content strings corresponding to each keyword
#   - Inner values: Their associated probabilities (rounded to two decimal places)

# Please note that when indicating that a field is not empty, you should also use the form "!= \\"null\\"", use the double quotes instead of the single quotes to indicate the string

A: Let's think step by step!"""

answer_gen_prob="""{
    "Visualize": {
        "BAR": 1.0
    },
    "SELECT": {
        "date_of_hire , SUM(employee_id)": 0.7,
        "TO_CHAR(date_of_hire, 'YYYY-MM') AS hire_month , COUNT(employee_id)": 0.2,
        "EXTRACT(MONTH FROM date_of_hire) AS hire_month , COUNT(employee_id)": 0.1
    },
    "FROM": {
        "employees": 1.0
    },
    "WHERE": {
        "wage BETWEEN 8000 AND 12000 AND COMMISSION_PCT != \"null\" OR Dept_ID != 40": 0.4,
        "wage >= 8000 AND wage <= 12000 AND (COMMISSION_PCT != \"null\" OR Dept_ID != 40)": 0.3,
        "wage BETWEEN 8000 AND 12000 AND (COMMISSION_PCT != \"null\" OR Dept_ID != 40)": 0.3
    },
    "GROUP BY": {
        "date_of_hire": 0.3,
        "hire_month": 0.3,
        "None": 0.4
    },
    "ORDER BY": {
        "SUM(employee_id) DESC": 0.7,
        "COUNT(employee_id) DESC": 0.3
    },
    "BIN": {
        "date_of_hire BY MONTH": 0.7,
        "None": 0.3
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

### Given Database Schemas, a Natural Language Question (NLQ), Possible Data Visualization Query (DVQs) and the Uncertain Data Visualization Query Information, please generate one or two questions you want to ask based on the content of "ORDER BY" of the Uncertain Data Visualization Query Information. Focus on how to choose the content of "ORDER BY" from the Uncertain Data Visualization Query Information."""

answer_gen_question = """Would you like the web accelerators to be sorted by their IDs in ascending or descending order on the y-axis? Also, would you prefer to prefix the column names in the ORDER BY clause with their table aliases?"""
