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

A: Let’s think step by step!"""


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
# Present a bar graph representing the IDs and names of web accelerators that are compatible with two or more browsers, and kindly sort the y-axis in ascending order.

### Possible Data Visualization Query (DVQs) with their probabilities: 
{
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification ORDER BY T1.identification ASC": 0.4,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification GROUP BY name, identification ORDER BY identification ASC": 0.3,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification HAVING COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.2,
    "Visualize BAR SELECT name , identification FROM Web_client_accelerator AS T1 JOIN accelerator_compatible_browser AS T2 ON T2.accelerator_identification = T1.identification WHERE COUNT(DISTINCT T2.browser_identification) >= 2 ORDER BY identification ASC": 0.1
}

#### Given Database Schemas, Natural Language Question (NLQ) and Possible Data Visualization Query (DVQs) with their probabilities, for every DVQ keyword existing in the Possible Data Visualization Query (DVQs), please statistic their contents (include "None") in the form of a dictionary in JSON format. Follow these instructions:
# 1. For each keyword, please statistic the content (include "None") in the DVQs, and give the probability mass function in the form of a dictionary.
# 2. For each keyword, the sum of the probability mass function must be 1.
# 3. You should also use the form "!= \\"null\\"" to escape the double quotes.
# 4. Keep two decimal places for the probabilities.

A: Let’s think step by step!"""

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

### Given Database Schemas, a Natural Language Question (NLQ), Possible Data Visualization Query (DVQs) and the Uncertain Data Visualization Query Information, please generate one or two questions you want to ask based on the content of "ORDER BY" of the Uncertain Data Visualization Query Information. Focus on how to choose the content of "ORDER BY" from the Uncertain Data Visualization Query Information."""

answer_gen_question = """Would you like the web accelerators to be sorted by their IDs in ascending or descending order on the y-axis? Also, would you prefer to prefix the column names in the ORDER BY clause with their table aliases?"""
