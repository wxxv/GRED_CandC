# import math
import json


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


a = """To compute the probability mass function (PMF) for each DVQ keyword from the provided candidate DVQs, we will follow the steps outlined in the prompt.

### Step 1: Extract Contents Associated with Each Keyword

The candidate DVQs provided are:

1. **DVQ 1**:
   ```
   LINE SELECT Document_Date , COUNT(Document_Date) FROM Ref_Document_Types AS T1 JOIN Documents AS T2 ON T1.Document_Type_Code = T2.Document_Type_Code GROUP BY Document_Type_Name ORDER BY Document_Date ASC BIN Document_Date BY YEAR
   ```
   Probability: 0.5

2. **DVQ 2**:
   ```
   LINE SELECT Document_Date , COUNT(Document_Date) FROM Ref_Document_Types AS T1 JOIN Documents AS T2 ON T1.Document_Type_Code = T2.Document_Type_Code GROUP BY Document_Type_Name ORDER BY Document_Date DESC BIN Document_Date BY YEAR
   ```
   Probability: 0.3

3. **DVQ 3**:
   ```
   LINE SELECT Document_Date , COUNT(Document_Date) FROM Ref_Document_Types AS T1 JOIN Documents AS T2 ON T1.Document_Type_Code = T2.Document_Type_Code GROUP BY Document_Type_Name ORDER BY Document_Type_Name ASC BIN Document_Date BY YEAR
   ```
   Probability: 0.2

### Step 2: Identify Keywords and Extract Content

The keywords we will extract are: `LINE`, `SELECT`, `FROM`, `JOIN`, `ON`, `GROUP BY`, `ORDER BY`, and `BIN`.

- **LINE**: All candidates contain "LINE".
- **SELECT**: All candidates contain "Document_Date , COUNT(Document_Date)".
- **FROM**: All candidates contain "Ref_Document_Types AS T1".
- **JOIN**: All candidates contain "Documents AS T2".
- **ON**: All candidates contain "T1.Document_Type_Code = T2.Document_Type_Code".
- **GROUP BY**: All candidates contain "Document_Type_Name".
- **ORDER BY**: Different contents for each candidate.
- **BIN**: All candidates contain "Document_Date BY YEAR".

### Step 3: Group Identical Contents and Sum Probabilities

Now we will group the contents and sum the probabilities:

- **LINE**:
  - "None": 0.0 (not applicable, all have "LINE")

- **SELECT**:
  - "Document_Date , COUNT(Document_Date)": 1.0 (all candidates have the same content)

- **FROM**:
  - "Ref_Document_Types AS T1": 1.0 (all candidates have the same content)

- **JOIN**:
  - "Documents AS T2": 1.0 (all candidates have the same content)

- **ON**:
  - "T1.Document_Type_Code = T2.Document_Type_Code": 1.0 (all candidates have the same content)

- **GROUP BY**:
  - "Document_Type_Name": 1.0 (all candidates have the same content)

- **ORDER BY**:
  - "Document_Date ASC": 0.5
  - "Document_Date DESC": 0.3
  - "Document_Type_Name ASC": 0.2

- **BIN**:
  - "Document_Date BY YEAR": 1.0 (all candidates have the same content)

### Step 4: Normalize Probabilities

Now we will normalize the probabilities for each keyword's content to ensure they sum to 1.0.

### Final JSON Output

```json
{
    "LINE": {
        "None": 1.00
    },
    "SELECT": {
        "Document_Date , COUNT(Document_Date)": 1.00
    },
    "FROM": {
        "Ref_Document_Types AS T1": 1.00
    },
    "JOIN": {
        "Documents AS T2": 1.00
    },
    "ON": {
        "T1.Document_Type_Code = T2.Document_Type_Code": 1.00
    },
    "GROUP BY": {
        "Document_Type_Name": 1.00
    },
    "ORDER BY": {
        "Document_Date ASC": 0.50,
        "Document_Date DESC": 0.30,
        "Document_Type_Name ASC": 0.20
    },
    "BIN": {
        "Document_Date BY YEAR": 1.00
    }
}
```"""
content_prob = a.split("```", 2)[1].split("json", 1)[-1]
print(content_prob)
content_prob = json.loads(content_prob)
print(content_prob)
