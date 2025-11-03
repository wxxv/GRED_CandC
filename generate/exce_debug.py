import openai
import json
import sqlite3
import os
import time
from tqdm import tqdm

# 替换为你的OpenAI API的base URL
api_base = ""

# 替换为你的OpenAI API密钥
api_key = ""

# 定义Openai Client
client = openai.Client(api_key=api_key, base_url=api_base)

SYSTEM_PROMPT = """You are now a query optimization module in the natural language interface of a data visualization system, responsible for optimizing data visualization queries based on natural language queries and database schemas.
Supplementary knowledge for data visualization queries is as follows:
- Data visualization queries use SQL statements to retrieve the data for the X-axis and Y-axis displayed in the visualization chart.
- Data visualization queries specify the type of chart to visualize using the `Visualize` keyword, such as `Visualize BAR / PIE / LINE / SCATTER`.
- When data visualization queries need to group data by time, they use the `BIN ... BY ...` clause in the end of the query rather than `strftime()` function, such as `BIN HIRE_DATE BY MONTH` to group HIRE_DATE by month."""

INSTRUCTION = """#### Given database schemas, a natural language query, an original data visualization query, and its execution results, please perform the following actions:
1. Analyze whether the original data visualization query matches the natural language query based on the database schemas:
   match, modify the original data visualization query based on the current database schemas;
2. Output the final data visualization query in the following format:
```sql
Visualize ...
```"""

ICL_PROMPT = """### Database Schemas
CREATE TABLE regions (
  REGION_ID decimal(5,0) NOT NULL,
  REGION_NAME varchar(25) DEFAULT NULL,
  PRIMARY KEY (REGION_ID)
);
insert into regions (REGION_ID, REGION_NAME) values (1, 'Europe\r') ;

CREATE TABLE countries (
  COUNTRY_ID varchar(2) NOT NULL,
  COUNTRY_NAME varchar(40) DEFAULT NULL,
  REGION_ID decimal(10,0) DEFAULT NULL,
  PRIMARY KEY (COUNTRY_ID),
  FOREIGN KEY (REGION_ID) REFERENCES regions (REGION_ID)
);
insert into countries (COUNTRY_ID, COUNTRY_NAME, REGION_ID) values ('AR', 'Argentina', 2) ;

CREATE TABLE departments (
  DEPARTMENT_ID decimal(4,0) NOT NULL DEFAULT 0,
  DEPARTMENT_NAME varchar(30) NOT NULL,
  MANAGER_ID decimal(6,0) DEFAULT NULL,
  LOCATION_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (DEPARTMENT_ID)
);
insert into departments (DEPARTMENT_ID, DEPARTMENT_NAME, MANAGER_ID, LOCATION_ID) values (10, 'Administration', 200, 1700) ;

CREATE TABLE jobs (
  JOB_ID varchar(10) NOT NULL DEFAULT ,
  JOB_TITLE varchar(35) NOT NULL,
  MIN_SALARY decimal(6,0) DEFAULT NULL,
  MAX_SALARY decimal(6,0) DEFAULT NULL,
  PRIMARY KEY (JOB_ID)
);
insert into jobs (JOB_ID, JOB_TITLE, MIN_SALARY, MAX_SALARY) values ('AD_PRES', 'President', 20000, 40000) ;

CREATE TABLE employees (
  EMPLOYEE_ID decimal(6,0) NOT NULL DEFAULT 0,
  FIRST_NAME varchar(20) DEFAULT NULL,
  LAST_NAME varchar(25) NOT NULL,
  EMAIL varchar(25) NOT NULL,
  PHONE_NUMBER varchar(20) DEFAULT NULL,
  HIRE_DATE date NOT NULL,
  JOB_ID varchar(10) NOT NULL,
  SALARY decimal(8,2) DEFAULT NULL,
  COMMISSION_PCT decimal(2,2) DEFAULT NULL,
  MANAGER_ID decimal(6,0) DEFAULT NULL,
  DEPARTMENT_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (EMPLOYEE_ID),
  FOREIGN KEY (DEPARTMENT_ID) REFERENCES departments(DEPARTMENT_ID),
  FOREIGN KEY (JOB_ID) REFERENCES jobs(JOB_ID)
);
insert into employees (EMPLOYEE_ID, FIRST_NAME, LAST_NAME, EMAIL, PHONE_NUMBER, HIRE_DATE, JOB_ID, SALARY, COMMISSION_PCT, MANAGER_ID, DEPARTMENT_ID) values (100, 'Steven', 'King', 'SKING', '515.123.4567', '1987-06-17', 'AD_PRES', 24000, 0, 0, 90) ;

CREATE TABLE job_history (
  EMPLOYEE_ID decimal(6,0) NOT NULL,
  START_DATE date NOT NULL,
  END_DATE date NOT NULL,
  JOB_ID varchar(10) NOT NULL,
  DEPARTMENT_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (EMPLOYEE_ID,START_DATE),
  FOREIGN KEY (EMPLOYEE_ID) REFERENCES employees(EMPLOYEE_ID),
  FOREIGN KEY (DEPARTMENT_ID) REFERENCES departments(DEPARTMENT_ID),
  FOREIGN KEY (JOB_ID) REFERENCES jobs(JOB_ID)
);
insert into job_history (EMPLOYEE_ID, START_DATE, END_DATE, JOB_ID, DEPARTMENT_ID) values (102, '1993-01-13', '1998-07-24', 'IT_PROG', 60) ;

CREATE TABLE locations (
  LOCATION_ID decimal(4,0) NOT NULL DEFAULT 0,
  STREET_ADDRESS varchar(40) DEFAULT NULL,
  POSTAL_CODE varchar(12) DEFAULT NULL,
  CITY varchar(30) NOT NULL,
  STATE_PROVINCE varchar(25) DEFAULT NULL,
  COUNTRY_ID varchar(2) DEFAULT NULL,
  PRIMARY KEY (LOCATION_ID),
  FOREIGN KEY (COUNTRY_ID) REFERENCES countries(COUNTRY_ID)
);
insert into locations (LOCATION_ID, STREET_ADDRESS, POSTAL_CODE, CITY, STATE_PROVINCE, COUNTRY_ID) values (1000, '1297 Via Cola di Rie', '989', 'Roma', '', 'IT') ;

### Natural Language Query: For employees who are not working in departments where managers have IDs between 100 and 200, create a bar chart depicting the distribution of job IDs and manager IDs, listed in descending order on the Y-axis.
### Original Data Visualization Query: Visualize BAR SELECT JOB_ID , MANAGER_ID FROM employees WHERE department_id NOT IN (SELECT department_id FROM departments WHERE manager_id BETWEEN 100 AND 200) ORDER BY MANAGER_ID DESC
### Execution Results:
## Chart Type: BAR
## Chart Data:
| X-axis | Y-axis |
| --- | --- |
| AC_ACCOUNT | 205 |
| MK_REP | 201 |

A: Let’s think step by step!

### **Step-by-Step** Analysis

1. **Natural Language Query**: "For employees who are not working in departments where managers have IDs between 100 and 200, create a bar chart depicting the distribution of job IDs and manager IDs, listed in descending order on the Y-axis."
2. **Original Data Visualization Query**:
   ```sql
   Visualize BAR SELECT JOB_ID, MANAGER_ID FROM employees WHERE department_id NOT IN (SELECT department_id FROM departments WHERE manager_id BETWEEN 100 AND 200) ORDER BY MANAGER_ID DESC
   ```
3. **Database Schema**:
   - The `employees` table has columns `EMPLOYEE_ID`, `FIRST_NAME`, `LAST_NAME`, `EMAIL`, `PHONE_NUMBER`, `HIRE_DATE`, `JOB_ID`, `SALARY`, `COMMISSION_PCT`, `MANAGER_ID`, `DEPARTMENT_ID`.
   - The `departments` table has columns `DEPARTMENT_ID`, `DEPARTMENT_NAME`, `MANAGER_ID`, `LOCATION_ID`.

### Verification

- The natural language query specifies a bar chart for employees not in departments managed by IDs between 100 and 200, showing job IDs and manager IDs, ordered by manager IDs in descending order.
- The original data visualization query selects `JOB_ID` and `MANAGER_ID` from the `employees` table, filtering out departments with managers having IDs between 100 and 200, and orders the results by `MANAGER_ID` in descending order.

### Conclusion

The original data visualization query matches the natural language query. It correctly filters employees based on the department manager's ID, selects the required columns, and orders the results as specified.

### Final Data Visualization Query

```sql
Visualize BAR SELECT JOB_ID, MANAGER_ID FROM employees WHERE department_id NOT IN (SELECT department_id FROM departments WHERE manager_id BETWEEN 100 AND 200) ORDER BY MANAGER_ID DESC
```


### Database Schemas
CREATE TABLE regions (
  REGIONID decimal(5,0) NOT NULL,
  REGIONNAME varchar(25) DEFAULT NULL,
  PRIMARY KEY (REGIONID)
);
insert into regions (REGIONID, REGIONNAME) values (1, 'Europe\r') ;

CREATE TABLE countries (
  COUNTRYID varchar(2) NOT NULL,
  COUNTRYNAME varchar(40) DEFAULT NULL,
  REGIONID decimal(10,0) DEFAULT NULL,
  PRIMARY KEY (COUNTRYID),
  FOREIGN KEY (REGIONID) REFERENCES regions (REGIONID)
);
insert into countries (COUNTRYID, COUNTRYNAME, REGIONID) values ('AR', 'Argentina', 2) ;

CREATE TABLE departments (
  Dept_ID decimal(4,0) NOT NULL DEFAULT 0,
  Dept_NAME varchar(30) NOT NULL,
  Manager_ID decimal(6,0) DEFAULT NULL,
  Location_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (Dept_ID)
);
insert into departments (Dept_ID, Dept_NAME, Manager_ID, Location_ID) values (10, 'Administration', 200, 1700) ;

CREATE TABLE jobs (
  JOB_ID varchar(10) NOT NULL DEFAULT ,
  JOB_TITLE varchar(35) NOT NULL,
  minimum_salary decimal(6,0) DEFAULT NULL,
  maximum_salary decimal(6,0) DEFAULT NULL,
  PRIMARY KEY (JOB_ID)
);
insert into jobs (JOB_ID, JOB_TITLE, minimum_salary, maximum_salary) values ('AD_PRES', 'President', 20000, 40000) ;

CREATE TABLE employees (
  employee_id decimal(6,0) NOT NULL DEFAULT 0,
  Fname varchar(20) DEFAULT NULL,
  Lname varchar(25) NOT NULL,
  Email_address varchar(25) NOT NULL,
  phone_number varchar(20) DEFAULT NULL,
  date_of_hire date NOT NULL,
  JOB_ID varchar(10) NOT NULL,
  wage decimal(8,2) DEFAULT NULL,
  COMMISSION_PCT decimal(2,2) DEFAULT NULL,
  Manager_ID decimal(6,0) DEFAULT NULL,
  Dept_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (employee_id),
  FOREIGN KEY (Dept_ID) REFERENCES departments(Dept_ID),
  FOREIGN KEY (JOB_ID) REFERENCES jobs(JOB_ID)
);
insert into employees (employee_id, Fname, Lname, Email_address, phone_number, date_of_hire, JOB_ID, wage, COMMISSION_PCT, Manager_ID, Dept_ID) values (100, 'Steven', 'King', 'SKING', '515.123.4567', '1987-06-17', 'AD_PRES', 24000, 0, 0, 90) ;

CREATE TABLE job_history (
  employee_id decimal(6,0) NOT NULL,
  START_DATE date NOT NULL,
  END_DATE date NOT NULL,
  JOB_ID varchar(10) NOT NULL,
  Dept_ID decimal(4,0) DEFAULT NULL,
  PRIMARY KEY (employee_id,START_DATE),
  FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
  FOREIGN KEY (Dept_ID) REFERENCES departments(Dept_ID),
  FOREIGN KEY (JOB_ID) REFERENCES jobs(JOB_ID)
);
insert into job_history (employee_id, START_DATE, END_DATE, JOB_ID, Dept_ID) values (102, '1993-01-13', '1998-07-24', 'IT_PROG', 60) ;

CREATE TABLE locations (
  Location_ID decimal(4,0) NOT NULL DEFAULT 0,
  STREET_ADDRESS varchar(40) DEFAULT NULL,
  POSTAL_CODE varchar(12) DEFAULT NULL,
  CITY varchar(30) NOT NULL,
  STATE_PROVINCE varchar(25) DEFAULT NULL,
  COUNTRYID varchar(2) DEFAULT NULL,
  PRIMARY KEY (Location_ID),
  FOREIGN KEY (COUNTRYID) REFERENCES countries(COUNTRYID)
);
insert into locations (Location_ID, STREET_ADDRESS, POSTAL_CODE, CITY, STATE_PROVINCE, COUNTRYID) values (1000, '1297 Via Cola di Rie', '989', 'Roma', '', 'IT') ;

### Natural Language Query: For the employees who are not employed in departments where managers have IDs between 100 and 200, present a bar chart illustrating the distribution of hire dates and the sum of salary grouped by weekdays. Please arrange the chart from the least to the greatest on the Y-axis.
### Original Data Visualization Query: Visualize BAR SELECT HIRE_DATE , SUM(SALARY) FROM employees WHERE Dept_ID NOT IN (SELECT Dept_ID FROM departments WHERE Manager_ID BETWEEN 100 AND 200)  ORDER BY SUM(SALARY) ASC BIN HIRE_DATE BY WEEKDAY
### Execution Results:
## Chart Type: BAR
## Chart Data:
Execution Error: no such column: HIRE_DATE, while executing query: SELECT strftime('%w', HIRE_DATE) as HIRE_DATE , SUM(SALARY) FROM employees WHERE Dept_ID NOT IN (SELECT Dept_ID FROM departments WHERE Manager_ID BETWEEN 100 AND 200)  ORDER BY SUM(SALARY) ASC

A: Let’s think step by step!

### **Step-by-Step Analysis**

1. **Natural Language Query**: "For the employees who are not employed in departments where managers have IDs between 100 and 200, present a bar chart illustrating the distribution of hire dates and the sum of salary grouped by weekdays. Please arrange the chart from the least to the greatest on the Y-axis."

2. **Original Data Visualization Query**:
   ```sql
   Visualize BAR SELECT HIRE_DATE , SUM(SALARY) FROM employees WHERE Dept_ID NOT IN (SELECT Dept_ID FROM departments WHERE Manager_ID BETWEEN 100 AND 200) ORDER BY SUM(SALARY) ASC BIN HIRE_DATE BY WEEKDAY   
   ```

3. **Database Schema**:
   - The `employees` table has columns `employee_id`, `Fname`, `Lname`, `Email_address`, `phone_number`, `date_of_hire`, `JOB_ID`, `wage`, `COMMISSION_PCT`, `Manager_ID`, `Dept_ID`.
   - The `departments` table has columns `Dept_ID`, `Dept_NAME`, `Manager_ID`, `Location_ID`.

### Verification

- The natural language query specifies a bar chart for employees not in departments managed by IDs between 100 and 200, showing the distribution of hire dates and the sum of salary grouped by weekdays, ordered by the sum of salary in ascending order.
- The original data visualization query attempts to select `HIRE_DATE` and `SUM(SALARY)` from the `employees` table, filtering out departments with managers having IDs between 100 and 200, and orders the results by `SUM(SALARY)` in ascending order. It also attempts to group `HIRE_DATE` by weekdays.

### Issue Identification

- The execution error indicates that there is no column named `HIRE_DATE` in the `employees` table. Instead, the correct column name is `date_of_hire`.
- The query should use `wage` instead of `SALARY` as the column name for salary in the `employees` table.

### Conclusion

The original data visualization query needs modification to use the correct column names from the `employees` table.

### Final Data Visualization Query

```sql
Visualize BAR SELECT date_of_hire , SUM(wage) FROM employees WHERE Dept_ID NOT IN (SELECT Dept_ID FROM departments WHERE Manager_ID BETWEEN 100 AND 200)  ORDER BY SUM(wage) ASC BIN date_of_hire BY WEEKDAY
```"""

def parse_dvq(dvq:str):
    """解析DVQ的函数，接收DVQ作为输入参数。"""
    try:
        sql = "SELECT " + dvq.split("SELECT", 1)[1].strip()
        chart_type = dvq.split("SELECT", 1)[0].split()[-1].strip()
    except:
        sql = "SELECT " + dvq.split("select", 1)[1].strip()
        chart_type = dvq.split("select", 1)[0].split()[-1].strip()

    if "BIN" in sql:
        sql = sql.split("BIN", 1)[0].strip()
        bin_column = dvq.split("BIN", 1)[1].split("BY", 1)[0].strip()
        bin_time = dvq.split("BIN", 1)[1].split("BY", 1)[-1].strip()
        bin = {"column": bin_column, "time": bin_time}
    else:
        bin = None
    return chart_type, sql, bin

def execute_sql(sql:str, db_id:str, bin:dict):
    """执行SQL语句的函数，接收SQL语句作为输入参数。"""
    db_folder = "./database/"
    db_path = os.path.join(db_folder, db_id + ".sqlite")

    if bin:
        # 将bin语句添加到SQL语句中
        bin_column = bin["column"]
        bin_time = bin["time"]
        
        # 将bin_time转换成时间戳，如bin_time为year，则转换成'%Y'
        if bin_time.lower() == "year":
            bin_time = "%Y"
        elif bin_time.lower() == "month":
            bin_time = "%Y-%m"
        elif bin_time.lower() == "day":
            bin_time = "%Y-%m-%d"
        elif bin_time.lower() == "weekday":
            bin_time = "%w"
        elif bin_time.lower() == "hour":
            bin_time = "%Y-%m-%d %H"
        else:
            bin_time = "%Y-%m-%d %H:%M:%S"

        sql = sql.replace(bin_column, f"strftime('{bin_time}', {bin_column}) as {bin_column}")
    else:
        pass

    try:
        # 执行SQL语句
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        # 将SQL执行结果转换为Markdown格式
        if len(results) > 2:
            results = results[:2]
        results_md = "| X-axis | Y-axis |\n| --- | --- |\n"
        for result in results:
            results_md += "| " + " | ".join(str(i) for i in result) + " |\n"
    except Exception as e:
        results_md = "Execution Error: " + str(e) + ", while executing query: " + sql

    return results_md

def get_schemas(db_id: str):
    """
    获取数据库的表结构信息的函数，接收数据库ID作为输入参数。
    最终需要输出数据库中所有table的CREATE TABLE语句以及一条INSERT语句。如下：
    CREATE TABLE customers (
        customer_id number ,
        customer_name text ,
        customer_details text ,
        primary key ( customer_id )
    )
    insert into customers (customer_id, customer_name, customer_details) values (1, ’Savannah’, ’rerum’) ;

    CREATE TABLE invoices (
        invoice_number number ,
        invoice_date time ,
        invoice_details text ,
        primary key ( invoice_number )
    )
    insert into invoices (invoice_number, invoice_date, invoice_details) values (1, ’1989-09-03 16:03:05’, ’vitae’) ;
    ...
    """
    schemas_str = ""
    conn = sqlite3.connect(f"./database/{db_id}.sqlite")
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        # Get the CREATE TABLE statement
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_table_sql = cursor.fetchone()[0].replace('"', "").replace("'", "").replace("`", "").replace("\t", " ")
        schemas_str += create_table_sql + ";\n"

        # Get one row of data from the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
        row = cursor.fetchone()

        if row:
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [info[1] for info in cursor.fetchall()]

            # Format the INSERT statement
            values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
            insert_sql = f"insert into {table_name} ({', '.join(columns)}) values ({values}) ;"
            schemas_str += insert_sql + "\n\n"
        else:
            schemas_str += "\n"

    conn.close()

    schemas_str = schemas_str.strip("\n").strip()
    return schemas_str

def prompt_maker(db_id:str, nlq:str, dvq:str):
    """生成提示信息的函数，接收数据库ID和DVQ作为输入参数。"""

    chart_type, sql, bin = parse_dvq(dvq)
    results_sql = execute_sql(sql, db_id, bin).strip()

    if chart_type.lower() in ["line", "bar", "scatter", "pie"]:
        result_chart = chart_type.upper()
    else:
        result_chart = "Unknown chart type"

    prompt = f"""{INSTRUCTION}

{ICL_PROMPT}


### Database Schemas
{get_schemas(db_id)}

### Natural Language Query: {nlq}
### Original Data Visualization Query: {dvq}
### Execution Results:
## Chart Type: {result_chart}
## Chart Data:
{results_sql}

A: Let’s think step by step! 
"""

    return prompt


def exec_debug(db_id:str, nlq, dvq:str, model="gpt-3.5-turbo-0125"):

    prompt = prompt_maker(db_id, nlq, dvq)

    # print(f"Prompt: {prompt}")

    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content":prompt
        }
    ]

    response = None
    while not response:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n = 1,
                stream = False,
                temperature=0.0
            )
        except:
            print("Error: Failed to generate feedback. Retrying...")
            time.sleep(10)

    # print(f"{response.choices[0].message.content}")
    dvq = response.choices[0].message.content.rsplit("```sql",1)[1].split("```",1)[0].replace("\n", " ")
    while "  " in dvq:
        dvq = dvq.replace("  ", " ")
    dvq = dvq.strip()
    # print(f"Feedback: {dvq}")
    return dvq

if __name__ == "__main__":
    file_path = "./data/{}/{}_result_nlq_rag.json"
    save_path = "./self-debug/data/{}/{}_exec_debug.json"

    if not os.path.exists("./self-debug/data/"):
        os.makedirs("./self-debug/data/")

    for mode in ['dev_nlq_schema', 'dev_nlq', 'dev_schema']:
        with open(file_path.format(mode, mode), 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_new = []
        if os.path.exists(save_path.format(mode, mode)):
            with open(save_path.format(mode, mode), 'r', encoding='utf-8') as f:
                data_new = json.load(f)

        for i, example in tqdm(enumerate(data), total=len(data), desc=f"Generating Feedback in {mode} mode"):
            if i < len(data_new):
                continue

            nlq = example['nlq']
            db_id = example['db_id']
            dvq = example['predict_rag_nlq']
            feedback = exec_debug(db_id, nlq, dvq)
            example['feedback'] = feedback
            data_new.append(example)

            if not os.path.exists("./self-debug/data/{}".format(mode)):
                os.makedirs("./self-debug/data/{}".format(mode))
        
            if i % 20 == 0:
                with open(save_path.format(mode, mode), 'w', encoding='utf-8') as f:
                    json.dump(data_new, f, indent=4)

        with open(save_path.format(mode, mode), 'w', encoding='utf-8') as f:
            json.dump(data_new, f, indent=4)


