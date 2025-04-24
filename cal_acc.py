import json
import pandas as pd
import src.trainer.metrics as metrics

cmp_path = "./data/{}/{}_cmp_cot_retrieval_sc.json"

mode = "dev_schema"

def process(text:str, result_type="pred"):
    # print(text)

    text = text.lower().replace("'", "\"").replace('", ', '",' ).replace("is not null", '!= \"null\"').replace(";", "").replace('"', '\"').replace(";", "").replace("=", " = ").replace("< =", " <= ").replace("> =", " >= ").replace("<>", " != ").replace("visualizevisualize", "visualize").replace("\"female\"", "\"f\"")
    text = text.split()
    
        

    text_new = []
    flag = False
    for id, token in enumerate(text):
        if token == 'visualize':
            flag = True
        if not flag:
            continue
        if "strftime" in token:
            token = token.replace("strftime(\"%w\",", "").replace("strftime(\"%y\",", "")[:-1]
        if "year(" in token:
            token = token.replace("year(", "")
        text_new.append(token)
    text = " ".join(text_new).replace(",", " , ").split()
        

    text_new = []
    rename = {}
    for id, token in enumerate(text):
        if token == "as":
            new_name = text[id + 1]
            ori_name = text[id - 1]
            rename[new_name] = ori_name
        # if token == 'from' and id < len(text) - 3:
        #     if text[id + 2] not in ['where', 'group', 'order', 'bin', 'having', 'limit']:
        #         new_name = text[id + 2]
        #         ori_name = text[id + 1]
        #         rename[new_name] = ori_name
    keywords = ['max', 'min', 'count', 'sum', 'avg']
    structure_tokens1 = ['visualize', 'select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'having', 'bin']
    VQL_dic = {}
    text_new = []
    flag = ""
    time = 0
    r = ""
    for token in text:
        if token.lower() in structure_tokens1:
            flag = token.lower()
            VQL_dic[flag] = []
        VQL_dic[flag].append(token)
    text_new = []
    for keyword in structure_tokens1:
        if keyword in VQL_dic:
            text_new.extend(VQL_dic[keyword])

    text = text_new

    text_new = []
    for id, token in enumerate(text):
        if token == "between":
            column = text[id - 1]
            value1 = text[id + 1]
            value2 = text[id + 3]
            text_new.append(">=")
            text_new.append(value1)
            text_new.append("and")
            text_new.append(column)
            text_new.append("<=")
            text_new.append(value2)
        if token == "between" or text[id - 1] == "between" or text[id - 2] == "between" or text[id - 3] == "between":
            continue
        text_new.append(token)
    text = text_new

    text_new = []
    for id, token in enumerate(text):
        if token == 'as':
            continue
        if text[id - 1].lower() == 'as':
            continue
        if token in rename:
            token = rename[token]
        # if token not in ['where', 'group', 'order', 'bin', 'having', 'limit'] and id > 2:
        #     if text[id - 2] == 'from':
        #         continue
        text_new.append(token)
    text = text_new

    text = " ".join(text).replace("(", " ( ").replace(")", " ) ").split()
    
    text_new = []
    for id, token in enumerate(text):
        if token.lower() in structure_tokens1:
            flag = token.lower()

        if token == 'as':
            continue
        if text[id - 1].lower() == 'as':
            continue
        if "." in token:
            tokens = token.split(".", 1)
            if tokens[0] in rename:
                tokens[0] = rename[tokens[0]]
            token = ".".join(tokens)

        text_new.append(token)
    text = text_new

    if 'join' not in text:
        text_new = []
        for id, token in enumerate(text):
            if token == 'as':
                continue
            if text[id - 1] == 'as':
                continue
            if '.' in token:
                token = token.split(".", 1)[1]
            text_new.append(token)

        text = text_new
    
    select_cnt = text.count('select')
    if select_cnt == 1 and 'join' not in text:
        text_new = []
        for id, token in enumerate(text):
            if '.' in token:
                token = token.split(".", 1)[1]

            text_new.append(token)
        text = text_new

    text_new = []
    flag = False
    for id, token in enumerate(text):
        if token == 'from':
            flag = True
        # if token.lower() == 'count' and id + 2 < len(text):
        #     text[id + 2] = '*' 
        elif token.lower() == 'asc':
            continue
        elif "." in token:
            token = token.split(".")[1]
        
        text_new.append(token)
    text = text_new

    

    text = " ".join(text).replace(" ( ", "(").replace(" ) ", ") ").replace(" )", ")")
    text = text.split()


    if "bin" in text :
        flag = False
        group_c = ""
        text_new = []
        for id, token in enumerate(text):
            if token == "group":
                group_c = text[id + 2]
            if token == "bin" and text[id + 1] == group_c:
                flag = True

        for id, token in enumerate(text):
            # if token == "group" or text[id - 1] == "group" or text[id - 2] == "group" and flag:
            #     continue
            if token == "time" and text[id - 3] == 'bin' and result_type=="pred":
                token = "month"
            text_new.append(token)

        text = text_new
    text = " ".join(text).replace("(", " ").replace(")"," ")
    while "  " in text:
        text = text.replace("  ", " ")

    # text = text.split()
    # text_new = []
    # flag_select2 = False
    # flag = ""
    # for id, token in enumerate(text):
    #     if token in structure_tokens1:
    #         if flag != "visualize" and token == "select":
    #             flag_select2 = True
    #         flag = token
    #     if token in keywords and flag == "select" and not flag_select2:
    #         if text[id - 1] == 'select' and text[id + 2] == 'from':
    #             text_new.append(text[id + 1])
    #     text_new.append(token)

    # text = " ".join(text_new)

    return text


cmp = pd.read_json(cmp_path.format(mode, mode))
question = cmp['question'].to_list()
preds_ori = cmp['pred_ori'].to_list()
tgts_ori = cmp['target'].to_list()

# with open(pred_path.format(mode, mode), 'r') as f:
#     preds_ori = json.load(f)

# with open(tgts_path.format(mode, mode), 'r') as f:
#     tgts_ori = json.load(f)

preds = [vql for vql in preds_ori]
tgts = [vql for vql in tgts_ori]

cmp = []
for q, pred, tgt, p_ori, t_ori in zip(question, preds, tgts, preds_ori,tgts_ori ):
    if pred == tgt:
        continue
    cmp.append({"question":q , "tgt":tgt, "pred":pred, "tgt_ori":t_ori,"pred_ori":p_ori,})
json.dump(cmp, open('./results_cmp.json', 'w'), indent=4)

metrics=metrics.Metrics()
acc_tree, acc_vis, acc_axis, acc_data = metrics.accuracy(preds, tgts, sql_type="vql")
acc_com = (acc_vis + acc_axis + acc_data) / 3
print(f"------------------------------------------{mode}---------------------------------------------------------")
print("data len: ", len(preds))
print('Test.  acc_tree: {:.4f}, acc_vis: {:.4f}, acc_axis: {:.4f}, acc_data: {:.4f}, acc_com: {:.4f},'.format(acc_tree, acc_vis, acc_axis, acc_data, acc_com))