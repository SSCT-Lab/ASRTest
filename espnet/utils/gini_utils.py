import copy
import logging
import os
import json
import numpy as np


def cau_sum_ginis(token_int, all_ginis):
    sum_gini = 0
    gini_list = []
    gini_key = ""
    if len(all_ginis) == 0:
        gini_list.append(0)
    else:
        for i in range(len(token_int)):
            token = token_int[i]
            gini_key += str(token) + " "
            if gini_key.strip() in all_ginis.keys():
                sum_gini += all_ginis[gini_key.strip()]
                gini_list.append(all_ginis[gini_key.strip()])
            else:
                key = gini_key.strip().split(" ")
                if len(key) > 1:
                    key = " ".join(key[:-1])
                else:
                    key = key[-1]
                sum_gini += all_ginis[key]
                gini_list.append(all_ginis[key])
    return sum_gini, gini_list


def cau_coverage(all_z_list, t):
    logging.info(f"the length of all_z_list is {len(all_z_list)}")
    count = 0
    sum_cell = 0
    for z_list in all_z_list:
        for i in range(len(z_list)):
            outputs = z_list[i]
            for o in outputs:
                if o > t:
                    count += 1
                sum_cell += 1
    logging.info(f"sum_cell is {sum_cell}, count is {count}")
    return count/sum_cell

def cau_coverage_v1(all_z_list, t):
    #logging.info(f"the length of all_z_list is {len(all_z_list)}")
    count = 0
    sum_cell = 0
    for state in all_z_list:
        #logging.info(f"state is {state}")
        for layer in state:
            lay_output = layer[0]
            #logging.info(f"layer is {lay_output}")
            for i in range(len(lay_output)):
                output = lay_output[i]
                #logging.info(f"output is {output}")
                for o in output:
                    if o > t:
                        count += 1
                    sum_cell += 1
    #logging.info(f"sum_cell is {sum_cell}, count is {count}")
    return count/sum_cell


def get_activate_cell(all_z_list, t):
    act_list = []
    for i in range(len(all_z_list)):
        for one_list in all_z_list[i]:
            for j in range(len(one_list)):
                if one_list[j] > t:
                    cell = (i,j)
                    act_list.append(cell)
    return act_list
                   
                   
        
def compare_avg_gini(sum_source, sum_new, count_s, count_n, T_a):
    #logging.info(f"the length of source is {len(source_ginis)}, the length of new is {len(new_ginis)}")
    source_avg = sum_source/count_s
    new_avg = sum_new/count_n
    if abs(source_avg - new_avg) > T_a:
        logging.info("over the T_a")
        return True
    else:
        return False

def get_avg_gini_diff(sum_source, sum_new, count_s, count_n):
    #logging.info(f"the length of source is {len(source_ginis)}, the length of new is {len(new_ginis)}")
    source_avg = sum_source/count_s
    new_avg = sum_new/count_n
    diff = new_avg - source_avg
    return diff    


def get_avg_gini(sum_new, count_n):
    new_avg = sum_new/count_n
    return new_avg


def compare_sum_gini(source_sum_gini, new_sum_gini, T_s):
    if abs(source_sum_gini - new_sum_gini) > T_s:
        return True
    else:
        return False


def caul_gini(softmax_value):
    gini = 0
    class_softmax = softmax_value.cpu().detach().numpy()
    #print("class_softmax:",class_softmax)
    for i in range(0, len(class_softmax)):
        gini += np.square(class_softmax[i])
    #print("gini value: ", 1-gini)
    return 1-gini


def parse_hypothesis(hyp, char_list, gini_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ")
    ginilist = " ".join([str(v) for v in gini_list])
    return text, token, tokenid, score, ginilist


def add_results_to_json(js, nbest_hyps, char_list, sum_gini, gini_list, cov):
    """Add N-best results to json.
    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]):
        List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js["utt2spk"] = js["utt2spk"]
    new_js["output"] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score, gini_list = parse_hypothesis(hyp, char_list, gini_list)

        # copy ground-truth
        if len(js["output"]) > 0:
            out_dic = dict(js["output"][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {"name": ""}

        # update name
        out_dic["name"] += "[%d]" % n

        # add recognition results
        out_dic["rec_text"] = rec_text
        out_dic["rec_token"] = rec_token
        out_dic["rec_tokenid"] = rec_tokenid
        out_dic["score"] = score
        out_dic["sum_gini"] = sum_gini
        out_dic["gini_list"] = gini_list
        out_dic["cov"] = cov

        # add to list of N-best result dicts
        new_js["output"].append(out_dic)

        # show 1-best result
        if n == 1:
            if "text" in out_dic.keys():
                logging.info("groundtruth: %s" % out_dic["text"])
            logging.info("prediction : %s" % out_dic["rec_text"])

    return new_js


def load_gini(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orgi_gini = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if "sum_gini" in file_path:
            orgi_gini[key] = float(line.split(" ")[1])
        else:
            if len(line[:-1].split(" ")[1:]) > 0:
                gini_list = [float(x) for x in line.split(" ")[1:]]
            else:
                gini_list = [0.0]
            orgi_gini[key] = gini_list
        line = f.readline()
    return orgi_gini


def load_cov(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    cov_dict = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        cov_dict[key] = line.split(" ")[1]
        line = f.readline()
    return cov_dict

def load_act_cell(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    act_cell = {}
    count = 1
    while line:
        print("current line: ", count)
        line = line[:-1]
        line = line.replace(", ", ",")
        key = line.split(" ")[0]
        act_cell[key] = line.split(" ")[1:]
        line = f.readline()
        count += 1
    return act_cell


def load_text(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orgi_text = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_text[key] = line.split(" ")[1:]
        line = f.readline()
    return orgi_text


def load_score(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orgi_score = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_score[key] = line.split(" ")[1]
        line = f.readline()
    return orgi_score


def load_token(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orgi_token = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_token[key] = line.split(" ")[1:]
        line = f.readline()
    return orgi_token


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def write_new_info(new_dir, file_name, result):
    mkdir(new_dir)
    f = open(new_dir + "/" + file_name, 'w')
    for key in result.keys():
        if isinstance(result[key], list):
            line = key + " " + " ".join([str(x) for x in result[key]]) + "\n"
        else:
            line = key + " " + str(result[key]) + "\n"
        f.write(line)


def update_data_json(json_data, key_name):
    new_json = copy.deepcopy(json_data)
    new_json["utts"].pop(key_name)
    return new_json


def load_data_json(path):
    with open(path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    return json_data

def write_json_data(path, new_json) :
    mkdir(path)
    with open(path + "/" + "data.json", "w", encoding='utf8')as fp:
        json.dump(new_json, fp, ensure_ascii=False)


def load_hyp_ref(path):
    f = open(path, 'r')
    line = f.readline()
    orgi_dict = {}
    while line:
        value = str(line)
        line = line[:-1]
        line = line.replace("\t", " ")
        key = line.split(" ")[-1][1:-1]
        idx = key.find("-")
        key = key[idx + 1:]
        orgi_dict[key] = value
        line = f.readline()
    return orgi_dict


def write_hyp_ref(path, name, result):
    mkdir(path)
    f = open(path + "/" + name, 'w')
    for key in result.keys():
        f.write(result[key])
    f.close()