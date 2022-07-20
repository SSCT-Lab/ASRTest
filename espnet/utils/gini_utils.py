import copy
import logging
import os
import json
import numpy as np
import math


class SpeechSort(object):
    def __init__(self, name, s_count, len_diff, mix_sort):
        self.name = name
        self.s_count = s_count
        self.len_diff = len_diff
        self.mix_sort = mix_sort


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


def read_wrd_trn(file_path):
    file = open(file_path, 'r')
    word_list = []
    line = file.readline()
    word_count = 0
    while line:
        idx = line.find("(")
        line = line[:idx]
        word_list.append(line.split(" "))
        word_count += len(line.split(" "))
        line = file.readline()
    print(word_count)
    return word_list, word_count


def load_token_file(file_path):
    f = open(file_path, "r")
    line = f.readline()
    token_dict = {}
    while line:
        line = line[:-1]
        split_line = line.split(" ")
        name = split_line[0]
        if "gini" in file_path:
            token_dict[name] = [float(gini) for gini in split_line[1:]]
        else:
            token_dict[name] = split_line[1:]
        line = f.readline()

    return token_dict

def get_word_gini(token_list, token_gini_list):
    all_word_list = {}
    all_gini_list = {}
    for name in token_list.keys():
        token = token_list[name]
        token_gini = token_gini_list[name]
        word_list = []
        gini_list = []
        word = ""
        word_gini = 0
        #c_count = 0
        for i in range(len(token)):
            if token[i] != token[0]:
                word = word + token[i]
                if token_gini[i] > word_gini:
                    word_gini = token_gini[i]
                #c_count += 1
            else:
                if len(word) != 0:
                    gini_list.append(word_gini)
                    word_list.append(word)
                    word = ""
                    #c_count = 0
                    word_gini = 0
        if len(word) != 0:
            gini_list.append(word_gini)
            word_list.append(word)
        all_word_list[name] = word_list
        all_gini_list[name] = gini_list

    return all_word_list, all_gini_list

def get_token_list(json_data):
    all_token = {}
    for name in json_data["utts"]:
        token = json_data["utts"][name]["output"][0]["rec_token"].split(" ")
        all_token[name.lower()] = token[:-1]
    return all_token
    

def get_word_gini_v1(json_data, token_gini_list, hyp_dict):
    all_word_list = {}
    all_gini_list = {}
    for name in json_data["utts"]:
        token = json_data["utts"][name]["output"][0]["rec_token"].split(" ")
        token_gini = token_gini_list[name.lower()]
        hyp_word = hyp_dict[name.lower()]
        word_list = list(hyp_word)
        for i in range(len(hyp_word)):
            if "*" in hyp_word[i]:
                word_list.remove(hyp_word[i])
        #word_list = []
        gini_list = []
        word = ""
        word_gini = 0
        w_idx = 0
        #c_count = 0
        for i in range(len(token)):
            if token[i] == "<eos>":
                break
            if (i == 0) & (token[i][0]!= "▁"):
                token[i] = "▁" + token[i]
            word = word + token[i]
            if token_gini[i] > word_gini:                
                word_gini = token_gini[i]
            if len(word) > 1:
                if word[0:2] == "▁▁":
                    word = word[1:]
                print(name)
                if word[1:] == word_list[w_idx].lower():
                    gini_list.append(word_gini)
                    w_idx += 1
                    word = ""
                    word_gini = 0
                    
        if w_idx != len(word_list):
            gini_list.append(word_gini)
        if len(word_list) != len(gini_list):
            print("name:", name)
            print("word, w_idx, word_gini", word, w_idx, word_gini)
            print(word_list, len(word_list))
            print(gini_list, len(gini_list))
        all_word_list[name.lower()] = word_list
        all_gini_list[name.lower()] = gini_list

    return all_word_list, all_gini_list


def get_gini_threshold(all_gini_list, eval_dict, orig_word_list):
    s_gini = 0
    i_gini = 0
    c_gini = 0
    s_count = 0
    i_count = 0
    c_count = 0
    d_count = 0
    for name in eval_dict.keys():
        if name not in all_gini_list.keys():
            gini_name = name.upper()
            gini_list = all_gini_list[gini_name]
        else:
            gini_list = all_gini_list[name]
        eval_list = eval_dict[name]
        for i in range(len(eval_list)):
            if eval_list[i] == "D":
                d_count += 1
                gini_list.insert(i, -1)
            if eval_list[i] == "S":
                s_gini += gini_list[i]
                s_count += 1
            if eval_list[i] == "I":
                #print("insert_gini:", name, gini_list[i])
                i_gini += gini_list[i]
                i_count += 1
            if eval_list[i] == "C":
                c_gini += gini_list[i]
                c_count += 1
    print(s_count, i_count, c_count, d_count)
    if s_count > i_count + d_count:
        flag = "s_first"
    else:
        flag = "len_first"
    if i_gini != 0:
        i_t = i_gini/i_count
    else:
        i_t = 0
    print("flag: ", flag)
    return s_gini/s_count, i_t, c_gini/c_count, flag


def judge_res(ref_word, hyp_word):
    eval_list = []
    for i in range(len(ref_word)):
        if ref_word[i] == hyp_word[i]:
            eval_list.append("C")
        elif "*" in ref_word[i]:
            eval_list.append("I")
        elif "*" in hyp_word[i]:
            eval_list.append("D")
        else:
            eval_list.append("S")
    return eval_list
 

def load_result_txt(file_path):
    f = open(file_path, 'r')
    ref_dict = {}
    hyp_dict = {}
    eval_dict = {}
    line = f.readline()
    while line:
        if "id: (" in line:
            name = line[:-1].replace("id: (", "").replace(")", "")
            name = "-".join(name.split("-")[1:])
        if "REF: " in line:
            ref = line[:-1].split()[1:]
            ref_dict[name] = ref
        if "HYP:  " in line:
            hyp = line[:-1].split()[1:]
            hyp_dict[name] = hyp
        if "Eval: " in line:
            eval_list = judge_res(ref, hyp)
            eval_dict[name] = eval_list
        line = f.readline()
    return ref_dict, hyp_dict, eval_dict


def gini_sort_v1(s_t, all_gini_list, orig_token, new_token, flag):
    print("flag is", flag)
    sort_list = []
    for key in new_token.keys():
        if "&" not in key:
            diff = 0
        else:
            orig_name = key.split("&")[0] + "-" + "-".join(key.split("&")[1].split("-")[1:])
            orig_len = len(orig_token[orig_name])
            new_len = len(new_token[key])
            diff = abs(new_len - orig_len)
        s_count = 0
        for gini in all_gini_list[key]:
            if gini > s_t:
                s_count += 1
        if len(all_gini_list[key]) == 0:
            sort_list.append(SpeechSort(key, 1.0, 1.0, 1.0))
        else:
            sort_list.append(SpeechSort(key, s_count/len(all_gini_list[key]), diff/len(all_gini_list[key]), (s_count + diff)/len(all_gini_list[key])))
    sort_list.sort(key=lambda x:x.mix_sort, reverse=True)

    return sort_list


def gini_sort(s_t, all_gini_list, orig_token, new_token, flag):
    sort_list = []
    s_count_list = []
    len_diff_list = []
    
    for key in new_token.keys():
        if "&" not in key:
            diff = 0
        else:
            orig_name = key.split("&")[0]
            orig_len = len(orig_token[orig_name])
            new_len = len(new_token[key])
            #diff = abs(new_len - orig_len)/orig_len
            diff = abs(new_len - orig_len)
        s_count = 0
        for gini in all_gini_list[key]:
            if gini > s_t:
                s_count += 1
        sort_list.append(SpeechSort(key, s_count/len(all_gini_list[key]), diff, (s_count + diff)/len(all_gini_list[key])))
    sort_list.sort(key=lambda x:x.mix_sort, reverse=True)

    return sort_list


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
    orig_gini = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if "sum_gini" in file_path:
            orig_gini[key] = float(line.split(" ")[1])
        else:
            if len(line[:-1].split(" ")[1:]) > 0:
                gini_list = [float(x) for x in line.split(" ")[1:]]
            else:
                gini_list = [0.0]
            orig_gini[key] = gini_list
        line = f.readline()
    return orig_gini


def load_gini_v1(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orig_gini = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0].lower()
        if "sum_gini" in file_path:
            orig_gini[key] = float(line.split(" ")[1])
        else:
            if len(line[:-1].split(" ")[1:]) > 0:
                gini_list = [float(x) for x in line.split(" ")[1:]]
            else:
                gini_list = [0.0]
            orig_gini[key] = gini_list
        line = f.readline()
    return orig_gini


def load_text(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orig_text = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orig_text[key] = line.split(" ")[1:]
        line = f.readline()
    return orig_text


def load_score(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orig_score = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orig_score[key] = line.split(" ")[1]
        line = f.readline()
    return orig_score


def load_token(file_path):
    f = open(file_path, 'r')
    line = f.readline()
    orig_token = {}
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orig_token[key] = line.split(" ")[1:]
        line = f.readline()
    return orig_token


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

        
def write_gini_select_result(new_dir, file_name, result):
    mkdir(new_dir)
    f = open(new_dir + "/" + file_name, 'w')
    for item in result:
        line = item.name + " " + str(item.len_diff) + " " + str(item.s_count)+ "\n"
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
    orig_dict = {}
    while line:
        value = str(line)
        line = line[:-1]
        line = line.replace("\t", " ")
        key = line.split(" ")[-1][1:-1]
        idx = key.find("-")
        key = key[idx + 1:]
        orig_dict[key] = value
        line = f.readline()
    return orig_dict


def write_hyp_ref(path, name, result):
    mkdir(path)
    f = open(path + "/" + name, 'w')
    for key in result.keys():
        f.write(result[key])
    f.close()
