import copy

from espnet.utils.gini_utils import *
import random
import os
import shutil


def sort_test_by_avg_gini_diff(old_dir, new_dir, aug_type, selected_num):
#     orgi_sum_gini = load_gini(old_dir + "/sum_gini")
#     orgi_token_gini = load_gini(old_dir + "/token_gini")
    new_token = load_token(new_dir + "/token")
    new_token_int = load_token(new_dir + "/token_int")
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_text = load_text(new_dir + "/text")
    new_score = load_score(new_dir + "/score")
    new_wer_ref = load_hyp_ref(new_dir + "/score_wer/ref.trn")
    new_wer_hyp = load_hyp_ref(new_dir + "/score_wer/hyp.trn")
    new_cer_ref = load_hyp_ref(new_dir + "/score_cer/ref.trn")
    new_cer_hyp = load_hyp_ref(new_dir + "/score_cer/hyp.trn")
    folder = os.path.exists(new_dir + "/score_ter/")
    if folder:
        new_ter_ref = load_hyp_ref(new_dir + "/score_ter/ref.trn")
        new_ter_hyp = load_hyp_ref(new_dir + "/score_ter/hyp.trn")
        ter_ref_result = dict(new_ter_ref)
        ter_hyp_result = dict(new_ter_hyp)
    token_result = dict()
    token_int_result = dict()
    token_gini_result = dict()
    sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()
    gini_diff = dict()
    for key in new_sum_gini.keys():
        orgi_name = key.split("&")[0]
#         diff = get_avg_gini_diff(orgi_sum_gini[orgi_name], new_sum_gini[key], len(orgi_token_gini[orgi_name]),
#                                 len(new_token_gini[key]))
        diff = new_sum_gini[key]/len(new_token_gini[key])
        gini_diff[key] = diff
    
    sort_result = sort_type_by_diff(aug_type, selected_num, gini_diff)
    
    for item in sort_result:
        key = item[0]
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]
        text_result[key] = new_text[key]
        score_result[key] = new_score[key]
        wer_hyp_result[key] = new_wer_hyp[key]
        wer_ref_result[key] = new_wer_ref[key]
        cer_hyp_result[key] = new_cer_hyp[key]
        cer_ref_result[key] = new_cer_ref[key]
        if folder:
            ter_hyp_result[key] = new_ter_hyp[key]
            ter_ref_result[key] = new_ter_ref[key]

    result_dir = new_dir + "/gini-" + str(selected_num)
    write_new_info(result_dir, "text", text_result)
    write_new_info(result_dir, "score", score_result)
    write_new_info(result_dir, "token", token_result)
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "token_int", token_int_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return sum_gini_result


def sort_test_by_avg_gini_v1(old_dir, new_dir, aug_type):
    orgi_sum_gini = load_gini(old_dir + "/sum_gini")
#     orgi_token_gini = load_gini(old_dir + "/token_gini")
    selected_num = len(orgi_sum_gini)
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")

    json_data = load_data_json(new_dir + "/data.json")
    new_json = copy.deepcopy(json_data)
    token_gini_result = dict()
    sum_gini_result = dict()
    gini_diff = dict()

    for key in json_data["utts"]:
        diff = new_sum_gini[key]/len(new_token_gini[key])
        gini_diff[key] = diff

    sort_result = sort_type_by_diff_v1(aug_type, selected_num, gini_diff)

    for item in sort_result:
        key = item[0]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]

    for key in json_data["utts"]:
        if key not in token_gini_result.keys():
            del new_json["utts"][key]

    result_dir = new_dir + "/gini-" + str(selected_num) 
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_json_data(result_dir, new_json)
    return sum_gini_result


def sort_test_by_cov_v1(old_dir, new_dir, aug_type):
    orgi_sum_gini = load_gini(old_dir + "/sum_gini")
    selected_num = len(orgi_sum_gini)
    new_cov = load_cov(new_dir + "/cov")
    json_data = load_data_json(new_dir + "/data.json")
    new_json = copy.deepcopy(json_data)
    cov_result = dict()
    
    sort_result = sort_type_by_diff_v1(aug_type, selected_num, new_cov)

    for item in sort_result:
        key = item[0]
        cov_result[key] = new_cov[key]
        
    for key in json_data["utts"]:
        if key not in cov_result.keys():
            del new_json["utts"][key]

    result_dir = new_dir + "/cov-" + str(selected_num) 
    write_new_info(result_dir, "cov", cov_result)
    write_json_data(result_dir, new_json)
    return cov_result
    
def random_samples_v1(old_dir, new_dir):
    orgi_sum_gini = load_gini(old_dir + "/sum_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    selected_num = len(orgi_sum_gini)
    json_data = load_data_json(new_dir + "/data.json")
    new_json = copy.deepcopy(json_data)
    select_keys = random.sample(list(new_sum_gini.keys()), selected_num)
    random_result = dict()
    for key in select_keys:
        random_result[key] = key
    for key in json_data["utts"]:
        if key not in select_keys:
            del new_json["utts"][key]            
    result_dir = new_dir + "/random-" + str(selected_num)
    write_json_data(result_dir, new_json) 
    return random_result
    
    
    
def sort_type_by_diff(aug_type, selected_num, diffs):
    class_dict = {}
    for key in diffs.keys():
        seq = key.split("&")[1].replace(".wav", "")
        if ("f" in seq) | ("n" in seq) | ("a" in seq):
            seq = key.split("&")[1].replace(".wav", "")[-1]
        if aug_type == "feature":
            seq = int(seq)
        if aug_type == "noise":
            seq = int(seq) % 8
        if aug_type == "room":
            seq = int(int(seq)/ 4)            
        if seq not in class_dict.keys():
            class_dict[seq] = list()
        item = (key, diffs[key])
        class_dict[seq].append(item)
        
    result = []
    num = int(selected_num/len(class_dict.keys()))
    for key in class_dict.keys():
        sort_result =  sorted(class_dict[key], reverse=True, key=lambda kv:(kv[1], kv[0]))
        sort_result = sort_result[:num]
        result = result + sort_result
        
    while len(result) < selected_num:
        key = random.choice(list(diffs.keys()))
        item = (key,diffs[key])
        if item not in result:
            result.append(item)
    total_gini = 0
    for item in result:
        total_gini += item[1]
    print(aug_type, "selects ", total_gini/len(result))
     
    return result


def sort_type_by_diff_v1(aug_type, selected_num, diffs):
    class_dict = {}
    for key in diffs.keys():
        seq = key.split("&")[1].replace(".wav", "").split("-")[0]
        if ("f" in seq) | ("n" in seq) | ("a" in seq):
            seq = key.split("&")[1].replace(".wav", "")[-1]
        if aug_type == "feature":
            seq = int(seq)
        if aug_type == "noise":
            seq = int(seq) % 8
        if aug_type == "room":
            seq = int(int(seq)/ 4)            
        if seq not in class_dict.keys():
            class_dict[seq] = list()
        item = (key, float(diffs[key]))
        class_dict[seq].append(item)
        
    result = []
    num = int(selected_num/len(class_dict.keys()))
    for key in class_dict.keys():
        sort_result =  sorted(class_dict[key], reverse=True, key=lambda kv:(kv[1], kv[0]))
        sort_result = sort_result[:num]
        result = result + sort_result
        
    while len(result) < selected_num:
        key = random.choice(list(diffs.keys()))
        item = (key,diffs[key])
        if item not in result:
            result.append(item)
    total_gini = 0
    for item in result:
        total_gini += float(item[1])
    print(aug_type, "selects ", total_gini/len(result))
    return result
    
def sort_by_cov(new_dir, selected_num):
    new_token = load_token(new_dir + "/token")
    new_token_int = load_token(new_dir + "/token_int")
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_text = load_text(new_dir + "/text")
    new_score = load_score(new_dir + "/score")
    new_cov = load_cov(new_dir + "/cov")
    new_wer_ref = load_hyp_ref(new_dir + "/score_wer/ref.trn")
    new_wer_hyp = load_hyp_ref(new_dir + "/score_wer/hyp.trn")
    new_cer_ref = load_hyp_ref(new_dir + "/score_cer/ref.trn")
    new_cer_hyp = load_hyp_ref(new_dir + "/score_cer/hyp.trn")
    folder = os.path.exists(new_dir + "/score_ter/")
    if folder:
        new_ter_ref = load_hyp_ref(new_dir + "/score_ter/ref.trn")
        new_ter_hyp = load_hyp_ref(new_dir + "/score_ter/hyp.trn")
        ter_ref_result = dict(new_ter_ref)
        ter_hyp_result = dict(new_ter_hyp)
    cov_result = dict()
    token_result = dict()
    token_int_result = dict()
    token_gini_result = dict()
    sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()
    
    sort_cov = sorted(new_cov.items(), reverse=True, key=lambda kv:(kv[1], kv[0]))
    sort_cov = sort_cov[: selected_num]

    for item in sort_cov:
        key = item[0]
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]
        text_result[key] = new_text[key]
        score_result[key] = new_score[key]
        wer_hyp_result[key] = new_wer_hyp[key]
        wer_ref_result[key] = new_wer_ref[key]
        cer_hyp_result[key] = new_cer_hyp[key]
        cer_ref_result[key] = new_cer_ref[key]
        cov_result[key] = new_cov[key]
        if folder:
            ter_hyp_result[key] = new_ter_hyp[key]
            ter_ref_result[key] = new_ter_ref[key]

    result_dir = new_dir + "/cov-" + str(selected_num)
    write_new_info(result_dir, "text", text_result)
    write_new_info(result_dir, "score", score_result)
    write_new_info(result_dir, "token", token_result)
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "token_int", token_int_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_new_info(result_dir, "cov", cov_result)
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return cov_result

def sort_test_by_cov_diff(old_dir, new_dir, aug_type, selected_num):
    old_cov = load_cov(old_dir + "/cov")
    new_token = load_token(new_dir + "/token")
    new_token_int = load_token(new_dir + "/token_int")
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_text = load_text(new_dir + "/text")
    new_score = load_score(new_dir + "/score")
    new_cov = load_cov(new_dir + "/cov")
    new_wer_ref = load_hyp_ref(new_dir + "/score_wer/ref.trn")
    new_wer_hyp = load_hyp_ref(new_dir + "/score_wer/hyp.trn")
    new_cer_ref = load_hyp_ref(new_dir + "/score_cer/ref.trn")
    new_cer_hyp = load_hyp_ref(new_dir + "/score_cer/hyp.trn")
    folder = os.path.exists(new_dir + "/score_ter/")
    if folder:
        new_ter_ref = load_hyp_ref(new_dir + "/score_ter/ref.trn")
        new_ter_hyp = load_hyp_ref(new_dir + "/score_ter/hyp.trn")
        ter_ref_result = dict(new_ter_ref)
        ter_hyp_result = dict(new_ter_hyp)
    cov_result = dict()
    token_result = dict()
    token_int_result = dict()
    token_gini_result = dict()
    sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()

    cov_diff = dict()
    for key in new_cov.keys():
        orgi_name = key.split("&")[0]
#         diff = float(old_cov[orgi_name]) - float(new_cov[key])
        diff = float(new_cov[key])
        cov_diff[key] = diff
   
    sort_result = sort_type_by_diff(aug_type, selected_num, cov_diff)
    
    for item in sort_result:
        key = item[0]
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]
        text_result[key] = new_text[key]
        score_result[key] = new_score[key]
        wer_hyp_result[key] = new_wer_hyp[key]
        wer_ref_result[key] = new_wer_ref[key]
        cer_hyp_result[key] = new_cer_hyp[key]
        cer_ref_result[key] = new_cer_ref[key]
        if folder:
            ter_hyp_result[key] = new_ter_hyp[key]
            ter_ref_result[key] = new_ter_ref[key]

    
    result_dir = new_dir + "/cov-" + str(selected_num)
    write_new_info(result_dir, "text", text_result)
    write_new_info(result_dir, "score", score_result)
    write_new_info(result_dir, "token", token_result)
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "token_int", token_int_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_new_info(result_dir, "cov", cov_result)
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return cov_result
    


def cov_diff_retrain(old_dir, new_dir, dataset, p):
#     old_cov = load_cov(old_dir + "/cov")
    new_cov = load_cov(new_dir + "/cov")
    feature_speech = dict()
    noise_speech = dict()
    room_speech = dict()
    for key in new_cov.keys():
        orgi_name = key.split("&")[0]
#         diff = float(old_cov[orgi_name]) - float(new_cov[key])
        diff = float(new_cov[key])
        seq = key.split("&")[1].replace(".wav", "")
        if ("dev" in new_dir) & (dataset == "TIMIT"):
            if "a" in seq:
                room_speech[key] = diff
            if "f" in seq:
                feature_speech[key] = diff
            if "n" in seq:
                noise_speech[key] = diff    
        else:
            if "a" in seq:
                room_speech[key] = diff
            else:
                seq = int(seq)
                if seq <= 7:
                    feature_speech[key] = diff
                else:
                    noise_speech[key] = diff
    print("cov select begin")
    feature_result = sort_type_by_diff("feature", int(len(new_cov) * p/3), feature_speech)
    noise_result = sort_type_by_diff("noise", int(len(new_cov) * p/3), noise_speech)
    room_result = sort_type_by_diff("room", int(len(new_cov) * p/3), room_speech)
    item_result = feature_result + noise_result + room_result
    cov_result = dict()
    for item in item_result:
        cov_result[item[0]] = item[1]
        
    result_dir = new_dir + "/retrain-cov"
    write_new_info(result_dir, "cov_speech", cov_result)
    return cov_result



def select_by_cam(new_dir, selected_num):
    act_cell_dict = load_act_cell(new_dir + "/act_cell")
    new_token = load_token(new_dir + "/token")
    new_token_int = load_token(new_dir + "/token_int")
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_text = load_text(new_dir + "/text")
    new_score = load_score(new_dir + "/score")
    new_cov = load_cov(new_dir + "/cov")
    new_wer_ref = load_hyp_ref(new_dir + "/score_wer/ref.trn")
    new_wer_hyp = load_hyp_ref(new_dir + "/score_wer/hyp.trn")
    new_cer_ref = load_hyp_ref(new_dir + "/score_cer/ref.trn")
    new_cer_hyp = load_hyp_ref(new_dir + "/score_cer/hyp.trn")
    folder = os.path.exists(new_dir + "/score_ter/")
    if folder:
        new_ter_ref = load_hyp_ref(new_dir + "/score_ter/ref.trn")
        new_ter_hyp = load_hyp_ref(new_dir + "/score_ter/hyp.trn")
        ter_ref_result = dict(new_ter_ref)
        ter_hyp_result = dict(new_ter_hyp)
    cov_result = dict()
    token_result = dict()
    token_int_result = dict()
    token_gini_result = dict()
    sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()
    
    count = 0
    
    keys = list(act_cell_dict.keys())
    random.shuffle(keys)
    current = act_cell_dict[keys[0]]
    select = dict()
    
    for i in range(1, len(keys)):
        if count >= selected_num:
            break
        key = keys[i]
        diff=list(set(current).difference(set(act_cell_dict[key])))
        if len(diff) > 0:
            select[key] = act_cell_dict[key]
            count += 1
            current = list(set(current).intersection(set(act_cell_dict[key])))
            
    if count < selected_num:
        print("count is ", count, "select_num is", selected_num)
        unselect = list(set(keys) - set(list(select.keys())))
        print("the length of unselect is", len(unselect))
        random_select = random.sample(unselect, selected_num - count)
        for key in random_select:
            select[key] = act_cell_dict[key]

    for key in select.keys():
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]
        text_result[key] = new_text[key]
        score_result[key] = new_score[key]
        wer_hyp_result[key] = new_wer_hyp[key]
        wer_ref_result[key] = new_wer_ref[key]
        cer_hyp_result[key] = new_cer_hyp[key]
        cer_ref_result[key] = new_cer_ref[key]
        cov_result[key] = new_cov[key]
        if folder:
            ter_hyp_result[key] = new_ter_hyp[key]
            ter_ref_result[key] = new_ter_ref[key]

    result_dir = new_dir + "/cam-" + str(selected_num)
    write_new_info(result_dir, "text", text_result)
    write_new_info(result_dir, "score", score_result)
    write_new_info(result_dir, "token", token_result)
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "token_int", token_int_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_new_info(result_dir, "cov", cov_result)
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return select


def random_sample(new_dir, selected_num):
    new_token = load_token(new_dir + "/token")
    new_token_int = load_token(new_dir + "/token_int")
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_text = load_text(new_dir + "/text")
    new_score = load_score(new_dir + "/score")
    new_wer_ref = load_hyp_ref(new_dir + "/score_wer/ref.trn")
    new_wer_hyp = load_hyp_ref(new_dir + "/score_wer/hyp.trn")
    new_cer_ref = load_hyp_ref(new_dir + "/score_cer/ref.trn")
    new_cer_hyp = load_hyp_ref(new_dir + "/score_cer/hyp.trn")
    folder = os.path.exists(new_dir + "/score_ter/")
    if folder:
        new_ter_ref = load_hyp_ref(new_dir + "/score_ter/ref.trn")
        new_ter_hyp = load_hyp_ref(new_dir + "/score_ter/hyp.trn")
        ter_ref_result = dict(new_ter_ref)
        ter_hyp_result = dict(new_ter_hyp)
    token_result = dict()
    token_int_result = dict()
    token_gini_result = dict()
    sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()
    keys = random.sample(new_token.keys(), selected_num)
    for key in keys:
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
        sum_gini_result[key] = new_sum_gini[key]
        text_result[key] = new_text[key]
        score_result[key] = new_score[key]
        wer_hyp_result[key] = new_wer_hyp[key]
        wer_ref_result[key] = new_wer_ref[key]
        cer_hyp_result[key] = new_cer_hyp[key]
        cer_ref_result[key] = new_cer_ref[key]
        if folder:
            ter_hyp_result[key] = new_ter_hyp[key]
            ter_ref_result[key] = new_ter_ref[key]

    result_dir = new_dir + "/random-"+ str(selected_num)
    write_new_info(result_dir, "text", text_result)
    write_new_info(result_dir, "score", score_result)
    write_new_info(result_dir, "token", token_result)
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_new_info(result_dir, "token_int", token_int_result)
    write_new_info(result_dir, "sum_gini", sum_gini_result)
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return sum_gini_result

        
def gen_new_prep_file(new_file, new_audio, dataset, flag, selected_num, test_set):
    f2 = open(new_file, "r")
    new_result = dict()
    line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        new_result[key] = line
        line = f2.readline()

    retrain_result = dict()
    for key in new_audio.keys():
        if ("spk2gender" in new_file) & ("TIMIT" in dataset):
            key = key.split("_")[0]
        retrain_result[key] = new_result[key]

    new_path = new_file.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    
    for key in retrain_result.keys():
        w.write(retrain_result[key] + "\n")
    w.close()
    return retrain_result

def gen_new_wavscp(new_wavscp, new_audio, dataset, flag, selected_num, test_set):
    f2 = open(new_wavscp, "r")
    new_scp = dict()
    line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        new_scp[key] = line
        line = f2.readline()
    retrain_scp = set()
    for key in new_audio.keys():
        if "ted" in dataset:
            key = key.split("-")[0]
        retrain_scp.add(new_scp[key])
    new_path = new_wavscp.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for line in retrain_scp:
        w.write(line + "\n")
    w.close()
    
def gen_new_recog2file(new_recogfile, new_audio, dataset, flag, selected_num, test_set):
    f2 = open(new_recogfile, "r")
    new_recog = dict()
    line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        new_recog[key] = line
        line = f2.readline()
    recog_result = set()
    for key in new_audio.keys():
        if "ted" in dataset:
            key = key.split("-")[0]
        recog_result.add(new_recog[key])
    new_path = new_recogfile.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for line in recog_result:
        w.write(line + "\n")
    w.close()    
    

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def gen_new_spk2utt(orgi_spk2utt, new_audio, dataset, flag, selected_num, test_set):
    f = open(orgi_spk2utt, "r")
    spk2utt = dict()
    line = f.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        spk2utt[key] = line.split(" ")[1:]
        line = f.readline()
    for key in new_audio.keys():
        if "an4" in dataset:
            spk = key.split("-")[0]
        if "TIMIT" in dataset:
            spk = key.split("_")[0]
        spk2utt[spk].append(key.replace(".wav", ""))
    new_path = orgi_spk2utt.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for key in spk2utt.keys():
        w.write(key.replace(".wav", "") + " " + " ".join(spk2utt[key]) + "\n")
    w.close()    


def gen_new_spk2utt_v1(new_utt2spk, new_spk2utt, dataset, flag, selected_num, test_set):
    spk2utt = dict()
    for key in new_utt2spk.keys():
        spk = new_utt2spk[key].split(" ")[-1]
        if spk in spk2utt.keys():
            spk2utt[spk].append(key)
        else:
            spk2utt[spk] = []
            spk2utt[spk].append(key)
            
    new_path = new_spk2utt.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for key in spk2utt.keys():
        w.write(key + " " + " ".join(spk2utt[key]) + "\n")
    w.close()

    
def gen_new_stm(new_stm, new_audio, dataset, flag, selected_num, test_set):
    f2 = open(new_stm, "r")
    orgi_stm = dict()
    line = f2.readline()
    start = []
    while ";;" in line:
        start.append(line)
        line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if key in orgi_stm.keys():
            orgi_stm[key].append(line)
        else:
            orgi_stm[key] = []
            orgi_stm[key].append(line)
        line = f2.readline()     
    stm_result = dict()
    for key in new_audio.keys():
        key = key.split("-")[0]
        if key not in stm_result.keys():
            stm_result[key] = orgi_stm[key]
        
    new_path = new_stm.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for line in start:
        w.write(line + "\n")
    for key in stm_result.keys():
        for line in stm_result[key]:
            w.write(line + "\n")
    w.close()

def copy_file(source, test_set, flag, selected_num):
    destination = source.split("/")
    destination[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(destination[:-1]))
    destination = "/".join(destination)
    shutil.copyfile(source, destination)
    if os.path.exists(destination):
        logging.info("copy success")
        
def copy_dir(source, test_set, flag, selected_num):
    destination = source.split("/")
    destination[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    mkdir("/".join(destination[:-1]))
    destination = "/".join(destination)
    shutil.copytree(source, destination)
        
        
def test_data_prep(orgi_dir, new_dir, dataset, new_audio, flag, selected_num, test_set):
    if "an4" in dataset:
        gen_new_prep_file(new_dir + "/text", new_audio, dataset, flag, selected_num, test_set)
        gen_new_prep_file(new_dir + "/utt2spk", new_audio, dataset, flag, selected_num, test_set)
        gen_new_wavscp(new_dir + "/wav.scp", new_audio, dataset, flag, selected_num, test_set)
        gen_new_spk2utt(orgi_dir + "/spk2utt", new_audio, dataset, flag, selected_num, test_set)
        
    if "TIMIT" in dataset:
        gen_new_prep_file(new_dir + "/text", new_audio, dataset, flag, selected_num, test_set)
        gen_new_prep_file(new_dir + "/utt2spk", new_audio, dataset, flag, selected_num, test_set)
        gen_new_prep_file(new_dir + "/spk2gender", new_audio, dataset, flag, selected_num, test_set)
        gen_new_wavscp(new_dir + "/wav.scp", new_audio, dataset, flag, selected_num, test_set)
        gen_new_spk2utt(orgi_dir + "/spk2utt", new_audio, dataset, flag, selected_num, test_set)
     
    if "ted" in dataset:
        new_utt2spk = gen_new_prep_file(new_dir + "/utt2spk", new_audio, dataset, flag, selected_num, test_set)
        gen_new_spk2utt_v1(new_utt2spk, new_dir + "/spk2utt", dataset, flag, selected_num, test_set)
        gen_new_wavscp(new_dir + "/wav.scp", new_audio, dataset, flag, selected_num, test_set)
        gen_new_prep_file(new_dir + "/text", new_audio, dataset, flag, selected_num, test_set)
        gen_new_prep_file(new_dir + "/segments", new_audio, dataset, flag, selected_num, test_set)
        #gen_new_prep_file(new_dir + "/utt2dur", new_audio, dataset, flag, selected_num, test_set)
        #gen_new_prep_file(new_dir + "/utt2num_frames", new_audio, dataset, flag, selected_num, test_set)
        gen_new_stm(new_dir + "/stm", new_audio, dataset, flag, selected_num, test_set)
        gen_new_recog2file(new_dir + "/reco2file_and_channel", new_audio, dataset, flag, selected_num, test_set)
        #gen_new_prep_file(new_dir + "/feats.scp", new_audio, dataset, flag, selected_num, test_set)
        copy_file(new_dir + "/glm", test_set, flag, selected_num)
        #copy_file(new_dir + "/frame_shift",test_set, flag, selected_num)
        #copy_dir(new_dir + "/conf", test_set, flag, selected_num)
        
    
def cua_orgi_gini(orgi_dir, new_dir):
    orgi_sum_gini = load_gini(orgi_dir + "/sum_gini")
    orgi_token_gini = load_gini(orgi_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_token_gini = load_gini(new_dir + "/token_gini")
    orgi_avg_gini = {}
    total_orgi_gini = 0
    for key in orgi_sum_gini.keys():
        orgi_avg_gini[key] = orgi_sum_gini[key] / len(orgi_token_gini[key])
        total_orgi_gini += orgi_sum_gini[key] / len(orgi_token_gini[key])
    print("orgi avg gini is", total_orgi_gini/(len(orgi_avg_gini)))
    new_avg_gini = {}
    total_new_gini = 0
    for key in new_sum_gini.keys():
        new_avg_gini[key] = new_sum_gini[key] / len(new_token_gini[key])
        total_new_gini += new_avg_gini[key]
    print("new avg gini is", total_new_gini/(len(new_avg_gini)))
    
    
if __name__ == '__main__':
    guide_v1("./orgi", "./new", 0.2)









