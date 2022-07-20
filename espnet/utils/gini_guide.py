import copy

from espnet.utils.gini_utils import *
import random
import os
import shutil

def sort_test_by_gini(old_dir, new_dir, aug_type, selected_num):
    new_token_int = load_gini(new_dir + "/token_int")
    new_token = load_token(new_dir + "/token")
    orig_token = load_token(old_dir + "/token")
    new_token_gini = load_gini(new_dir + "/token_gini")
    orig_token_gini = load_gini(old_dir + "/token_gini")
    orig_word_list, orig_gini_list = get_word_gini(orig_token, orig_token_gini)
    new_word_list, new_gini_list = get_word_gini(new_token, new_token_gini)
    ref_dict, hyp_dict, orig_eval_dict = load_result_txt(old_dir + "/score_wer/result.txt")
    #s_t, i_t, c_t, flag = get_gini_threshold(orig_gini_list, orig_eval_dict, orig_word_list)
    s_t, i_t, c_t, flag = get_gini_threshold(orig_token_gini, orig_eval_dict, orig_token)
    #flag = "len_first"
    sort_list = gini_sort(s_t, new_gini_list, orig_token, new_token, flag)
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
    #sum_gini_result = dict()
    text_result = dict()
    score_result = dict()
    wer_ref_result = dict()
    wer_hyp_result = dict()
    cer_ref_result = dict()
    cer_hyp_result = dict()
   
    for item in sort_list[:selected_num]:
        key = item.name
        token_result[key] = new_token[key]
        token_int_result[key] = new_token_int[key]
        token_gini_result[key] = new_token_gini[key]
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
    write_hyp_ref(result_dir + "/score_wer/", "hyp.trn", wer_hyp_result)
    write_hyp_ref(result_dir + "/score_wer/", "ref.trn", wer_ref_result)
    write_hyp_ref(result_dir + "/score_cer/", "hyp.trn", cer_hyp_result)
    write_hyp_ref(result_dir + "/score_cer/", "ref.trn", cer_ref_result)
    if folder:
        write_hyp_ref(result_dir + "/score_ter/", "hyp.trn", ter_hyp_result)
        write_hyp_ref(result_dir + "/score_ter/", "ref.trn", ter_ref_result)
    return token_result

def sort_test_by_gini_v1(old_dir, new_dir, aug_type):
    orig_token_gini = load_gini_v1(old_dir + "/token_gini")
    selected_num = len(orig_token_gini)
    new_token_gini = load_gini_v1(new_dir + "/token_gini") 
    print(new_dir + "/data.json")
    new_json_data = load_data_json(new_dir + "/data.json")
    orig_json_data = load_data_json(old_dir + "/data.json")
    res_json = copy.deepcopy(new_json_data)
    token_gini_result = dict()
    orig_ref_dict, orig_hyp_dict, orig_eval_dict = load_result_txt(old_dir + "/result.wrd.txt")
    new_ref_dict, new_hyp_dict, new_eval_dict = load_result_txt(new_dir + "/result.wrd.txt")
    orig_token_list = get_token_list(orig_json_data)
    new_token_list = get_token_list(new_json_data)
    orig_word_list, orig_gini_list = get_word_gini_v1(orig_json_data, orig_token_gini, orig_hyp_dict)
    new_word_list, new_gini_list = get_word_gini_v1(new_json_data, new_token_gini, new_hyp_dict)
    s_t, i_t, c_t, flag = get_gini_threshold(orig_gini_list, orig_eval_dict, orig_word_list)
    print(s_t, i_t, c_t)
    sort_list = gini_sort_v1(s_t, new_gini_list, orig_word_list, new_word_list, flag)[:selected_num]
    res_key = []
    for item in sort_list:
        key = item.name
        token_gini_result[key] = new_token_gini[key]
        res_key.append(key)
        
    for key in new_json_data["utts"]:
        if key.lower() not in res_key:
            del res_json["utts"][key]

    result_dir = new_dir + "/gini-" + str(selected_num) 
    write_new_info(result_dir, "token_gini", token_gini_result)
    write_json_data(result_dir, res_json)
    return token_gini_result

    
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
    

def gen_new_prep_file(new_file, new_audio, dataset, flag, selected_num, test_set):
    f2 = open(new_file, "r")
    new_result = dict()
    line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if "ted" in dataset:
            key = key.lower()
        new_result[key] = line
        line = f2.readline()

    retrain_result = dict()
    print(new_audio)
    for key in new_audio.keys():
        if ("spk2gender" in new_file) & ("TIMIT" in dataset):
            key = key.split("_")[0]
        retrain_result[key] = new_result[key]

    new_path = new_file.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
        if "ted" in dataset:
            key = key.lower()
        new_scp[key] = line
        line = f2.readline()
    retrain_scp = set()
    for key in new_audio.keys():
        if "ted" in dataset:
            key = key.split("-")[0]
        retrain_scp.add(new_scp[key])
    new_path = new_wavscp.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
        key = line.split(" ")[0].lower()
        new_recog[key] = line
        line = f2.readline()
    recog_result = set()
    for key in new_audio.keys():
        if "ted" in dataset:
            key = key.split("-")[0]
        recog_result.add(new_recog[key])
    new_path = new_recogfile.split("/")
    new_path[-2] = test_set + "-"+ flag+ "-" + str(selected_num)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
        if "ted" in dataset:
            key = key.lower()
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
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
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
    destination[-2] = destination[-2] + ".orig"
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
        
    




