import copy

from espnet.utils.gini_utils import *
import random
import os
import shutil

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


# def cov_retrain(new_dir, p):
#     new_cov = load_cov(new_dir + "/cov")
#     feature_speech = dict()
#     noise_speech = dict()
#     room_speech = dict()
#     for key in new_cov.keys():
#         seq = key.split("&")[1].replace(".wav", "")
#         if "a" in seq:
#             room_speech[key] = new_cov[key]
#         else:
#             seq = int(seq)
#             if seq <= 7:
#                 feature_speech[key] = new_cov[key]
#             else:
#                 noise_speech[key] = new_cov[key]
    
    
#     cov_result = dict()
#     for item in feature_speech:
#         cov_result[item[0]] = item[1]
#     for item in noise_speech:
#         cov_result[item[0]] = item[1]
#     for item in room_speech:
#         cov_result[item[0]] = item[1]
#     result_dir = new_dir + "/retrain-cov"
#     write_new_info(result_dir, "cov_speech", cov_result)
#     return cov_result

def cam_retrain(new_dir, p):
    act_cell_dict = load_act_cell(new_dir + "/act_cell")
    feature_speech = dict()
    noise_speech = dict()
    room_speech = dict()
    for key in act_cell_dict.keys():
        seq = key.split("&")[1].replace(".wav", "")
        if "a" in seq:
            room_speech[key] = act_cell_dict[key]
        else:
            seq = int(seq)
            if seq <= 7:
                feature_speech[key] = act_cell_dict[key]
            else:
                noise_speech[key] = act_cell_dict[key]

    selected_num = int(len(act_cell_dict) * p / 3)

    count = 0
    feature_keys = list(feature_speech.keys())
    random.shuffle(feature_keys)
    current = feature_speech[feature_keys[0]]
    feature_select = dict()
    for i in range(1, len(feature_keys)):
        if count >= selected_num:
            break
        key = feature_keys[i]
        diff = list(set(current).difference(set(feature_speech[key])))
        if len(diff) > 0:
            feature_select[key] = feature_speech[key]
            count += 1
            current = list(set(current).intersection(set(feature_speech[key])))

    if count < selected_num:
        select_keys = list(feature_select.keys())
        unselect = list(set(feature_keys) - set(select_keys))
        random_select = random.sample(unselect, selected_num - count)
        for key in random_select:
            feature_select[key] = feature_speech[key]

    count = 0
    noise_keys = list(noise_speech.keys())
    random.shuffle(noise_keys)
    current = noise_speech[noise_keys[0]]
    noise_select = dict()
    for i in range(1, len(noise_keys)):
        if count >= selected_num:
            break
        key = noise_keys[i]
        diff = list(set(current).difference(set(noise_speech[key])))
        if len(diff) > 0:
            noise_select[key] = noise_speech[key]
            count += 1
            current = list(set(noise_speech[key]).intersection(set(current)))

    if count < selected_num:
        select_keys = list(noise_select.keys())
        unselect = list(set(noise_keys) - set(select_keys))
        random_select = random.sample(unselect, selected_num - count)
        for key in random_select:
            noise_select[key] = noise_speech[key]

    count = 0
    room_keys = list(room_speech.keys())
    random.shuffle(room_keys)
    current = room_speech[room_keys[0]]
    room_select = dict()
    for i in range(1, len(room_keys)):
        if count >= selected_num:
            break
        key = room_keys[i]
        diff = list(set(current).difference(set(room_speech[key])))
        if len(diff) > 0:
            room_select[key] = room_speech[key]
            count += 1
            current = list(set(current).intersection(set(room_speech[key])))

    if count < selected_num:
        select_keys = list(room_select.keys())
        unselect = list(set(room_keys) - set(select_keys))
        random_select = random.sample(unselect, selected_num - count)
        for key in random_select:
            room_select[key] = room_speech[key]

    feature_select.update(noise_select)
    feature_select.update(room_select)
    return feature_select


def random_select(new_dir, dataset, p):
    new_token_gini = load_gini(new_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    feature_speech = dict()
    noise_speech = dict()
    room_speech = dict()
    
    for key in new_token_gini.keys():
        seq = key.split("&")[1].replace(".wav", "")
        diff = new_sum_gini[key]/len(new_token_gini[key])
        if "ted" in dataset:
            if "a" in seq:
                room_speech[key] = diff
            else:
                seq = int(seq.split("-")[0])
                if seq <= 7:
                    feature_speech[key] = diff
                else:
                    noise_speech[key] = diff
        else:
            if ("train-new" in new_dir) | (dataset == "an4"):
                if "a" in seq:
                    room_speech[key] = diff
                else:
                    seq = int(seq)
                    if seq <= 7:
                        feature_speech[key] = diff
                    else:
                        noise_speech[key] = diff
            if ("dev" in new_dir) & (dataset == "TIMIT"):
                if "a" in seq:
                    room_speech[key] = diff
                if "f" in seq:
                    feature_speech[key] = diff
                if "n" in seq:
                    noise_speech[key] = diff
            
        
    
    keys1 = random.sample(feature_speech.keys(), int(len(new_token_gini) * p/3))
    keys2 = random.sample(noise_speech.keys(), int(len(new_token_gini) * p/3))
    keys3 = random.sample(room_speech.keys(), int(len(new_token_gini) * p/3))
    
    total_dict = dict()
    feature_dict = dict()
    total_gini = 0
    print("random select begin")
    for key in keys1:
        feature_dict[key] = feature_speech[key]
        total_dict[key] = feature_speech[key]
        total_gini += feature_speech[key]
    print("feature select ", total_gini/len(keys1))
    total_gini = 0
    noise_dict = dict()
    for key in keys2:
        noise_dict[key] = noise_speech[key]
        total_dict[key] = noise_speech[key]
        total_gini += noise_speech[key]
    print("noise select ", total_gini/len(keys2))
    total_gini = 0
    room_dict = dict()
    for key in keys3:
        room_dict[key] = room_speech[key]
        total_dict[key] = room_speech[key]
        total_gini += room_speech[key]
    print("room select ", total_gini/len(keys3))
    
    return total_dict
#     result_dir = new_dir + "/retrain-feature-random"
#     write_new_info(result_dir, "random_speech", feature_dict)
#     result_dir = new_dir + "/retrain-noise-random"
#     write_new_info(result_dir, "random_speech", noise_dict)
#     result_dir = new_dir + "/retrain-room-random"
#     write_new_info(result_dir, "random_speech", room_dict)
    
#     return feature_dict, noise_dict, room_dict

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def gen_new_spk2utt(orgi_spk2utt, new_audio, dataset, flag):
    f = open(orgi_spk2utt, "r")
    spk2utt = dict()
    line = f.readline()
    for key in new_audio.keys():
        if "an4" in dataset:
            spk = key.split("-")[0]
        if "TIMIT" in dataset:
            spk = key.split("_")[0]
        if spk in spk2utt.keys():
            spk2utt[spk].append(key.replace(".wav", ""))
        else:
            spk2utt[spk] = []
            spk2utt[spk].append(key.replace(".wav", ""))
    new_path = orgi_spk2utt.split("/")
    if "dev" in orgi_spk2utt:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for key in spk2utt.keys():
        w.write(key.replace(".wav", "") + " " + " ".join(spk2utt[key]) + "\n")
    w.close()

def gen_new_wavscp(orgi_wavscp, new_wavscp, new_audio, dataset, flag):
    f1 = open(orgi_wavscp, "r")
    orgi_scp = dict()
    line = f1.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_scp[key] = line
        line = f1.readline()
    
    f2 = open(new_wavscp, "r")
    new_scp = dict()
    line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        new_scp[key] = line
        line = f2.readline()
    
    retrain_scp = list()
    for key in new_audio.keys():
        if "ted" in dataset:
            key = key.split("-")[0]
        if key in orgi_scp.keys():
            retrain_scp.append(orgi_scp[key])
        else:
            retrain_scp.append(new_scp[key])
    
    new_path = orgi_wavscp.split("/")
    if "dev" in new_wavscp:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for line in retrain_scp:
        w.write(line + "\n")
    w.close()

            
# text utt2spk
def gen_new_prep_file(orgi_file, new_file, new_audio, dataset, flag):
    f1 = open(orgi_file, "r")
    orgi_result = dict()
    line = f1.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_result[key] = line
        line = f1.readline()
    
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
        if ("spk2gender" in orgi_file) & ("TIMIT" in dataset):
            key = key.split("_")[0]
        if key in new_result.keys():
            retrain_result[key] = new_result[key]
        else:
            retrain_result[key] = orgi_result[key]

    new_path = orgi_file.split("/")
    if "dev" in new_file:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    
    for key in retrain_result.keys():
        w.write(retrain_result[key] + "\n")
    w.close()
    return retrain_result

def gen_new_spk2utt_v1(new_utt2spk, new_spk2utt, dataset, flag):
    spk2utt = dict()
    for key in new_utt2spk.keys():
        spk = new_utt2spk[key].split(" ")[-1]
        if spk in spk2utt.keys():
            spk2utt[spk].append(key)
        else:
            spk2utt[spk] = []
            spk2utt[spk].append(key)
            
    new_path = new_spk2utt.split("/")
    if "dev" in new_spk2utt:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for key in spk2utt.keys():
        w.write(key + " " + " ".join(spk2utt[key]) + "\n")
    w.close()

    
def gen_new_stm(orgi_file, new_file, new_audio, dataset, flag):
    f = open(orgi_file, "r")
    orgi_stm = dict()
    line = f.readline()
    start = []
    while ";;" in line:
        start.append(line)
        line = f.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if key in orgi_stm.keys():
            orgi_stm[key].append(line)
        else:
            orgi_stm[key] = []
            orgi_stm[key].append(line)
        line = f.readline()
    
    f2 = open(new_file, "r")
    new_stm = dict()
    line = f2.readline()
    while ";;" in line:
        line = f2.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        if key in new_stm.keys():
            new_stm[key].append(line)
        else:
            new_stm[key] = []
            new_stm[key].append(line)
        line = f2.readline()
    
    stm_result = dict()
    for key in new_audio.keys():
        key = key.split("-")[0]
        if key in orgi_stm.keys():
            stm_result[key] = orgi_stm[key]
        if key in new_stm.keys():
            stm_result[key] = new_stm[key]
    
    new_path = orgi_file.split("/")
    if "dev" in new_file:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
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

def gen_new_recog2file(orgi_recogfile, new_recogfile, new_audio, dataset, flag):
    f1 = open(orgi_recogfile, "r")
    orgi_recog = dict()
    line = f1.readline()
    while line:
        line = line[:-1]
        key = line.split(" ")[0]
        orgi_recog[key] = line
        line = f1.readline()
        
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
        if key in orgi_recog.keys():
            recog_result.add(orgi_recog[key])
        else:
            recog_result.add(new_recog[key])
            
    new_path = orgi_recogfile.split("/")
    if "dev" in new_recogfile:
        new_path[-2] = "dev-"+str(flag)
    else:
        new_path[-2] = "retrain-"+str(flag)
    if "ted" in dataset:
        new_path[-2] = new_path[-2] + ".orig"
    mkdir("/".join(new_path[:-1]))
    new_path = "/".join(new_path)
    w = open(new_path, "w")
    for line in recog_result:
        w.write(line + "\n")
    w.close()

def copy_file(source, flag):
    destination = source.split("/")
    if "dev" in source:
        destination[-2] = "dev-"+str(flag)
    else:
        destination[-2] = "retrain-"+str(flag)
    destination[-2] = destination[-2] + ".orig"
    mkdir("/".join(destination[:-1]))
    destination = "/".join(destination)
    shutil.copyfile(source, destination)
    if os.path.exists(destination):
        logging.info("copy success")
        
def copy_dir(source, flag):
    destination = source.split("/")
    if "dev" in source:
        destination[-2] = "dev-"+str(flag)
    else:
        destination[-2] = "retrain-"+str(flag)
    mkdir("/".join(destination[:-1]))
    destination = "/".join(destination)
    shutil.copytree(source, destination)    
    
def mix_train_set(orgi_dir, new_dir, dataset, orgi_num, p):
    orgi_sum_gini = load_gini(orgi_dir + "/sum_gini")
    orgi_token_gini = load_gini(orgi_dir + "/token_gini")
    new_sum_gini = load_gini(new_dir + "/sum_gini")
    new_token_gini = load_gini(new_dir + "/token_gini")
    orgi_cov = load_cov(orgi_dir + "/cov")
    new_cov = load_cov(new_dir + "/cov")
    orgi_select = random.sample(list(orgi_sum_gini.keys()), orgi_num)
    gini_select_result = {}
    cov_select_result = {}
    total_orgi_gini = 0
    for key in orgi_select:
        gini_select_result[key] = orgi_sum_gini[key] / len(orgi_token_gini[key])
        cov_select_result[key] = orgi_cov[key]
        total_orgi_gini += gini_select_result[key]
    print("orgi avg gini is", total_orgi_gini/(len(orgi_select)))
    new_select = random_select(new_dir, dataset, p)
    total_new_gini = 0
    for key in new_select.keys():
        gini_select_result[key] = new_sum_gini[key] / len(new_token_gini[key])
        cov_select_result[key] = new_cov[key]
        total_new_gini += gini_select_result[key]
    print("new avg gini is", total_new_gini/(len(new_select)))
    
    return gini_select_result, cov_select_result


def tuple2dic(tup):
    result = {}
    for item in tup:
        result[item[0]] = item[1]
    return result

def avg_gini_retrain(new_dir, select_result, p):
    gini_result =  sorted(select_result.items(), reverse=True, key=lambda kv:(kv[1], kv[0]))
    gini_result = gini_result[:int(len(select_result) * p)]
    gini_result = tuple2dic(gini_result)
    result_dir = new_dir + "/retrain-gini-" + str(p)
    write_new_info(result_dir, "avg_gini", gini_result)
    return gini_result   

def random_retrain(new_dir, select_result, p):
    random_result = random.sample(select_result.items(), int(len(select_result) * p))
    random_result = tuple2dic(random_result)
    result_dir = new_dir + "/retrain-random-" + str(p)
    write_new_info(result_dir, "random", random_result)
    return random_result

def cov_retrain(new_dir, select_result, p):
    cov_result = sorted(select_result.items(), reverse=True, key=lambda kv:(kv[1], kv[0]))
    cov_result = cov_result[:int(len(select_result) * p)]
    cov_result = tuple2dic(cov_result)
    result_dir = new_dir + "/retrain-cov-" + str(p)
    write_new_info(result_dir, "cov", cov_result)
    return cov_result   
    
def retrain_data_prep(orgi_dir, new_dir, dataset, new_audio, flag):
    if "an4" in dataset:
        gen_new_prep_file(orgi_dir + "/text", new_dir + "/text", new_audio, dataset, flag)
        gen_new_prep_file(orgi_dir + "/utt2spk", new_dir + "/utt2spk", new_audio, dataset, flag)
        gen_new_wavscp(orgi_dir + "/wav.scp", new_dir + "/wav.scp", new_audio, dataset, flag)
        gen_new_spk2utt(orgi_dir + "/spk2utt", new_audio, dataset, flag)
        
    if "TIMIT" in dataset:
        gen_new_prep_file(orgi_dir + "/text", new_dir + "/text", new_audio, dataset, flag)
        gen_new_prep_file(orgi_dir + "/utt2spk", new_dir + "/utt2spk", new_audio, dataset, flag)
        gen_new_prep_file(orgi_dir + "/spk2gender", new_dir + "/spk2gender", new_audio, dataset, flag)
        gen_new_wavscp(orgi_dir + "/wav.scp", new_dir + "/wav.scp", new_audio, dataset, flag)
        gen_new_spk2utt(orgi_dir + "/spk2utt", new_audio, dataset, flag)
    
    if "ted" in dataset:
        new_utt2spk = gen_new_prep_file(orgi_dir + "/utt2spk",new_dir + "/utt2spk", new_audio, dataset, flag)
        gen_new_spk2utt_v1(new_utt2spk, new_dir + "/spk2utt", dataset, flag)
        gen_new_wavscp(orgi_dir + "/wav.scp",new_dir + "/wav.scp", new_audio, dataset, flag)
        gen_new_prep_file(orgi_dir + "/text", new_dir + "/text", new_audio, dataset, flag)
        gen_new_prep_file(orgi_dir + "/segments", new_dir + "/segments", new_audio, dataset, flag)
        #gen_new_prep_file(orgi_dir + "/utt2dur", new_dir + "/utt2dur", new_audio, dataset, flag)
        #gen_new_prep_file(orgi_dir + "/utt2num_frames", new_dir + "/utt2num_frames", new_audio, dataset, flag)
        gen_new_stm(orgi_dir + "/stm", new_dir + "/stm", new_audio, dataset, flag)
        gen_new_recog2file(orgi_dir + "/reco2file_and_channel",new_dir + "/reco2file_and_channel", new_audio, dataset, flag)
        #gen_new_prep_file(orgi_dir + "/feats.scp",new_dir + "/feats.scp", new_audio, dataset, flag)
        copy_file(new_dir + "/glm", flag)
        #copy_file(new_dir + "/frame_shift",flag)
        #copy_dir(new_dir + "/conf", flag)
    

        
    
        
        
