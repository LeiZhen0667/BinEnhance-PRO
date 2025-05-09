import argparse
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import os
import pickle

def set_random_seed(seed_value=1234):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def get_func_classes(func_embeddings, MIN_BLOCKS=0):
    classes = []
    func2id = {}
    cop_func = []
    for funcN, embed in func_embeddings.items():
        func = funcN.split("_")[0] + "|||" + funcN.split("_")[5] + "|||" + funcN.split("|||")[-1]
        cop_func.append(funcN)
        if func not in func2id:
            func2id[func] = len(func2id)
    print("cop func:" + str(len(cop_func)) + "/" + str(len(func_embeddings)))
    for func, id in func2id.items():
        classes.append([])
    for funcN in cop_func:
        func = funcN.split("_")[0] + "|||" + funcN.split("_")[5] + "|||" + funcN.split("|||")[-1]
        classes[func2id[func]].append(funcN)
    new_class = []
    for cl in classes:
        if len(cl) >= 2:
            np.random.shuffle(cl)
            new_class.append(cl)
    np.random.shuffle(new_class)
    return new_class
    pass

def generate_eval_data(classes, poolsize=10000):
    sample_num = poolsize
    max_pos = 10 if poolsize > 20 else poolsize / 2
    max_sample = 5000
    eval_datas = []
    np.random.shuffle(classes)
    classes_eval = classes
    sl_data = {}
    for class_eval in tqdm(classes_eval):
        cl_index = classes_eval.index(class_eval)
        eval_data = []
        pos_num = 0
        for ce in class_eval[1:len(class_eval)]:
            pos_num += 1
            eval_data.append((class_eval[0], ce, 1))
            if pos_num >= max_pos:
                break
        if pos_num == 0:
            continue
        while len(eval_data) < sample_num:
            flag = 0
            for i in range(10):
                index1 = np.random.randint(0, len(classes_eval))

                if index1 != cl_index:
                    flag = 1
                    break
            if flag == 0:
                continue
            flag = 0
            for i in range(10):
                index2 = np.random.randint(0, len(classes_eval[index1]))
                # s_d = str(class_eval[0]) + str(classes_eval[index1][index2]) + str(0)
                s_d = (class_eval[0], classes_eval[index1][index2], 0)
                if str(s_d) not in sl_data:
                    sl_data[str(s_d)] = 0
                    flag = 1
                    break
            if flag != 1:
                continue
            else:
                eval_data.append((class_eval[0], classes_eval[index1][index2], 0))
        eval_datas.append(eval_data)
        if len(eval_datas) >= max_sample:
            break
    return eval_datas
    pass

def eval_by_map(Gs_embed, eval_datas):
    lens = []
    sims = []
    labels = []
    for eds in tqdm(eval_datas):
        leng = 0
        label = []
        feat_s = []
        feat_e = []
        feat_s.append(Gs_embed[eds[0][0]])
        for ed in eds:
            feat_e.append(Gs_embed[ed[1]])
            label.append(str(ed[0]) + "||||" + str(ed[1]) + "||||" + str(ed[2]))
            leng += ed[2]
        sims.append(cosine_similarity(np.array(feat_s), np.array(feat_e)).reshape(-1))
        lens.append(leng)
        labels.append(label)
    print("it is calculating map score ...")
    ans = cal_map_metrics(sims, labels, lens)
    return ans
    pass

def cal_map_metrics(sims, labels, lens):
    q = len(sims)
    map = 0.0
    aps = []
    for i in tqdm(range(q)):
        ap = 0.0
        cos_num = 0.0
        sim = sims[i]
        label = labels[i]
        sorted_list = sorted([(si, li) for si, li in zip(sim, label)], reverse=True)
        label = [li for _, li in sorted_list]
        for j in range(min(10, lens[i])):
            if type(label[j]) == int:
                if int(label[j]) == 1:
                    cos_num += 1.0
                    ap = ap + cos_num / float(j + 1)
            else:
                if int(label[j].split("||||")[-1]) == 1:
                    cos_num += 1.0
                    ap = ap + cos_num / float(j + 1)
        ap = ap / lens[i]
        aps.append(ap)
        map += ap
    ans = map / float(q)
    return ans

def comp_jaccard_sim_weight(i1, i2):
    i1 = set(i1)
    i2 = set(i2)
    i1_un_i2 = i1.union(i2)
    score2 = len(i1_un_i2)
    if score2 == 0:
        return 0.01
    i1_in_i2 = i1.intersection(i2)
    score1 = len(i1_in_i2)
    sim = score1 / score2
    return sim

def calculate_rtf_similarity_two(func_name, basic_dict):
    if func_name[0] in basic_dict and func_name[1] in basic_dict:
        a_func_dict = basic_dict[func_name[0]]
        p_func_dict = basic_dict[func_name[1]]
        sim = comp_jaccard_sim_weight(a_func_dict, p_func_dict)
    else:
        sim = 0.01
    return sim

def map_eval(eval_type=0, eval_p="", embed_path="", model="", poolsize=10000):
    set_random_seed()
    eval_func_embeddings_path = embed_path
    with open(eval_func_embeddings_path, "r") as f:
        func_embeddings = json.load(f)
    print(eval_func_embeddings_path)
    if not os.path.exists(eval_p):
        os.makedirs(eval_p)
    eval_data_path = eval_p + "/" + model + "_" + str(poolsize) + "_eval_data_" + str(eval_type) + ".json"

    print("it is generating eval data ...")
    if not os.path.exists(eval_data_path):
        classes = get_func_classes(func_embeddings, MIN_BLOCKS=eval_type)
        eval_datas = generate_eval_data(classes, poolsize=poolsize)
        e = {}
        e["d"] = eval_datas
        with open(eval_data_path, "w") as f:
            json.dump(e, f)
    else:
        with open(eval_data_path, "r") as f:
            eval_datas = json.load(f)["d"]
    print("generate eval data completed ...")
    print(len(eval_datas))
    ans = eval_by_map(func_embeddings, eval_datas)
    return ans

def init_eval_poolsize(model="", PS=[], eval_path="", embed_path=""):
    ans = []
    for p in PS:
        a = map_eval(eval_type=0, eval_p=eval_path, poolsize=p, embed_path=embed_path, model=model)
        ans.append(a)
    return ans

def binenhance_poolsize(model, data_base, device, PS=[], eval_path="", embed_path="", ispro=""):
    base_path = '/mnt/data/leizhen/BinEnhance-main-1/EESG'
    f_strings_path = os.path.join(base_path, "all_strings.json")
    f_gv_path = os.path.join(base_path, "all_global_vars.json")
    f_ef_path = os.path.join(base_path, "all_external_funcs.json")
    f_rs = load_embeddings(f_strings_path)
    f_gd = load_embeddings(f_gv_path)
    f_ef = load_embeddings(f_ef_path)

    ans = []
    if ispro == False:
        model_path = os.path.join(os.path.join(data_base, "BinEnhance_models"), model)
    else:
        model_path = os.path.join(os.path.join(data_base, "BinEnhance-PRO_models"), model)
    for p in PS:
        a = combine_eval(model_path, f_rs, f_gd, f_ef, eval_type=0, model=model, poolsize=p, savepaths=embed_path, eval_p=eval_path, device=device, ispro=ispro)
        ans.append(a)
    return ans

def combine_eval(model_name, f_strings, global_vars, external_funcs, device, eval_type=5,  model="", poolsize=128, eval_p="", savepaths="",ispro=""):
    set_random_seed()
    eval_data_path = eval_p + "/" + model + "_" + str(poolsize) + "_eval_data_" + str(eval_type) + ".json"
    # eval_data_path = eval_p + model + "/" + model + "_" + str(poolsize) + "_eval_data_" + str(eval_type) + yarchc + yoptc + tarchc + toptc + ".json"
    if not os.path.exists(eval_p):
        os.makedirs(eval_p)
    if os.path.exists(eval_data_path):
        with open(eval_data_path, "r") as f:
            eval_datas = json.load(f)["d"]
    else:
        print("Eval data is not exist!")
        return
    if ispro == False:
        model_path = model_name + "/BinEnhance-best.pt"
    else:
        model_path = model_name + "/BinEnhance-PRO-best.pt"

    model = torch.load(model_path, map_location=device)
    predict_num = 0
    lens = []
    labels = []
    cs = []
    ss = []
    es = []
    ds = []
    with open(savepaths, "r") as f:
        subgraphs_embed = json.load(f)
    tq = tqdm(eval_datas, position=0, ncols=80)
    pos = len(eval_datas[0])
    for eds in tq:
        leng = 0
        label = []
        a_fn = eds[0][0]
        if a_fn not in subgraphs_embed:
            continue
        feats_a = [subgraphs_embed[a_fn]]
        feats_b = []
        sim_s = []
        sim_e = []
        sim_d = []
        for ed in eds:
            b_fn = ed[1]
            if b_fn not in subgraphs_embed:
                continue
            funcNames = [a_fn, b_fn]
            feats_b.append(subgraphs_embed[b_fn])
            label.append(ed[2])
            leng += ed[2]
            predict_num += 1
            sim_s.append(calculate_rtf_similarity_two(funcNames, f_strings))
            sim_d.append(calculate_rtf_similarity_two(funcNames, global_vars))
            sim_e.append(calculate_rtf_similarity_two(funcNames, external_funcs))
        if leng == 0:
            continue
        cos_dist = cosine_similarity(np.array(feats_a), np.array(feats_b)).reshape(-1)
        cs.append(cos_dist)
        ss.append(sim_s)
        es.append(sim_e)
        ds.append(sim_d)
        lens.append(leng)
        labels.append(label)
    cs = np.array(cs).flatten()
    ss = np.array(ss).flatten()
    es = np.array(es).flatten()
    ds = np.array(ds).flatten()
    combined = torch.tensor(np.stack((cs, ss, ds, es), axis=1)).float().to(device)
    sim = model.combine_sims(combined)
    sim = model.sim_activate(sim).cpu().detach().numpy().tolist()
    sims = [sim[i:i + pos] for i in range(0, len(sim), pos)]
    print("predict num:" + str(predict_num) + "/" + str(sum([len(label) for label in labels])))
    print("it is calculating map score ...")
    ans = cal_map_metrics(sims, labels, lens)
    print("block_num>" + str(eval_type) + ":eval completed! MAP:" + str(ans))
    return ans

def load_embeddings(path):
    """Helper function to load embeddings from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="the path of function embedding(including initial model and BinEnhance).")
    args = parser.parse_args()
    print(args)
    data_path = args.data_path
    models = ["HermesSim", "Gemini", "Asteria", "TREX", "Asm2vec", "Diemph","Codeart"]
    PS = [2, 16, 32, 128, 512, 1024, 2048, 4096, 8192, 10000]
    ans = []

    for model in models:
        ans.append([[], [], []])
        print("---------------------------------------" + model + "--------------------------------------------")
        eval_path = os.path.join(os.path.join(data_path, "Eval_datas"), model)
        init_embed_path = os.path.join(os.path.join(data_path, model), "test_function_embeddings.json")
        binenhance_embed_path = os.path.join(os.path.join(data_path, model + "+BinEnhance"), "test_function_embeddings.json")
        
        binenhance_pro_embed_path = os.path.join(os.path.join(data_path, model + "+BinEnhance-PRO"), "test_function_embeddings.json")
        ans[models.index(model)][0] = init_eval_poolsize(model=model, PS=PS, eval_path=eval_path, embed_path=init_embed_path)
        
        ans[models.index(model)][1] = binenhance_poolsize(model=model, data_base=data_path, PS=PS, eval_path=eval_path,
                                                    embed_path=binenhance_embed_path, device=device, ispro=False)

        ans[models.index(model)][2] = binenhance_poolsize(model=model, data_base=data_path, PS=PS, eval_path=eval_path,
                                                    embed_path=binenhance_pro_embed_path, device=device, ispro=True)
        print("---------------------------------------" + model + "--------------------------------------------")
        print("PoolSize:" + str(PS))
        print("Original Model MAP scores:" + str(ans[models.index(model)][0]))
        print("Original Model + BinEnhance MAP scores:" + str(ans[models.index(model)][1]))
        print("Original Model + BinEnhance-PRO MAP scores:" + str(ans[models.index(model)][2]))
        print("---------------------------------------" + model + "--------------------------------------------")

if __name__ == '__main__':
    main()