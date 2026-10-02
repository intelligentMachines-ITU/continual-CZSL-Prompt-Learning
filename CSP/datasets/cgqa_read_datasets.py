import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    # "cgqa": os.path.join(DIR_PATH, "../data/cgqa"),
    "cgqa": "./data/CCZSL_benchmark/CGQA/", #set path of dataset
    }

selected_session_g_pair2_idx = "cgqa-s4"        #max session-4 we train, used for GT Ids. 

t7_files = {
    "cgqa-s0" : os.path.join(DATASET_PATHS["cgqa"], "session 0/metadata_compositional-split-natural.t7"),
    "cgqa-s1" : os.path.join(DATASET_PATHS["cgqa"], "session 1/1metadata_compositional-split-natural.t7"),
    # "cgqa-s01": os.path.join(DATASET_PATHS["cgqa"], "session 1/01 pairs/compositional-split-natural/01metadata_compositional-split-natural.t7"),
    "cgqa-s2" : os.path.join(DATASET_PATHS["cgqa"], "session 2/2metadata_compositional-split-natural.t7"),
    # "cgqa-s012" : os.path.join(DATASET_PATHS["cgqa"], "session 2/012 pairs/compositional-split-natural/012metadata_compositional-split-natural.t7"),
    "cgqa-s3" : os.path.join(DATASET_PATHS["cgqa"], "session 3/3metadata_compositional-split-natural.t7"),
    # "cgqa-s0123" : os.path.join(DATASET_PATHS["cgqa"], "session 3/0123 pairs/compositional-split-natural/0123metadata_compositional-split-natural.t7"),
    "cgqa-s4" : os.path.join(DATASET_PATHS["cgqa"], "session 4/4metadata_compositional-split-natural.t7"),
    # "cgqa-s01234" : os.path.join(DATASET_PATHS["cgqa"], "session 4/01234 pairs/compositional-split-natural/01234metadata_compositional-split-natural.t7"),
}



teacher_t7_files = {
    "cgqa-s1" : os.path.join(DATASET_PATHS["cgqa"], "session 0/metadata_compositional-split-natural.t7"),
    "cgqa-s2" : os.path.join(DATASET_PATHS["cgqa"], "session 1/1metadata_compositional-split-natural.t7"),
    "cgqa-s3" : os.path.join(DATASET_PATHS["cgqa"], "session 2/2metadata_compositional-split-natural.t7"),
    "cgqa-s4" : os.path.join(DATASET_PATHS["cgqa"], "session 3/3metadata_compositional-split-natural.t7"),
}

train_val_test_txtfiles = {
    "cgqa-s0" : os.path.join(DATASET_PATHS["cgqa"], "session 0/compositional-split-natural"),
    "cgqa-s1" : os.path.join(DATASET_PATHS["cgqa"], "session 1/01 pairs/compositional-split-natural"),
    "cgqa-s2" : os.path.join(DATASET_PATHS["cgqa"], "session 2/012 pairs/compositional-split-natural"),
    "cgqa-s3" : os.path.join(DATASET_PATHS["cgqa"], "session 3/0123 pairs/compositional-split-natural"),
    "cgqa-s4" : os.path.join(DATASET_PATHS["cgqa"], "session 4/01234 pairs/compositional-split-natural"),
}

#already added
ALL_ATTR_OBJ_PAIRS = {
     "cgqa-s0": (233, 363),
     "cgqa-s1": (268, 421), 
     "cgqa-s2": (300, 488),
     "cgqa-s3": (336, 552),
     "cgqa-s4": (375, 614),
}

#in trainpairs.txt file.
TRAIN_PAIRS = {
     "cgqa-s0": 2392,
     "cgqa-s1": 2883, 
     "cgqa-s2": 3655,
     "cgqa-s3": 4491,
     "cgqa-s4": 5053
}

import os
import json
import ast

def load_json_keys(path):
    with open(path, encoding="utf-8") as f:
        return list(json.load(f).keys())

def load_pair_keys(path):
    with open(path, encoding="utf-8") as f:
        return [ast.literal_eval(k) for k in json.load(f).keys()]

ALL_ATTR_OBJ = {
    "cgqa-s0": {
        "all_attrs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_attr2idx_session0.json")
        ),
        "all_objs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_obj2idx_session0.json")
        ),
        "all_pairs": load_pair_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_pair2idx_session0.json")
        ),
    },

    "cgqa-s1": {
        "all_attrs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_attr2idx_session1.json")
        ),
        "all_objs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_obj2idx_session1.json")
        ),
        "all_pairs": load_pair_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_pair2idx_session1.json")
        ),
    },

    "cgqa-s2": {
        "all_attrs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_attr2idx_session2.json")
        ),
        "all_objs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_obj2idx_session2.json")
        ),
        "all_pairs": load_pair_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_pair2idx_session2.json")
        ),
    },

    "cgqa-s3": {
        "all_attrs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_attr2idx_session3.json")
        ),
        "all_objs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_obj2idx_session3.json")
        ),
        "all_pairs": load_pair_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_pair2idx_session3.json")
        ),
    },

    "cgqa-s4": {
        "all_attrs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_attr2idx_session4.json")
        ),
        "all_objs": load_json_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_obj2idx_session4.json")
        ),
        "all_pairs": load_pair_keys(
            os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_pair2idx_session4.json")
        ),
    }
}

def json_load(path):
    with open(path, encoding="utf-8") as f:
        return list(json.load(f))
import ast

def load_pair2idx(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ans  = {ast.literal_eval(k): v for k, v in raw.items()}
    print(f"ans: {ans}")
    return ans

g_objattrpair2idx_path = {
     "cgqa-s0" : 
     {
          "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_obj2idx_session0.json")),
          "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_attr2idx_session0.json")),
          "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["cgqa"], "session 0/idx/updated_pair2idx_session0.json")),
     },
    
    "cgqa-s1" : 
     {
          "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_obj2idx_session1.json")),
          "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_attr2idx_session1.json")),
          "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["cgqa"], "session 1/idx/updated_pair2idx_session1.json")),
     },

    "cgqa-s2" : 
     {
          "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_obj2idx_session2.json")),
          "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_attr2idx_session2.json")),
          "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["cgqa"], "session 2/idx/updated_pair2idx_session2.json")),
     },

    "cgqa-s3" : 
     {
          "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_obj2idx_session3.json")),
          "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_attr2idx_session3.json")),
          "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["cgqa"], "session 3/idx/updated_pair2idx_session3.json")),
     },

    "cgqa-s4" : 
     {
          "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_obj2idx_session4.json")),
          "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_attr2idx_session4.json")),
          "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["cgqa"], "session 4/idx/updated_pair2idx_session4.json")),
     }
}