import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    "utzappos": "./data/CCZSL_benchmark/ut-zappos/", #set path of dataset
    }

selected_session_g_pair2_idx = "utzappos-s2"      #max session-2 we train, used for GT Ids. 


t7_files = {
    "utzappos-s0" : os.path.join(DATASET_PATHS["utzappos"], "session 0/metadata_compositional-split-natural.t7"),
    "utzappos-s1" : os.path.join(DATASET_PATHS["utzappos"], "session 1/1metadata_compositional-split-natural.t7"),
    # "utzappos-s01": os.path.join(DATASET_PATHS["utzappos"], "session 1/01 pairs/01metadata_compositional-split-natural.t7"),
    "utzappos-s2" : os.path.join(DATASET_PATHS["utzappos"], "session 2/2metadata_compositional-split-natural.t7"),
    # "utzappos-s012" : os.path.join(DATASET_PATHS["utzappos"], "session 2/012 pairs/012metadata_compositional-split-natural.t7"),
}

teacher_t7_files = {
    "utzappos-s1" : os.path.join(DATASET_PATHS["utzappos"], "session 0/metadata_compositional-split-natural.t7"),
    "utzappos-s2" : os.path.join(DATASET_PATHS["utzappos"], "session 1/1metadata_compositional-split-natural.t7"),
}

train_val_test_txtfiles = {
    "utzappos-s0" : os.path.join(DATASET_PATHS["utzappos"], "session 0"),
    "utzappos-s1" : os.path.join(DATASET_PATHS["utzappos"], "session 1/01 pairs"),
    "utzappos-s2" : os.path.join(DATASET_PATHS["utzappos"], "session 2/012 pairs"),
}

ALL_ATTR_OBJ_PAIRS = {
     "utzappos-s0": (8, 6),
     "utzappos-s1": (8+4, 6+3), 
     "utzappos-s2": (8+4+4, 6+3+3),
}

#in trainpairs.txt file.
TRAIN_PAIRS = {
     "utzappos-s0": 24,
     "utzappos-s1": 24+27, 
     "utzappos-s2": 24+27+32
}

ALL_ATTR_OBJ = {

    "utzappos-s0":
    {   
        "all_attrs": ['Cotton', 'Faux.Fur', 'Full.grain.leather', 'Leather',  'Nylon', 'Patent.Leather',  'Suede', 'Synthetic' ], 
        "all_objs": ['Boots.Mid-Calf', 'Shoes.Boat.Shoes',  'Shoes.Flats', 'Shoes.Loafers',  'Shoes.Sneakers.and.Athletic.Shoes', 'Slippers'],
        "all_pairs" : [('Cotton', 'Boots.Mid-Calf'), ('Cotton', 'Shoes.Flats'), ('Cotton', 'Shoes.Loafers'), ('Cotton', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Faux.Fur', 'Boots.Mid-Calf'), ('Faux.Fur', 'Slippers'), ('Full.grain.leather', 'Boots.Mid-Calf'), ('Full.grain.leather', 'Shoes.Boat.Shoes'), ('Full.grain.leather', 'Shoes.Flats'), ('Full.grain.leather', 'Shoes.Loafers'), ('Full.grain.leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Boots.Mid-Calf'), ('Leather', 'Shoes.Boat.Shoes'), ('Leather', 'Shoes.Flats'), ('Leather', 'Shoes.Loafers'), ('Leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Slippers'), ('Nylon', 'Boots.Mid-Calf'), ('Nylon', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Nylon', 'Slippers'), ('Patent.Leather', 'Boots.Mid-Calf'), ('Patent.Leather', 'Shoes.Flats'), ('Patent.Leather', 'Shoes.Loafers'), ('Suede', 'Boots.Mid-Calf'), ('Suede', 'Shoes.Boat.Shoes'), ('Suede', 'Shoes.Flats'), ('Suede', 'Shoes.Loafers'), ('Suede', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Suede', 'Slippers'), ('Synthetic', 'Boots.Mid-Calf'), ('Synthetic', 'Shoes.Flats'), ('Synthetic', 'Shoes.Loafers'), ('Synthetic', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Synthetic', 'Slippers')]
    },

    "utzappos-s1":
    {   
        "all_attrs": ['Cotton', 'Faux.Fur', 'Full.grain.leather', 'Leather',  'Nylon', 'Patent.Leather',  'Suede', 'Synthetic' , 'Canvas', 'Faux.Leather', 'Nubuck', 'Satin'],
        "all_objs": ['Boots.Mid-Calf', 'Shoes.Boat.Shoes',  'Shoes.Flats', 'Shoes.Loafers',  'Shoes.Sneakers.and.Athletic.Shoes', 'Slippers', 'Boots.Ankle','Shoes.Clogs.and.Mules','Shoes.Oxfords'],
        "all_pairs": [('Cotton', 'Boots.Mid-Calf'), ('Cotton', 'Shoes.Flats'), ('Cotton', 'Shoes.Loafers'), ('Cotton', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Faux.Fur', 'Boots.Mid-Calf'), ('Faux.Fur', 'Slippers'), ('Full.grain.leather', 'Boots.Mid-Calf'), ('Full.grain.leather', 'Shoes.Boat.Shoes'), ('Full.grain.leather', 'Shoes.Flats'), ('Full.grain.leather', 'Shoes.Loafers'), ('Full.grain.leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Boots.Mid-Calf'), ('Leather', 'Shoes.Boat.Shoes'), ('Leather', 'Shoes.Flats'), ('Leather', 'Shoes.Loafers'), ('Leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Slippers'), ('Nylon', 'Boots.Mid-Calf'), ('Nylon', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Nylon', 'Slippers'), ('Patent.Leather', 'Boots.Mid-Calf'), ('Patent.Leather', 'Shoes.Flats'), ('Patent.Leather', 'Shoes.Loafers'), ('Suede', 'Boots.Mid-Calf'), ('Suede', 'Shoes.Boat.Shoes'), ('Suede', 'Shoes.Flats'), ('Suede', 'Shoes.Loafers'), ('Suede', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Suede', 'Slippers'), ('Synthetic', 'Boots.Mid-Calf'), ('Synthetic', 'Shoes.Flats'), ('Synthetic', 'Shoes.Loafers'), ('Synthetic', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Synthetic', 'Slippers'),('Canvas', 'Boots.Ankle'), ('Canvas', 'Boots.Mid-Calf'), ('Canvas', 'Shoes.Boat.Shoes'), ('Canvas', 'Shoes.Clogs.and.Mules'), ('Canvas', 'Shoes.Flats'), ('Canvas', 'Shoes.Loafers'), ('Canvas', 'Shoes.Oxfords'), ('Canvas', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Faux.Fur', 'Boots.Ankle'), ('Faux.Leather', 'Boots.Ankle'), ('Faux.Leather', 'Boots.Mid-Calf'), ('Faux.Leather', 'Shoes.Flats'), ('Faux.Leather', 'Shoes.Loafers'), ('Faux.Leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Full.grain.leather', 'Boots.Ankle'), ('Full.grain.leather', 'Shoes.Clogs.and.Mules'), ('Full.grain.leather', 'Shoes.Oxfords'), ('Leather', 'Boots.Ankle'), ('Leather', 'Shoes.Clogs.and.Mules'), ('Leather', 'Shoes.Oxfords'), ('Nubuck', 'Boots.Ankle'), ('Nubuck', 'Boots.Mid-Calf'), ('Nubuck', 'Shoes.Boat.Shoes'), ('Nubuck', 'Shoes.Clogs.and.Mules'), ('Nubuck', 'Shoes.Flats'), ('Nubuck', 'Shoes.Loafers'), ('Nubuck', 'Shoes.Oxfords'), ('Nubuck', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Nylon', 'Boots.Ankle'), ('Patent.Leather', 'Boots.Ankle'), ('Patent.Leather', 'Shoes.Clogs.and.Mules'), ('Patent.Leather', 'Shoes.Oxfords'), ('Satin', 'Shoes.Flats'), ('Suede', 'Boots.Ankle'), ('Suede', 'Shoes.Clogs.and.Mules'), ('Suede', 'Shoes.Oxfords'), ('Synthetic', 'Boots.Ankle'), ('Synthetic', 'Shoes.Clogs.and.Mules'), ('Synthetic', 'Shoes.Oxfords')]
        
    },

    "utzappos-s2":
    {   
        "all_attrs": ['Cotton', 'Faux.Fur', 'Full.grain.leather', 'Leather',  'Nylon', 'Patent.Leather',  'Suede', 'Synthetic' , 'Canvas', 'Faux.Leather', 'Nubuck', 'Satin', 'Hair.Calf', 'Rubber', 'Sheepskin',  'Wool'],
        "all_objs": ['Boots.Mid-Calf', 'Shoes.Boat.Shoes',  'Shoes.Flats', 'Shoes.Loafers',  'Shoes.Sneakers.and.Athletic.Shoes', 'Slippers', 'Boots.Ankle','Shoes.Clogs.and.Mules','Shoes.Oxfords','Boots.Knee.High',  'Sandals', 'Shoes.Heels'],
        "all_pairs": [('Cotton', 'Boots.Mid-Calf'), ('Cotton', 'Shoes.Flats'), ('Cotton', 'Shoes.Loafers'), ('Cotton', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Faux.Fur', 'Boots.Mid-Calf'), ('Faux.Fur', 'Slippers'), ('Full.grain.leather', 'Boots.Mid-Calf'), ('Full.grain.leather', 'Shoes.Boat.Shoes'), ('Full.grain.leather', 'Shoes.Flats'), ('Full.grain.leather', 'Shoes.Loafers'), ('Full.grain.leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Boots.Mid-Calf'), ('Leather', 'Shoes.Boat.Shoes'), ('Leather', 'Shoes.Flats'), ('Leather', 'Shoes.Loafers'), ('Leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Leather', 'Slippers'), ('Nylon', 'Boots.Mid-Calf'), ('Nylon', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Nylon', 'Slippers'), ('Patent.Leather', 'Boots.Mid-Calf'), ('Patent.Leather', 'Shoes.Flats'), ('Patent.Leather', 'Shoes.Loafers'), ('Suede', 'Boots.Mid-Calf'), ('Suede', 'Shoes.Boat.Shoes'), ('Suede', 'Shoes.Flats'), ('Suede', 'Shoes.Loafers'), ('Suede', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Suede', 'Slippers'), ('Synthetic', 'Boots.Mid-Calf'), ('Synthetic', 'Shoes.Flats'), ('Synthetic', 'Shoes.Loafers'), ('Synthetic', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Synthetic', 'Slippers'),('Canvas', 'Boots.Ankle'), ('Canvas', 'Boots.Mid-Calf'), ('Canvas', 'Shoes.Boat.Shoes'), ('Canvas', 'Shoes.Clogs.and.Mules'), ('Canvas', 'Shoes.Flats'), ('Canvas', 'Shoes.Loafers'), ('Canvas', 'Shoes.Oxfords'), ('Canvas', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Faux.Fur', 'Boots.Ankle'), ('Faux.Leather', 'Boots.Ankle'), ('Faux.Leather', 'Boots.Mid-Calf'), ('Faux.Leather', 'Shoes.Flats'), ('Faux.Leather', 'Shoes.Loafers'), ('Faux.Leather', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Full.grain.leather', 'Boots.Ankle'), ('Full.grain.leather', 'Shoes.Clogs.and.Mules'), ('Full.grain.leather', 'Shoes.Oxfords'), ('Leather', 'Boots.Ankle'), ('Leather', 'Shoes.Clogs.and.Mules'), ('Leather', 'Shoes.Oxfords'), ('Nubuck', 'Boots.Ankle'), ('Nubuck', 'Boots.Mid-Calf'), ('Nubuck', 'Shoes.Boat.Shoes'), ('Nubuck', 'Shoes.Clogs.and.Mules'), ('Nubuck', 'Shoes.Flats'), ('Nubuck', 'Shoes.Loafers'), ('Nubuck', 'Shoes.Oxfords'), ('Nubuck', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Nylon', 'Boots.Ankle'), ('Patent.Leather', 'Boots.Ankle'), ('Patent.Leather', 'Shoes.Clogs.and.Mules'), ('Patent.Leather', 'Shoes.Oxfords'), ('Satin', 'Shoes.Flats'), ('Suede', 'Boots.Ankle'), ('Suede', 'Shoes.Clogs.and.Mules'), ('Suede', 'Shoes.Oxfords'), ('Synthetic', 'Boots.Ankle'), ('Synthetic', 'Shoes.Clogs.and.Mules'), ('Synthetic', 'Shoes.Oxfords'),('Canvas', 'Sandals'), ('Cotton', 'Sandals'), ('Cotton', 'Shoes.Heels'), ('Faux.Leather', 'Boots.Knee.High'), ('Faux.Leather', 'Sandals'), ('Faux.Leather', 'Shoes.Heels'), ('Full.grain.leather', 'Boots.Knee.High'), ('Full.grain.leather', 'Sandals'), ('Full.grain.leather', 'Shoes.Heels'), ('Hair.Calf', 'Boots.Ankle'), ('Hair.Calf', 'Sandals'), ('Hair.Calf', 'Shoes.Flats'), ('Hair.Calf', 'Shoes.Heels'), ('Hair.Calf', 'Shoes.Loafers'), ('Leather', 'Boots.Knee.High'), ('Leather', 'Sandals'), ('Leather', 'Shoes.Heels'), ('Nubuck', 'Sandals'), ('Nubuck', 'Shoes.Heels'), ('Nylon', 'Boots.Knee.High'), ('Nylon', 'Sandals'), ('Patent.Leather', 'Sandals'), ('Patent.Leather', 'Shoes.Heels'), ('Rubber', 'Boots.Ankle'), ('Rubber', 'Boots.Knee.High'), ('Rubber', 'Boots.Mid-Calf'), ('Rubber', 'Sandals'), ('Rubber', 'Shoes.Flats'), ('Rubber', 'Shoes.Sneakers.and.Athletic.Shoes'), ('Satin', 'Sandals'), ('Satin', 'Shoes.Heels'), ('Sheepskin', 'Boots.Ankle'), ('Sheepskin', 'Boots.Mid-Calf'), ('Sheepskin', 'Slippers'), ('Suede', 'Boots.Knee.High'), ('Suede', 'Sandals'), ('Suede', 'Shoes.Heels'), ('Synthetic', 'Boots.Knee.High'), ('Synthetic', 'Sandals'), ('Synthetic', 'Shoes.Heels'), ('Wool', 'Shoes.Clogs.and.Mules'), ('Wool', 'Shoes.Sneakers.and.Athletic.Shoes'),('Wool', 'Slippers')]
    },
}

import json 

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
    "utzappos-s0" :
     {
        "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 0/idx/updated_obj2idx_session0.json")),
        "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 0/idx/updated_attr2idx_session0.json")),
        "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["utzappos"], "session 0/idx/updated_pair2idx_session0.json")),
   },
    
    "utzappos-s1" : 
     {
        "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 1/idx/updated_obj2idx_session1.json")),
        "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 1/idx/updated_attr2idx_session1.json")),
        "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["utzappos"], "session 1/idx/updated_pair2idx_session1.json")),
   },

    "utzappos-s2" : 
     {
        "g_obj2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 2/idx/updated_obj2idx_session2.json")),
        "g_attr2idx" : json_load(os.path.join(DATASET_PATHS["utzappos"], "session 2/idx/updated_attr2idx_session2.json")),
        "g_pair2idx" : load_pair2idx(os.path.join(DATASET_PATHS["utzappos"], "session 2/idx/updated_pair2idx_session2.json")),
    },
}