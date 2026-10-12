import os

from datasets.compil_read_datasets import build_config

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Defines DATASET_PATHS, t7_files, teacher_t7_files, train_val_test_txtfiles, ALL_ATTR_OBJ_PAIRS,
# TRAIN_PAIRS, ALL_ATTR_OBJ, g_objattrpair2idx_path and selected_session_g_pair2_idx
# for sessions cgqa-compil-s0 .. cgqa-compil-s4.
globals().update(build_config(
    "cgqa-compil",
    images_root=os.path.join(DIR_PATH, "../download_data/CCZSL_benchmark/CGQA/"),  # dir holding images/
    generated_dir=os.path.join(DIR_PATH, "../data/ComPIL/generated/cgqa"),
))
