#!/bin/bash

# # Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate troika

# # Base path
BASE="$(cd "$(dirname "$0")/.." && pwd)"   # code/ folder of this checkout

# python -u $BASE/test.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_1vis_0_1txts0_SPADT_up_down_proj_updated01Test/s0 \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/data/model/s1model/0_65KD_0_1vis_0_1txts0_SPADT_up_down_proj_updated01Test/val_best.pt
#     # --load_model $BASE/data/model/s1model/0_65KD_0_1vis_0_1txt_SPADT_up_down_proj_updated01Test/val_best.pt


#make mean_attr, mean_obj = True, True in troika.py
# mean_feat = batch_patch[:,1:].mean(dim=1) #sara
python -u $BASE/test.py \
   --clip_arch $BASE/model/ViT-L-14.pt \
   --dataset_path $BASE/download_data/data/ut-zappos \
   --save_path $BASE/data/model/s0model/s0_att_mean_cls_guided_obj_mean_cls_guided_comp_mean_cls_guided/s0 \
   --yml_path $BASE/config/troika/ut-zappos.yml \
   --num_workers 4 \
   --seed 0 \
   --load_model $BASE/data/model/s0model/s0_att_mean_cls_guided_obj_mean_cls_guided_comp_mean_cls_guided/val_best.pt
   2>&1 | tee $BASE/data/model/s0model/s0_att_mean_cls_guided_obj_mean_cls_guided_comp_mean_cls_guided_test.log