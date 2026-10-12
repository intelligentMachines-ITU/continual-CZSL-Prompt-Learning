#!/bin/bash

# # Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate troika

# Base path
BASE="$(cd "$(dirname "$0")/.." && pwd)"   # code/ folder of this checkout

python -u $BASE/train.py \
    --clip_arch $BASE/model/ViT-L-14.pt \
    --dataset_path $BASE/download_data/CCZSL_benchmark/ut-zappos \
    --save_path $BASE/data/model/utzappos_s0/ \
    --yml_path $BASE/config/troika/ut-zappos.yml \
    --num_workers 4 \
    --seed 0 \
   #  --load_model $BASE/model/s0model/val_best.pt

# python -u $BASE/train.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_5SPADT_up_down_proj \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/model/s0model/val_best.pt

# python -u $BASE/train.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_1SPADT_up_down_proj_updated01Test \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/model/s0model/val_best.pt

# python -u $BASE/train.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_1vis_0_5txt_SPADT_up_down_proj_updated01Test \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/model/s0model/val_best.pt

# python -u $BASE/train.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_1vis_0_5txts0_SPADT_up_down_proj_updated01Test \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/model/s0model/val_best.pt

# python -u $BASE/train.py \
#     --clip_arch $BASE/model/ViT-L-14.pt \
#     --dataset_path $BASE/download_data/data/ut-zappos \
#     --save_path $BASE/data/model/s1model/0_65KD_0_001vis_0_1_SPADT_0_2_ratio_as_spu_up_down_proj_updated01Test \
#     --yml_path $BASE/config/troika/ut-zappos.yml \
#     --num_workers 4 \
#     --seed 0 \
#     --load_model $BASE/model/s0model/val_best.pt

# -------- resumed
# #session-0 with attention 
# mean_feat, attn = self.patch_pool(batch_patch)
# batch_img_features = [mean_feat, attr_feat, obj_feat]  

# python -u $BASE/train.py \
#    --clip_arch $BASE/model/ViT-L-14.pt \
#    --dataset_path $BASE/download_data/data/ut-zappos \
#    --save_path $BASE/data/model/s0model/s0_att_mean_cls_guided_obj_mean_cls_guided_comp_cls \
#    --yml_path $BASE/config/troika/ut-zappos.yml \
#    --num_workers 4 \
#    --seed 0 \
#    2>&1 | tee $BASE/data/model/s0model/s0_att_mean_cls_guided_obj_mean_cls_guided_comp_cls_train.log

# #session-0 with mean
# #   mean_feat = batch_patch[:,1:].mean(dim=1) #sara
# python -u $BASE/train.py \
#    --clip_arch $BASE/model/ViT-L-14.pt \
#    --dataset_path $BASE/download_data/data/ut-zappos \
#    --save_path $BASE/data/model/s0model/s0_att_patchmean_cls_guided_obj_cls_comp_cls \
#    --yml_path $BASE/config/troika/ut-zappos.yml \
#    --num_workers 4 \
#    --seed 0 \
#    2>&1 | tee $BASE/data/model/s0model/s0_att_patchmean_cls_guided_obj_cls_comp_cls_train.log


# gdown --fuzzy "https://drive.google.com/file/d/1CwN4NBpvV3Z32nDG-EEtpAcFQCa-JCoI/view?usp=drive_link"