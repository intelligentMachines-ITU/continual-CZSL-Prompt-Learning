#!/bin/bash

# ---------------------------------------
# Note: Before testing, please check {dataset_name}_read_datasets.py in datasets folder and enter the correct data paths for t7_files.
# t7_files path determines which metadata will be used by model for testing. 
# For example when use session 1 data only, use the path for utzappos-s1 = "../1metadata_compositional-split-natural.t7" and 
# when use 01 accumulative data, use utzappos-s1 = "../01metadata_compositional-split-natural.t7"
# ---------------------------------------

#session-0 utzappos
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --dataset utzappos \
  --clip_model ViT-L/14 \
  --soft_embeddings data/model/utzappos-b4/session0/soft_embeddings_epoch_20.pt \
  --context_length 16 \
  --text_encoder_batch_size 36 \
  --eval_batch_size 16 \
  --experiment_name csp \
  --session utzappos-s0  \
  --teacher_session utzappos-s0 | tee data/model/utzappos-b4/session0/s0_test.log

# #session-1 utzappos
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset utzappos \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/utzappos-b4/session1/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session utzappos-s1  \
#   --teacher_session utzappos-s0 | tee data/model/utzappos-b4/session1/s1_meta1_test.log
#   #--teacher_session utzappos-s0 | tee data/model/utzappos-b4/session1/s01_meta1_test.log

# #session-2 utzappos
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset utzappos \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/utzappos-b4/session2/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session utzappos-s2  \
#   --teacher_session utzappos-s1 | tee data/model/utzappos-b4/session2/s2_meta2_test.log
#   #--teacher_session utzappos-s1 | tee data/model/utzappos-b4/session2/s2_meta012_test.log