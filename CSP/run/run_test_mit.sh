#!/bin/bash

# ---------------------------------------
# Note: Before testing, please check {dataset_name}_read_datasets.py in datasets folder and enter the correct data paths for t7_files.
# t7_files path determines which metadata will be used by model for testing. 
# For example when use session 1 data only, use the path for mit-states-s1 = "../1metadata_compositional-split-natural.t7" and 
# when use 01 accumulative data, use mit-states-s1 = "../01metadata_compositional-split-natural.t7"
# ---------------------------------------

#session-0 mit
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --dataset mit-states \
  --clip_model ViT-L/14 \
  --soft_embeddings data/model/mit-stattes-b4/session0/soft_embeddings_epoch_20.pt \
  --context_length 16 \
  --text_encoder_batch_size 36 \
  --eval_batch_size 16 \
  --experiment_name csp \
  --session mit-states-s0  \
  --teacher_session mit-states-s0 | tee data/model/mit-stattes-b4/session0/s0_test.log

# session 1       change session names
# import          from datasets.mit_read_datasets import *  in evaluate
# check t7 file path 

# #session-1 mit
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset mit-states \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/mit-stattes-b4/session1/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session mit-states-s1  \
#   --teacher_session mit-states-s0 | tee data/model/mit-stattes-b4/session1/s1_meta1_test.log

# #session-2 mit
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset mit-states \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/mit-stattes-b4/session2/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session mit-states-s2  \
#   --teacher_session mit-states-s1 | tee data/model/mit-stattes-b4/session2/s2_meta2_test.log

#session-3 mit
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --dataset mit-states \
  --clip_model ViT-L/14 \
  --soft_embeddings data/model/mit-stattes-b4/session3/soft_embeddings_epoch_20.pt \
  --context_length 16 \
  --text_encoder_batch_size 36 \
  --eval_batch_size 16 \
  --experiment_name csp \
  --session mit-states-s3  \
  --teacher_session mit-states-s2 | tee data/model/mit-stattes-b4/session3/s3_meta3_test.log


  