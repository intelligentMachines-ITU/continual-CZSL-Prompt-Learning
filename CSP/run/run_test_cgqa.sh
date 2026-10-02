#!/bin/bash

# ---------------------------------------
# Note: Before testing, please check {dataset_name}_read_datasets.py in datasets folder and enter the correct data paths for t7_files.
# t7_files path determines which metadata will be used by model for testing. 
# For example when use session 1 data only, use the path for cgqa-s1 = "../1metadata_compositional-split-natural.t7" and 
# when use 01 accumulative data, use cgqa-s1 = "../01metadata_compositional-split-natural.t7"
# ---------------------------------------

#session-0 cgqa
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --dataset cgqa \
  --clip_model ViT-L/14 \
  --soft_embeddings data/model/cgqa/session0/soft_embeddings_epoch_20.pt \
  --context_length 16 \
  --text_encoder_batch_size 36 \
  --eval_batch_size 16 \
  --experiment_name csp \
  --session cgqa-s0  \
  --teacher_session cgqa-s0 | tee data/model/cgqa/session0/s0_test_meta0_pairs0.log

# #session-1 cgqa
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset cgqa \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/cgqa/session1/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session cgqa-s1  \
#   --teacher_session cgqa-s0 | tee data/model/cgqa/session1/s1_meta1_test.log
#   # --teacher_session cgqa-s0 | tee data/model/cgqa/session1/s1_meta01_test.log

# #session-2 cgqa
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset cgqa \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/cgqa/session2/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session cgqa-s2  \
#   --teacher_session cgqa-s1 | tee data/model/cgqa/session2/s2_meta2_test.log
#   # --teacher_session cgqa-s1 | tee data/model/cgqa/session2/s2_meta012_test.log

# #session-3 cgqa
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset cgqa \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/cgqa/session3/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session cgqa-s3  \
#   --teacher_session cgqa-s2 | tee data/model/cgqa/session3/s3_meta3_test.log
#   # --teacher_session cgqa-s2 | tee data/model/cgqa/session3/s3_meta0123_test.log

# #session-4 cgqa
# CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
#   --dataset cgqa \
#   --clip_model ViT-L/14 \
#   --soft_embeddings data/model/cgqa/session4/soft_embeddings_epoch_20.pt \
#   --context_length 16 \
#   --text_encoder_batch_size 36 \
#   --eval_batch_size 16 \
#   --experiment_name csp \
#   --session cgqa-s4  \
#   --teacher_session cgqa-s3 | tee data/model/cgqa/session4/s4_meta4_test.log
#   # --teacher_session cgqa-s3 | tee data/model/cgqa/session4/s4_meta01234_test.log