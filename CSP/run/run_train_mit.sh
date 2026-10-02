#!/bin/bash
#set batch size. 

# #session1
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset mit-states \
#   --experiment_name csp \
#   --clip_model ViT-L/14 \
#   --seed 0 \
#   --epochs 20 \
#   --lr 5e-05 \
#   --attr_dropout 0.3 \
#   --weight_decay 0.00001 \
#   --train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --context_length 8 \
#   --save_path data/model/mit-states/session1/ \
#   --soft_embeddings data/model/mit-states/session0/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session mit-states-s1  \
#   --teacher_session mit-states-s0 | tee data/model/mit-states/b4_s1_train_meta1_pairs1.log

# #session2
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset mit-states \
#   --experiment_name csp \
#   --clip_model ViT-L/14 \
#   --seed 0 \
#   --epochs 20 \
#   --lr 5e-05 \
#   --attr_dropout 0.3 \
#   --weight_decay 0.00001 \
#   --train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --context_length 8 \
#   --save_path data/model/mit-states/session2/ \
#   --soft_embeddings data/model/mit-states/session1/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session mit-states-s2  \
#   --teacher_session mit-states-s1 | tee data/model/mit-states/b4_s2_train_meta2_pairs2.log


# #session3
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset mit-states \
#   --experiment_name csp \
#   --clip_model ViT-L/14 \
#   --seed 0 \
#   --epochs 20 \
#   --lr 5e-05 \
#   --attr_dropout 0.3 \
#   --weight_decay 0.00001 \
#   --train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --context_length 8 \
#   --save_path data/model/mit-states/session3/ \
#   --soft_embeddings data/model/mit-states/session2/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session mit-states-s3  \
#   --teacher_session mit-states-s2 | tee data/model/mit-states/b4_s3_train_meta3_pairs3.log
