#!/bin/bash
#set batch size. 

# #session1
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset utzappos \
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
#   --save_path data/model/utzappos-b4/session1/ \
#   --soft_embeddings data/model/utzappos-b4/session0/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session utzappos-s1  \
#   --teacher_session utzappos-s0 | tee data/model/utzappos-b4/b4_s1_train_meta1_pairs1.log

# #session2
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset utzappos \
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
#   --save_path data/model/utzappos-b4/session2/ \
#   --soft_embeddings data/model/utzappos-b4/session1/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session utzappos-s2  \
#   --teacher_session utzappos-s1 | tee data/model/utzappos-b4/b4_s2_train_meta2_pairs2.log