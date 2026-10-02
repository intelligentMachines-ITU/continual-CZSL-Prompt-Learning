#!/bin/bash
#set batch size. 

# #session1
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset cgqa \
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
#   --save_path data/model/cgqa/session1/ \
#   --soft_embeddings data/model/cgqa/session0/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session cgqa-s1  \
#   --teacher_session cgqa-s0 | tee data/model/cgqa/b4_s1_train_meta1_pairs01.log

# #session2
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset cgqa \
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
#   --save_path data/model/cgqa/session2/ \
#   --soft_embeddings data/model/cgqa/session1/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session cgqa-s2  \
#   --teacher_session cgqa-s1 | tee data/model/cgqa/b4_s2_train_meta2_pairs012.log

# #session3
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset cgqa \
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
#   --save_path data/model/cgqa/session3/ \
#   --soft_embeddings data/model/cgqa/session2/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session cgqa-s3  \
#   --teacher_session cgqa-s2 | tee data/model/cgqa/b4_s3_train_meta3_pairs0123.log

# #session4
# CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
#   --dataset cgqa \
#   --experiment_name csp \
#   --clip_model ViT-L/14 \
#   --seed 0 \
#   --epochs 20 \
#   --lr 5e-05 \
#   --attr_dropout 0.3 \
#   --weight_decay 0.00001 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 2 \
#   --context_length 8 \
#   --save_path data/model/cgqa/session4-b2/ \
#   --soft_embeddings data/model/cgqa/session3/soft_embeddings_epoch_20.pt \
#   --save_every_n 5 \
#   --session cgqa-s4  \
#   --teacher_session cgqa-s3 | tee data/model/cgqa/b2_s4_train_meta4_pairs01234.log

