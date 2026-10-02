#!/bin/bash
#set batch size. 

#session0
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --dataset mit-states \
  --experiment_name csp \
  --clip_model ViT-L/14 \
  --seed 0 \
  --epochs 20 \
  --lr 5e-05 \
  --attr_dropout 0.3 \
  --weight_decay 0.00001 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --context_length 8 \
  --save_path data/model/mit-states/session0/ \
  --save_every_n 5 \
  --session mit-states-s0  \
  --teacher_session mit-states-s0 | tee data/model/mit-states/s0_train_meta0_pairs0.log
