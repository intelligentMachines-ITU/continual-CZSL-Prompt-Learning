#!/bin/bash
# Train Troika on the CCZSL cgqa benchmark, sessions 0..4.
# Session 0: train.py. Sessions 1+: session_train.py, with the previous session's val_best.pt
# as teacher and student init. Hyperparameters (batch size, lr, epochs, alpha) come from
# config/troika/cgqa.yml, which overrides command-line flags; comment out sessions you don't need.
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so train.py, datasets/ and data/ resolve
set -e -o pipefail

OUT=data/model/cgqa
mkdir -p $OUT

#session0
mkdir -p $OUT/session0
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session0/ \
  --session cgqa-s0 \
  --teacher_session cgqa-s0 2>&1 | tee $OUT/s0_train.log

#session1
mkdir -p $OUT/session1
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session1/ \
  --load_model $OUT/session0/val_best.pt \
  --session cgqa-s1 \
  --teacher_session cgqa-s0 2>&1 | tee $OUT/s1_train.log

#session2
mkdir -p $OUT/session2
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session2/ \
  --load_model $OUT/session1/val_best.pt \
  --session cgqa-s2 \
  --teacher_session cgqa-s1 2>&1 | tee $OUT/s2_train.log

#session3
mkdir -p $OUT/session3
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session3/ \
  --load_model $OUT/session2/val_best.pt \
  --session cgqa-s3 \
  --teacher_session cgqa-s2 2>&1 | tee $OUT/s3_train.log

#session4
mkdir -p $OUT/session4
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session4/ \
  --load_model $OUT/session3/val_best.pt \
  --session cgqa-s4 \
  --teacher_session cgqa-s3 2>&1 | tee $OUT/s4_train.log
