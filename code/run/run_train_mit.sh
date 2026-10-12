#!/bin/bash
# Train Troika on the CCZSL mit-states benchmark, sessions 0..3.
# Session 0: train.py. Sessions 1+: session_train.py, with the previous session's val_best.pt
# as teacher and student init. Hyperparameters (batch size, lr, epochs, alpha) come from
# config/troika/mit-states.yml, which overrides command-line flags; comment out sessions you don't need.
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so train.py, datasets/ and data/ resolve
set -e -o pipefail

OUT=data/model/mit-states
mkdir -p $OUT

#session0
mkdir -p $OUT/session0
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/mit-states.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session0/ \
  --session mit-states-s0 \
  --teacher_session mit-states-s0 2>&1 | tee $OUT/s0_train.log

#session1
mkdir -p $OUT/session1
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/mit-states.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session1/ \
  --load_model $OUT/session0/val_best.pt \
  --session mit-states-s1 \
  --teacher_session mit-states-s0 2>&1 | tee $OUT/s1_train.log

#session2
mkdir -p $OUT/session2
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/mit-states.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session2/ \
  --load_model $OUT/session1/val_best.pt \
  --session mit-states-s2 \
  --teacher_session mit-states-s1 2>&1 | tee $OUT/s2_train.log

#session3
mkdir -p $OUT/session3
CUDA_VISIBLE_DEVICES=0 python -u session_train.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/mit-states.yml \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session3/ \
  --load_model $OUT/session2/val_best.pt \
  --session mit-states-s3 \
  --teacher_session mit-states-s2 2>&1 | tee $OUT/s3_train.log
