#!/bin/bash
# Train Troika on the ComPIL cgqa-compil split (tasks 0..4).
# Task 0: train.py. Tasks 1+: session_train.py, with the previous task's val_best.pt as teacher and student init.
# Data: data/ComPIL/generated/ (see data/ComPIL/README.md). Hyperparameters come from config/troika/cgqa-compil.yml.
# Run a subset of tasks with: TASKS="2 3 4" bash run_compil/run_train_cgqa_ComPIL.sh
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so train.py, datasets/ and data/ resolve
set -e -o pipefail

DATASET=cgqa-compil
OUT=data/model/ComPIL/cgqa
TASKS=${TASKS:-"0 1 2 3 4"}
ALPHA=${ALPHA:-0.65}

COMMON="--clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/cgqa-compil.yml \
  --num_workers 4 \
  --seed 0"

for k in $TASKS; do
  mkdir -p $OUT/session$k
  if [ "$k" -eq 0 ]; then
    python -u train.py $COMMON \
      --save_path $OUT/session0/ \
      --session $DATASET-s0 \
      --teacher_session $DATASET-s0 2>&1 | tee $OUT/s0_train.log
  else
    t=$((k - 1))
    python -u session_train.py $COMMON \
      --alpha $ALPHA \
      --save_path $OUT/session$k/ \
      --load_model $OUT/session$t/val_best.pt \
      --session $DATASET-s$k \
      --teacher_session $DATASET-s$t 2>&1 | tee $OUT/s${k}_train.log
  fi
done
