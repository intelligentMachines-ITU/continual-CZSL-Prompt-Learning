#!/bin/bash
# Evaluate Troika on the ComPIL cgqa-compil split (tasks 0..4), using the models from run_train_cgqa_ComPIL.sh.
# test.py writes $OUT/session{k}/val_best.closed.json, read by run_compil/summarize_compil.py.
# Run a subset of tasks with: TASKS="4" bash run_compil/run_test_cgqa_ComPIL.sh
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so test.py, datasets/ and data/ resolve
set -e -o pipefail

DATASET=cgqa-compil
OUT=data/model/ComPIL/cgqa
TASKS=${TASKS:-"0 1 2 3 4"}

for k in $TASKS; do
  t=$(( k > 0 ? k - 1 : 0 ))
  python -u test.py \
    --clip_arch model/ViT-L-14.pt \
    --yml_path config/troika/cgqa-compil.yml \
    --num_workers 4 \
    --seed 0 \
    --save_path $OUT/session$k/ \
    --load_model $OUT/session$k/val_best.pt \
    --session $DATASET-s$k \
    --teacher_session $DATASET-s$t 2>&1 | tee $OUT/session$k/s${k}_test.log
done
