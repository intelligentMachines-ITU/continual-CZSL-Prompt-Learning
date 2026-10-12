#!/bin/bash
# Forgetting table (fS) for Troika on the ComPIL cgqa-compil split.
# For every step k and task j <= k, evaluates session k's model on task j's seen test images
# (plus session k's unseen test images) via the extra session $DATASET-s{k}-fs{j}.
# Needs the models from run_train_cgqa_ComPIL.sh. Summarize with:
#   python run_compil/summarize_compil.py cgqa
# Run a subset of steps with: STEPS="3 4" bash run_compil/run_fs_cgqa_ComPIL.sh
[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so test.py, datasets/ and data/ resolve
set -e -o pipefail

DATASET=cgqa-compil
OUT=data/model/ComPIL/cgqa
STEPS=${STEPS:-"0 1 2 3 4"}
mkdir -p $OUT/fs

for k in $STEPS; do
  for j in $(seq 0 $k); do
    # test.py writes <load_model without .pt>.closed.json; a per-cell symlink to the model
    # keeps each cell's JSON separate from the main test's.
    ln -sf ../session$k/val_best.pt $OUT/fs/s${k}_task${j}.pt
    python -u test.py \
      --clip_arch model/ViT-L-14.pt \
      --yml_path config/troika/cgqa-compil.yml \
      --num_workers 4 \
      --seed 0 \
      --save_path $OUT/fs/ \
      --load_model $OUT/fs/s${k}_task${j}.pt \
      --session $DATASET-s$k-fs$j \
      --teacher_session $DATASET-s$k 2>&1 | tee $OUT/fs/s${k}_task${j}.log
  done
done
