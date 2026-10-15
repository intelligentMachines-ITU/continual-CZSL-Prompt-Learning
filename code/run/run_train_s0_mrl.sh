[ -n "$BASH_VERSION" ] || exec bash "$0" "$@"   # re-run under bash when started with sh
cd "$(dirname "$0")/.."                         # code/, so train.py, datasets/ and data/ resolve
set -e -o pipefail

# Session 0 training with the Matryoshka (MRL) loss turned on.
# Usage: DATASET=cgqa|mit-states|utzappos BACKBONE=L14|B16|B32 MRL_WEIGHT=0.1 bash run/run_train_s0_mrl.sh
DATASET=${DATASET:-mit-states}
BACKBONE=${BACKBONE:-B16}
MRL_WEIGHT=${MRL_WEIGHT:-0.1}

case $DATASET in
  cgqa)       YML=config/troika/cgqa.yml ;;
  mit-states) YML=config/troika/mit-states.yml ;;
  utzappos)   YML=config/troika/ut-zappos.yml ;;
  *) echo "unknown DATASET: $DATASET" >&2; exit 1 ;;
esac

# ViT-L/14 is loaded from the local checkpoint; ViT-B weights are downloaded by load_clip on first use
case $BACKBONE in
  L14) CLIP_ARCH=model/ViT-L-14.pt ;;
  B16) CLIP_ARCH=ViT-B/16 ;;
  B32) CLIP_ARCH=ViT-B/32 ;;
  *) echo "unknown BACKBONE: $BACKBONE" >&2; exit 1 ;;
esac

OUT=data/model/$DATASET-mrl-$BACKBONE
mkdir -p $OUT/session0
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch $CLIP_ARCH \
  --yml_path $YML \
  --num_workers 4 \
  --seed 0 \
  --save_path $OUT/session0/ \
  --session $DATASET-s0 \
  --teacher_session $DATASET-s0 \
  --use_mrl \
  --mrl_weight $MRL_WEIGHT 2>&1 | tee $OUT/s0_train.log
