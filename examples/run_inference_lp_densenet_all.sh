#!/bin/bash
set -euo pipefail

# =========================================================
# Run inference/evaluation for all AMOS classification targets
# after linear probing has been trained.
#
# This script loads the LP checkpoints from RESULTS_ROOT and
# evaluates the specified split.
#
# Usage:
#   bash scripts/run_inference_lp_densenet_all.sh
#
# Optional environment variables:
#   ROOT         : project root directory
#   EMBEDS_ROOT  : directory containing extracted .h5 feature files
#   LABELS_ROOT  : directory containing downstream labels
#   RESULTS_ROOT : directory containing LP checkpoints/results
#   SPLIT        : data split to evaluate, e.g., val or test
#   BATCH_SIZE   : inference batch size
#   NUM_WORKERS  : number of dataloader workers
#   PYTHON       : python executable
# =========================================================

ROOT="${ROOT:-/data/workspace/m1461010/CVPR26-3DCTFMCompetition}"
PYTHON="${PYTHON:-python}"

EMBEDS_ROOT="${EMBEDS_ROOT:-$ROOT/densenet_new/lp_features_window}"
LABELS_ROOT="${LABELS_ROOT:-$ROOT/AMOS-clf-tr-val/labels}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/densenet_new/lp_results_window}"

INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-$ROOT/cvpr26_inference_LP.py}"

SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TARGETS=(
  adrenal_hyperplasia
  ascites
  atherosclerosis
  cholecystitis
  colorectal_cancer
  fatty_liver
  gallstone
  hydronephrosis
  kidney_stone
  liver_calcifications
  liver_cyst
  liver_lesion
  lymphadenopathy
  renal_cyst
  splenomegaly
)

echo "Root directory       : $ROOT"
echo "Embeddings directory : $EMBEDS_ROOT"
echo "Labels directory     : $LABELS_ROOT"
echo "Results directory    : $RESULTS_ROOT"
echo "Inference script     : $INFERENCE_SCRIPT"
echo "Evaluation split     : $SPLIT"

for target in "${TARGETS[@]}"; do
  echo "======================================"
  echo "Running LP inference for target: $target"
  echo "======================================"

  "$PYTHON" "$INFERENCE_SCRIPT" \
    --embeds_root "$EMBEDS_ROOT" \
    --labels_root "$LABELS_ROOT" \
    --target "$target" \
    --split "$SPLIT" \
    --ckpt_dir "$RESULTS_ROOT/$target/results" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
done

echo "All LP inference jobs finished."
