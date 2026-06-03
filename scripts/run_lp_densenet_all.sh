#!/bin/bash
set -euo pipefail

# =========================================================
# Run linear probing training for all AMOS classification targets.
#
# This script assumes that DenseNet feature files have already been
# extracted and saved as .h5 files.
#
# Usage:
#   bash scripts/run_lp_densenet_all.sh
#
# Optional environment variables:
#   ROOT        : project root directory
#   EMBEDS_ROOT : directory containing extracted .h5 feature files
#   LABELS_ROOT : directory containing downstream labels
#   OUT_ROOT    : output directory for LP checkpoints and results
#   PYTHON      : python executable
# =========================================================

ROOT="${ROOT:-/data/workspace/m1461010/CVPR26-3DCTFMCompetition}"
PYTHON="${PYTHON:-python}"

EMBEDS_ROOT="${EMBEDS_ROOT:-$ROOT/densenet_new/lp_features_window}"
LABELS_ROOT="${LABELS_ROOT:-$ROOT/AMOS-clf-tr-val/labels}"
OUT_ROOT="${OUT_ROOT:-$ROOT/densenet_new/lp_results_window}"

RUN_LP_SCRIPT="${RUN_LP_SCRIPT:-$ROOT/run_LP.py}"

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

echo "Root directory      : $ROOT"
echo "Embeddings directory: $EMBEDS_ROOT"
echo "Labels directory    : $LABELS_ROOT"
echo "Output directory    : $OUT_ROOT"
echo "LP script           : $RUN_LP_SCRIPT"

mkdir -p "$OUT_ROOT"

for target in "${TARGETS[@]}"; do
  echo "======================================"
  echo "Running linear probing for target: $target"
  echo "======================================"

  "$PYTHON" "$RUN_LP_SCRIPT" \
    --embeds_root "$EMBEDS_ROOT" \
    --labels_root "$LABELS_ROOT" \
    --target "$target" \
    --out_dir "$OUT_ROOT/$target/results"
done

echo "All linear probing jobs finished."
