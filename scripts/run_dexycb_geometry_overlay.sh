#!/usr/bin/env bash
# Run in the repository after applying the patch. No training is started.
set -euo pipefail
cd "$(dirname "$0")/.."
audit_python=${HUG_PYTHON:-/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python}
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

"$audit_python" -m unittest discover -s tests -p test_geometry_overlay.py -v
"$audit_python" -m unittest discover -s tests -p test_native_mano.py -v
"$audit_python" -m unittest discover -s tests -p test_repair_shape_gt_geometry.py -v

split_root=/root/code/vepfs/dataset/hand_recon_hug/splits_v2
overlay_root=/root/code/vepfs/dataset/hand_recon_hug
common=(
  --dataset-root "$overlay_root/dexycb_v4_fullres_shape_gt"
  --train-list "$split_root/dexycb_train.clean.txt"
  --val-list "$split_root/dexycb_val.clean.txt"
  --test-list "$split_root/dexycb_test.clean.txt"
  --workers 8 --torch-threads 2 --batch-size 256
)

"$audit_python" scripts/build_dexycb_geometry_overlay.py "${common[@]}" \
  --per-split 512 --out-dir "$overlay_root/dexycb_native_overlay_smoke_v1" --resume
"$audit_python" scripts/build_dexycb_geometry_overlay.py "${common[@]}" \
  --out-dir "$overlay_root/dexycb_native_overlay_v1" --resume
