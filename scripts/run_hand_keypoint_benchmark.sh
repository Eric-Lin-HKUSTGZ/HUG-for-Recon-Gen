#!/usr/bin/env bash
set -euo pipefail

repo="/root/code/HUG-for-Recon-Gen"
hug_python="/root/code/vepfs/miniconda3/envs/hug_mediapipe/bin/python"
pose_python="/root/code/vepfs/miniconda3/envs/pose/bin/python"
output_root="${1:-/root/code/vepfs/HUG-for-Recon-Gen/hand_keypoint_benchmark/dexycb_test_detector_1000}"
max_samples="${2:-1000}"
bbox_source="${3:-detector}"

cd "$repo"

"$hug_python" -m src.benchmark_hand_keypoints prepare \
  --output-root "$output_root" \
  --max-samples "$max_samples" \
  --bbox-source "$bbox_source" \
  --detector-device 0

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$pose_python" \
  -m src.benchmark_hand_keypoints rtmpose \
  --output-root "$output_root" \
  --device cuda:0 \
  --batch-size 64

"$hug_python" -m src.benchmark_hand_keypoints report \
  --output-root "$output_root" \
  --visual-samples 16

echo "Benchmark complete: $output_root"
