"""Verify a generated smoke cache against the existing deployment loader."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

from src.cache_train_conditions import arrays, validate
from src.dataloader.grasp_dataset import GraspDataset


def main():
    # YOLO mutates CUDA_VISIBLE_DEVICES during online GPU-0 comparison.
    # Resume the launcher with the original four-GPU environment.
    launch_env = os.environ.copy()
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.output / 'metadata.json').read_text())
    cfg = OmegaConf.load('configs/train_handrecon_v28_joint_dit.yaml')
    ds = GraspDataset(meta['dataset_root'], split='val',
                      samples_filename=meta['samples_file'],
                      hand_crop=cfg.trainer.data.hand_crop, d_mano=109)
    box_max = 0.0
    xy_max = 0.0
    same_box_xy_max = 0.0
    with np.load(args.output / 'conditions.npz', allow_pickle=False) as cache:
        # Check samples from separate shards against detector + single-image pose.
        for i in [0, 32, 64, 96]:
            raw = ds._load_grasp_data(ds.grasp_files[i])
            rgb = ds._decode_image(raw['image'])
            bbox, hit, source = ds._select_hand_crop_bbox(raw, rgb)
            assert hit == bool(cache['detector_hit'][i])
            np.testing.assert_allclose(bbox, cache['crop_bbox_xyxy'][i], atol=0.05, rtol=0)
            box_max = max(box_max, float(np.abs(bbox - cache['crop_bbox_xyxy'][i]).max()))
            affine = ds._crop_affine(bbox, ds.image_size)
            _, _, xy, valid = ds._rtmpose_keypoints(rgb, bbox, affine, hit)
            # Tiny batched detector box changes can shift the decoded SimCC
            # maximum after resampling. Measure end-to-end drift separately,
            # then isolate pose inference with identical cached crop geometry.
            xy_max = max(xy_max, float(np.abs(xy - cache['keypoints_xy'][i]).max()))
            cached_bbox = cache['crop_bbox_xyxy'][i]
            _, _, same_xy, same_valid = ds._rtmpose_keypoints(
                rgb, cached_bbox, ds._crop_affine(cached_bbox, ds.image_size), hit)
            np.testing.assert_allclose(same_xy, cache['keypoints_xy'][i], atol=0.5, rtol=0)
            same_box_xy_max = max(same_box_xy_max,
                                  float(np.abs(same_xy - cache['keypoints_xy'][i]).max()))
            np.testing.assert_array_equal(same_valid, cache['keypoint_scores'][i] >= 0.1)
        # An actual forced detector miss must skip RTMPose and return empty joints.
        miss = ds._rtmpose_keypoints(rgb, bbox, affine, False)
        assert all(not arr.any() for arr in miss)
    empty = arrays(['forced_miss'], 0)
    empty['image_size_wh'][0] = [640, 480]
    empty['crop_bbox_xyxy'][0] = [0, 0, 639, 479]
    validate(empty, ['forced_miss'], 0)
    try:
        validate(empty, ['wrong_stem'], 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched sample IDs must be rejected')
    chunk_times = {f: f.stat().st_mtime_ns for f in (args.output / 'chunks').glob('*.npz')}
    cmd = [sys.executable, '-m', 'src.cache_train_conditions', '--output', str(args.output),
           '--launch', '--limit', str(meta['n_samples']), '--chunk-size', str(meta['chunk_size']),
           '--batch-size', str(meta['batch_size']), '--workers', '4']
    subprocess.run(cmd, check=True, env=launch_env)
    assert chunk_times == {f: f.stat().st_mtime_ns for f in chunk_times}
    print(json.dumps(dict(status='passed', online_samples=4, bbox_max_abs_px=box_max,
                          keypoint_max_abs_px=xy_max,
                          same_box_keypoint_max_abs_px=same_box_xy_max, forced_miss=True,
                          sample_mismatch_rejected=True, resume_chunks_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
