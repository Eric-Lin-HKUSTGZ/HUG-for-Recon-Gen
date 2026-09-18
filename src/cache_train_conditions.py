"""Offline detector + RTMPose cache. Never uses labels to construct conditions.

Run with --launch to spawn one worker per GPU, resume atomic chunks, and merge.
The final NPZ is ordered exactly like the supplied sample list; load with
np.load(path, allow_pickle=False). Coordinates are original-image pixels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from src.benchmark_hand_keypoints import (
    DEFAULT_DATASET_ROOT, DEFAULT_DETECTOR, DEFAULT_RTMPOSE_CONFIG,
    DEFAULT_RTMPOSE_CHECKPOINT, read_stems, load_sample,
    choose_detector_bbox, expanded_square_bbox, _rtmpose_batch,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, data):
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def metadata(args, stems):
    return dict(schema_version=1, n_samples=len(stems),
                dataset_root=str(args.dataset_root), samples_file=str(args.samples_file),
                samples_sha256=sha256(args.samples_file),
                detector=str(DEFAULT_DETECTOR), detector_sha256=sha256(DEFAULT_DETECTOR),
                rtmpose_config=str(DEFAULT_RTMPOSE_CONFIG),
                rtmpose_config_sha256=sha256(DEFAULT_RTMPOSE_CONFIG),
                rtmpose_checkpoint=str(DEFAULT_RTMPOSE_CHECKPOINT),
                rtmpose_sha256=sha256(DEFAULT_RTMPOSE_CHECKPOINT),
                generator_sha256=sha256(__file__), detector_conf=0.25, detector_iou=0.7,
                detector_selection='highest-confidence right class (>0.5), else any hand',
                detector_imgsz=512, crop_expand=1.5, min_crop_side=24,
                coordinates='original image pixels; xy / xyxy',
                joint_order='wrist, thumb x4, index x4, middle x4, ring x4, pinky x4',
                scores='raw RTMPose scores, no thresholding or clipping',
                detector_miss='pose not run; zero keypoints/scores; full-frame crop',
                chunk_size=args.chunk_size, batch_size=args.batch_size)


def arrays(stems, start):
    n = len(stems)
    return dict(sample=np.asarray(stems), index=np.arange(start, start + n),
                image_size_wh=np.zeros((n, 2), np.int32),
                detector_bbox_xyxy=np.zeros((n, 4), np.float32),
                crop_bbox_xyxy=np.zeros((n, 4), np.float32),
                detector_hit=np.zeros(n, bool), detector_score=np.zeros(n, np.float32),
                detector_class=np.full(n, -1, np.int16),
                keypoints_xy=np.zeros((n, 21, 2), np.float32),
                keypoint_scores=np.zeros((n, 21), np.float32),
                pose_returned=np.zeros(n, bool))


def validate(data, stems, start):
    n = len(stems)
    for key, template in arrays(stems, start).items():
        assert data[key].shape == template.shape, (key, data[key].shape)
        assert data[key].dtype.kind == template.dtype.kind, key
    assert np.array_equal(data['sample'], stems)
    assert np.array_equal(data['index'], np.arange(start, start + n))
    assert np.all(data['image_size_wh'] > 0)
    for key in ('detector_bbox_xyxy', 'crop_bbox_xyxy', 'detector_score',
                'keypoints_xy', 'keypoint_scores'):
        assert np.isfinite(data[key]).all(), key
    hit = data['detector_hit']
    assert np.array_equal(hit, data['pose_returned'])
    assert np.all(data['keypoints_xy'][~hit] == 0)
    assert np.all(data['keypoint_scores'][~hit] == 0)
    box = data['detector_bbox_xyxy'][hit]
    assert np.all(box[:, 2:] > box[:, :2])


def worker(args, stems):
    lock = (args.output / f'worker_{args.rank}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    sys.path.extend(['/root/code/vepfs/miniconda3/envs/pose/lib/python3.10/site-packages',
                     '/root/code/vepfs/third_party/mmpose-1.3.2'])
    import torch
    from ultralytics import YOLO
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmpose.apis import init_model

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.set_device(args.rank)
    detector = YOLO(str(DEFAULT_DETECTOR))
    pose = init_model(str(DEFAULT_RTMPOSE_CONFIG), str(DEFAULT_RTMPOSE_CHECKPOINT),
                      device=f'cuda:{args.rank}',
                      cfg_options={'model.backbone._scope_': 'mmpose'})
    init_default_scope(pose.cfg.get('default_scope', 'mmpose'))
    pose.eval()
    pipeline = Compose(pose.cfg.test_dataloader.dataset.pipeline)
    started = time.monotonic()
    processed = 0
    def read_image(stem):
        # Labels in the source pickle are never used for boxes or keypoints.
        return load_sample(args.dataset_root / f'{stem}.pkl')[1]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk_id, start in enumerate(range(0, len(stems), args.chunk_size)):
            if chunk_id % args.workers != args.rank:
                continue
            subset = stems[start:start + args.chunk_size]
            dest = args.output / 'chunks' / f'{start:09d}.npz'
            if dest.exists():
                with np.load(dest, allow_pickle=False) as saved:
                    validate(saved, subset, start)
                continue
            data = arrays(subset, start)
            for offset in range(0, len(subset), args.batch_size):
                batch_stems = subset[offset:offset + args.batch_size]
                images = list(pool.map(read_image, batch_stems))
                results = detector(images, conf=0.25, iou=0.7, imgsz=512,
                                   rect=True, device=args.rank, verbose=False)
                assert len(results) == len(images)
                pose_items, pose_indices = [], []
                for j, (image, result) in enumerate(zip(images, results)):
                    k = offset + j
                    h, w = image.shape[:2]
                    data['image_size_wh'][k] = [w, h]
                    box, info = choose_detector_bbox(result)
                    if box is None:
                        data['crop_bbox_xyxy'][k] = [0, 0, w - 1, h - 1]
                        continue
                    crop = expanded_square_bbox(box, expand=1.5)
                    data['detector_bbox_xyxy'][k] = box
                    data['crop_bbox_xyxy'][k] = crop
                    data['detector_hit'][k] = True
                    data['detector_score'][k] = info['score']
                    data['detector_class'][k] = info['class_id']
                    pose_items.append(({'crop_bbox_xyxy': crop}, image))
                    pose_indices.append(k)
                if pose_items:
                    with torch.inference_mode():
                        preds = _rtmpose_batch(pose, pipeline, pseudo_collate, pose_items)
                    assert len(preds) == len(pose_indices)
                    for k, pred in zip(pose_indices, preds):
                        xy = np.asarray(pred.pred_instances.keypoints[0], np.float32)
                        score = np.asarray(pred.pred_instances.keypoint_scores[0], np.float32)
                        if xy.shape != (21, 2) or score.shape != (21,) or not (
                                np.isfinite(xy).all() and np.isfinite(score).all()):
                            raise ValueError(f'Invalid RTMPose output: {subset[k]}')
                        data['keypoints_xy'][k] = xy
                        data['keypoint_scores'][k] = score
                        data['pose_returned'][k] = True
            validate(data, subset, start)
            atomic_npz(dest, data)
            processed += len(subset)
            elapsed = time.monotonic() - started
            progress = dict(rank=args.rank, newly_processed=processed, last_chunk=start,
                            samples_per_second=round(processed / elapsed, 2),
                            last_update=time.time())
            atomic_json(args.output / f'progress_{args.rank}.json', progress)
            print(json.dumps(progress), flush=True)
    print(f'worker {args.rank} complete', flush=True)


def merge(args, stems):
    all_data = arrays(stems, 0)
    for start in range(0, len(stems), args.chunk_size):
        subset = stems[start:start + args.chunk_size]
        with np.load(args.output / 'chunks' / f'{start:09d}.npz', allow_pickle=False) as d:
            validate(d, subset, start)
            for key in all_data:
                all_data[key][start:start + len(subset)] = d[key]
    validate(all_data, stems, 0)
    dest = args.output / 'conditions.npz'
    atomic_npz(dest, all_data)
    with np.load(dest, allow_pickle=False) as saved:
        validate(saved, stems, 0)
    hit = all_data['detector_hit']
    score = all_data['keypoint_scores']
    summary = dict(status='complete', n_samples=len(stems),
                   detector_hits=int(hit.sum()), detector_misses=int((~hit).sum()),
                   detector_hit_rate=float(hit.mean()),
                   at_least_two_joints_score_ge_0_1=int(((score >= 0.1).sum(1) >= 2).sum()),
                   pose_returned=int(all_data['pose_returned'].sum()),
                   cache=str(dest), cache_sha256=sha256(dest),
                   cache_bytes=dest.stat().st_size, completed_at=time.time())
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument('--samples-file', type=Path, default=Path(
        '/root/code/vepfs/dataset/hand_recon_hug/splits_v2/dexycb_train.clean.txt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--chunk-size', type=int, default=1024)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--merge-only', action='store_true')
    args = p.parse_args()
    stems = read_stems(args.samples_file)
    if args.limit:
        stems = stems[:args.limit]
    assert len(stems) == len(set(stems)) and stems
    assert 0 <= args.rank < args.workers and args.batch_size > 0 and args.chunk_size > 0
    assert all(not Path(s).is_absolute() and '..' not in Path(s).parts for s in stems)
    if args.launch or args.merge_only:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / 'launch.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / 'chunks').mkdir(exist_ok=True)
        meta = metadata(args, stems)
        meta_path = args.output / 'metadata.json'
        if meta_path.exists():
            assert json.loads(meta_path.read_text()) == meta, 'Cache metadata mismatch'
        else:
            atomic_json(meta_path, meta)
        if args.launch:
            children = []
            for rank in range(args.workers):
                cmd = [sys.executable, '-u', '-m', 'src.cache_train_conditions',
                       '--dataset-root', str(args.dataset_root), '--samples-file', str(args.samples_file),
                       '--output', str(args.output), '--batch-size', str(args.batch_size),
                       '--chunk-size', str(args.chunk_size), '--workers', str(args.workers),
                       '--rank', str(rank)]
                if args.limit:
                    cmd += ['--limit', str(args.limit)]
                log = (args.output / f'worker_{rank}.log').open('a')
                children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
                log.close()
            codes = [child.wait() for child in children]
            if any(codes):
                raise RuntimeError(f'Worker failures {codes}; rerun identical command to resume')
        merge(args, stems)
    else:
        worker(args, stems)


if __name__ == '__main__':
    main()
