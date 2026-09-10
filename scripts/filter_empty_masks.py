"""Filter out converted pkls whose hand is outside the 224 center-square crop,
producing clean train/val/test stem lists.

- HO3D train / DexYCB (all splits): criterion = hand mask non-empty (mask
  bytes are stored in the pkl). ~3% of converted frames are affected.
- HO3D_v3 evaluation split: the official eval set ships NO segmentation
  masks (object_mask is empty bytes), so the criterion is instead the stored
  condition_point (projected wrist) lying inside the 224x224 crop.

Empty-mask frames teach "reconstruct an invisible hand" - pure noise for the
reconstruction task.

Usage:
    python scripts/filter_empty_masks.py --root /root/code/vepfs/dataset/hand_recon_hug \
        [--workers 32] [--lists dexycb_test,ho3d_eval]

Reads splits/<name>.txt, writes splits/<name>.clean.txt. Default lists:
ho3d_train, ho3d_val, dexycb_train, dexycb_val, dexycb_test, ho3d_eval.
"""

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tyro
from rich.console import Console

console = Console()

DEFAULT_LISTS = (
    "ho3d_train", "ho3d_val",
    "dexycb_train", "dexycb_val",
    "dexycb_test", "ho3d_eval",
)


def mask_nonempty(pkl_path: str) -> bool:
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        m = cv2.imdecode(np.frombuffer(d["object_mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
        return m is not None and int((m > 0).sum()) > 0
    except Exception:
        return False


def wrist_in_crop(pkl_path: str) -> bool:
    """HO3D eval criterion: projected wrist (condition_point) inside the crop."""
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        u, v = float(d["condition_point"][0]), float(d["condition_point"][1])
        w = d["camera"]["width"]
        h = d["camera"]["height"]
        return 0.0 <= u < w and 0.0 <= v < h
    except Exception:
        return False


def filter_list(
    root: Path, list_path: Path, workers: int, dataset_dir: Path | None = None
) -> None:
    stems = [s for s in list_path.read_text().splitlines() if s]
    if dataset_dir is None:
        # ho3d_train.txt / dexycb_val.txt -> ho3d/ , dexycb/ ; ho3d_eval.txt -> ho3d_eval/
        first = list_path.name.split("_")[0]
        dataset_dir = root / ("ho3d_eval" if list_path.stem == "ho3d_eval" else first)
    paths = [str(dataset_dir / f"{s}.pkl") for s in stems]
    # eval pkls have no mask -> use the wrist-projection criterion instead
    keep_fn = wrist_in_crop if list_path.stem == "ho3d_eval" else mask_nonempty
    keep = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for stem, ok in zip(stems, ex.map(keep_fn, paths, chunksize=64)):
            if ok:
                keep.append(stem)
    out = list_path.with_suffix(".clean.txt")
    out.write_text("\n".join(keep) + "\n")
    console.print(
        f"[green]{list_path.name}: {len(stems)} -> {len(keep)} "
        f"(剔除 {len(stems) - len(keep)}, {100*(len(stems)-len(keep))/max(len(stems),1):.1f}%)[/green]"
    )


def main(
    root: Path,
    workers: int = 32,
    lists: str | None = None,
    dataset_path: Path | None = None,
    split_dir: Path | None = None,
) -> None:
    splits = split_dir or root / "splits"
    names = lists.split(",") if lists else DEFAULT_LISTS
    for name in names:
        ds_dir = dataset_path if name.startswith("dexycb") and dataset_path else None
        filter_list(root, splits / f"{name}.txt", workers, ds_dir)


if __name__ == "__main__":
    tyro.cli(main)
