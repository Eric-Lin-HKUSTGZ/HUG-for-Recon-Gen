"""Compute norm_stats (per-group mean/std of the 99D MANO state), per dataset.

Two-level output so adding a new dataset later does NOT require recomputing
existing ones:

  1. Per-dataset stats: one JSON per data dir (cached in --stats-dir),
     storing {n, mean, std} per group. Reused on later runs unless --recompute.
  2. Merged stats: exact parallel merge of the per-dataset JSONs into the
     layout assets/norm_stats.json uses ({translation, wrist_rot, finger_rot:
     {mean, std}}), consumable by training/inference directly.

Merge math (exact, population variance):
    mean = Σ nᵢ·meanᵢ / Σ nᵢ
    var  = Σ nᵢ·(varᵢ + (meanᵢ − mean)²) / Σ nᵢ

Usage:
    python scripts/compute_norm_stats.py \
        --data-dirs /root/code/vepfs/dataset/hand_recon_hug/ho3d \
                    /root/code/vepfs/dataset/hand_recon_hug/dexycb \
        --out assets/norm_stats_handrecon.json \
        --stats-dir assets/norm_stats_parts \
        --split-dir /root/code/vepfs/dataset/hand_recon_hug/splits \
        [--max-samples-per-set 100000] [--workers 16] [--recompute]

Per-dataset train-split allowlists live OUTSIDE the (huge) pkl directories.
By default the script looks for `<split-dir>/<dataset-dir-name>.txt`. For a
versioned dataset directory whose basename differs from the split prefix, pass
aligned `--split-files` explicitly (one file per `--data-dirs` entry).

Adding a dataset later: rerun with the new dir appended to --data-dirs and its
aligned split file; cached per-dataset stats are reused, only the new one is scanned.
"""

import glob
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import tyro
from rich.console import Console

console = Console()

GROUPS_99 = {"translation": (0, 3), "wrist_rot": (3, 9), "finger_rot": (9, 99)}
GROUPS_109 = {**GROUPS_99, "shape": (99, 109)}


def read_params(pkl_path: str, include_shape: bool = False):
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        g = d.get("grasp")
        if g is None:
            return None
        parts = [g["t"].flatten(), g["R_6d"].flatten(), g["pose_6d"].flatten()]
        if include_shape:
            # Converted samples contain canonical "shape" plus the original
            # subject-specific "shape_gt". Learnable 109D training must use
            # the subject shape when available.
            shape = np.asarray(g.get("shape_gt", g.get("shape")), dtype=np.float64).flatten()
            if shape.size != 10:
                return None
            parts.append(shape)
        x = np.concatenate(parts).astype(np.float64)
        expected = 109 if include_shape else 99
        if x.shape[0] != expected or not np.isfinite(x).all():
            return None
        return x
    except Exception:
        return None


def stats_of_dir(
    data_dir: Path,
    max_samples,
    workers,
    split_dir: Path | None,
    split_file: Path | None = None,
    include_shape: bool = False,
) -> dict:
    """Streaming mean/var (population) over one dataset dir.

    If `<split_dir>/<data_dir.name>.txt` exists, only those stems are used
    (e.g. an official train split), so test frames never enter the stats.
    Split files live outside the pkl dir to keep huge dirs clean.
    """
    files = sorted(glob.glob(str(data_dir / "**" / "*.pkl"), recursive=True))
    include_file = split_file or (
        split_dir / f"{data_dir.name}.txt" if split_dir else None
    )
    if include_file is not None and include_file.exists():
        allow = set(include_file.read_text().splitlines())
        files = [f for f in files if Path(f).stem in allow]
        console.print(f"  {data_dir.name}: 按 {include_file} 过滤 -> {len(files)} pkls")
    if max_samples is not None and len(files) > max_samples:
        rng = np.random.default_rng(42)
        idx = sorted(
            rng.choice(len(files), size=max_samples, replace=False).tolist()
        )
        files = [files[i] for i in idx]
    n = 0
    dim = 109 if include_shape else 99
    mean = np.zeros(dim, dtype=np.float64)
    m2 = np.zeros(dim, dtype=np.float64)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        reader = partial(read_params, include_shape=include_shape)
        for x in ex.map(reader, files, chunksize=64):
            done += 1
            if done % 20000 == 0:
                console.print(f"  {data_dir.name}: {done}/{len(files)}")
            if x is None:
                continue
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
    var = m2 / max(n, 1)
    # Keep degenerate dimensions finite and trainable. The converted datasets
    # currently use one canonical MANO shape, so shape std can be exactly zero.
    std = np.sqrt(var)
    std = np.where(std > 1e-6, std, 1.0)
    return {
        "n": n,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def cache_name(
    data_dir: Path,
    split_dir: Path | None,
    split_file: Path | None = None,
    include_shape: bool = False,
) -> str:
    key = str(data_dir.resolve())
    inc = split_file or (split_dir / f"{data_dir.name}.txt" if split_dir else None)
    if inc is not None and inc.exists():  # 过滤文件内容变化时缓存自动失效
        st = inc.stat()
        key += f"|include:{st.st_size}:{st.st_mtime}"
    key += f"|shape:{int(include_shape)}"
    return hashlib.md5(key.encode()).hexdigest()[:10] + "_" + data_dir.name + ".json"


def merge(parts: list[dict], include_shape: bool = False) -> dict:
    """Exact parallel merge of per-dataset {n, mean, std} into HUG layout."""
    ns = np.array([p["n"] for p in parts], dtype=np.float64)
    means = np.array([p["mean"] for p in parts], dtype=np.float64)
    vars_ = np.array([p["std"] for p in parts], dtype=np.float64) ** 2
    n_tot = ns.sum()
    mean = (ns[:, None] * means).sum(0) / n_tot
    var = (ns[:, None] * (vars_ + (means - mean) ** 2)).sum(0) / n_tot
    std = np.sqrt(var)
    std = np.where(std > 1e-6, std, 1.0)
    groups = GROUPS_109 if include_shape else GROUPS_99
    return {
        g: {"mean": mean[a:b].tolist(), "std": std[a:b].tolist()}
        for g, (a, b) in groups.items()
    }


def main(
    data_dirs: list[Path],
    out: Path,
    stats_dir: Path,
    split_dir: Path | None = None,
    split_files: list[Path] | None = None,
    max_samples_per_set: int | None = None,
    workers: int = 16,
    recompute: bool = False,
    include_shape: bool = False,
) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    if split_files is not None and len(split_files) != len(data_dirs):
        raise ValueError("--split-files must have exactly one file per --data-dirs entry")
    parts = []
    for i, d in enumerate(data_dirs):
        split_file = split_files[i] if split_files is not None else None
        cache = stats_dir / cache_name(d, split_dir, split_file, include_shape)
        if cache.exists() and not recompute:
            console.print(f"[cyan]{d}: 用缓存 {cache.name}[/cyan]")
            parts.append(json.load(open(cache)))
            continue
        console.print(f"[cyan]{d}: 扫描计算中...[/cyan]")
        st = stats_of_dir(
            d,
            max_samples_per_set,
            workers,
            split_dir,
            split_file,
            include_shape=include_shape,
        )
        json.dump(st, open(cache, "w"), indent=2)
        console.print(f"[green]{d}: n={st['n']}  缓存 -> {cache.name}[/green]")
        parts.append(st)

    merged = merge(parts, include_shape=include_shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(merged, open(out, "w"), indent=2)
    n_tot = sum(p["n"] for p in parts)
    console.print(f"[bold green]合并 {len(parts)} 个数据集, n={n_tot} -> {out}[/bold green]")
    console.print(
        f"translation mean={np.round(merged['translation']['mean'],4)} "
        f"std={np.round(merged['translation']['std'],4)}"
    )


if __name__ == "__main__":
    tyro.cli(main)
