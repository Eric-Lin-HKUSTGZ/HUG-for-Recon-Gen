"""Generate an offline recording-level val split for 1M-HUGS training.

Why recording-level: each physical grasp yields hundreds of (frame, grasp) pairs
(including a same-timestep `_grayscale` twin), so a random frame-level split
leaks — frames of one grasp would land on both sides. Grouping by recording
(the stem minus `<frame>_<hash>[_grayscale]`) keeps every frame of a grasp on
the same side.

Usage:
    python scripts/make_val_split.py \
        --dataset-path /root/code/vepfs/dataset/1m-hugs/grasp_data \
        --n-recordings 48

Writes `{dataset_path}/split_val.txt` (one stem per line), consumable by
GraspDataset via `samples_filename="split_val.txt"`.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro
from rich.console import Console

console = Console()


def recording_key(stem: str) -> str:
    """'8_ball_1_00000021_2df00295[_grayscale]' -> '8_ball_1'."""
    s = stem.removesuffix("_grayscale")
    return s.rsplit("_", 2)[0]


def main(
    dataset_path: Path,
    n_recordings: int = 48,
    seed: int = 42,
    out_name: str = "split_val.txt",
) -> None:
    samples_file = dataset_path / "samples.txt"
    if not samples_file.exists():
        raise FileNotFoundError(f"{samples_file} not found; build the index first")
    stems = [s for s in samples_file.read_text().splitlines() if s]

    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[recording_key(stem)].append(stem)

    recordings = sorted(groups)
    rng = np.random.default_rng(seed)
    held_out = sorted(
        rng.choice(len(recordings), size=min(n_recordings, len(recordings)), replace=False)
        .tolist()
    )
    val_stems = sorted(s for i in held_out for s in groups[recordings[i]])

    out_path = dataset_path / out_name
    out_path.write_text("\n".join(val_stems) + "\n")

    n_gray = sum(s.endswith("_grayscale") for s in val_stems)
    console.print(
        f"[green]held out {len(held_out)}/{len(recordings)} recordings -> "
        f"{len(val_stems)} frames ({n_gray} grayscale twins) "
        f"= {100.0 * len(val_stems) / len(stems):.2f}% of {len(stems)} frames[/green]"
    )
    console.print(f"[cyan]wrote {out_path}[/cyan]")
    console.print(f"held-out recordings: {[recordings[i] for i in held_out][:8]} ...")


if __name__ == "__main__":
    tyro.cli(main)
