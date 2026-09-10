"""Generate train/val/test stem lists for the converted hand-recon datasets.

- HO3D: recording-level holdout (whole sequences) from the converted train pkls.
- DexYCB: official s0 manifests (s0_train / s0_val / s0_test).
- HO3D_v3 evaluation split: all converted ho3d_eval pkls (inference schema,
  no MANO GT) -> splits/ho3d_eval.txt.
  Frames without hand annotations were skipped at conversion, so lists are
  intersected with the pkls that actually exist.

Outputs land in `<split_dir>` (NOT inside the huge pkl dirs):
    ho3d_train.txt / ho3d_val.txt / ho3d_eval.txt
    dexycb_train.txt / dexycb_val.txt / dexycb_test.txt

GraspDataset accepts these via `samples_filename` (absolute paths work).

Usage:
    python scripts/make_handrecon_splits.py \
        --root /root/code/vepfs/dataset/hand_recon_hug \
        --dexycb-dir /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
        --split-dir /root/code/vepfs/dataset/hand_recon_hug/splits \
        --ho3d-val-sequences 5
"""

import json
from pathlib import Path

import numpy as np
import tyro
from rich.console import Console

console = Console()

DEX_MANIFESTS = Path("/root/code/vepfs/dataset/dex-ycb/manifests")


def dex_color_to_stem(color_file: str) -> str:
    # "20200709-subject-01/20200709_141754/836212060125/color_000000.jpg"
    subj, seq, serial, fname = color_file.split("/")
    frame = fname.replace("color_", "").replace(".jpg", "")
    return f"dexycb_{subj}_{seq}_{serial}_{frame}"


def write_list(path: Path, stems: list[str]) -> None:
    path.write_text("\n".join(sorted(stems)) + "\n")
    console.print(f"[green]wrote {path} ({len(stems)} stems)[/green]")


def main(
    root: Path,
    dexycb_dir: Path | None = None,
    ho3d_dir: Path | None = None,
    ho3d_eval_dir: Path | None = None,
    split_dir: Path | None = None,
    ho3d_val_sequences: int = 5,
    seed: int = 42,
) -> None:
    # Explicit dataset paths avoid accidentally building lists against an old
    # conversion when a versioned canonical directory is present.
    splits = split_dir or root / "splits"
    splits.mkdir(parents=True, exist_ok=True)

    # ---- HO3D: sequence-level holdout ----
    ho3d_dir = ho3d_dir or root / "ho3d"
    stems = sorted(p.stem for p in ho3d_dir.glob("*.pkl"))
    by_seq: dict[str, list[str]] = {}
    for s in stems:  # stem = ho3d_<SEQ>_<frame>
        by_seq.setdefault(s.rsplit("_", 1)[0], []).append(s)
    seqs = sorted(by_seq)
    rng = np.random.default_rng(seed)
    val_seqs = sorted(
        rng.choice(len(seqs), size=min(ho3d_val_sequences, len(seqs)), replace=False).tolist()
    )
    val_seq_names = {seqs[i] for i in val_seqs}
    ho3d_val = [s for q in val_seq_names for s in by_seq[q]]
    ho3d_train = [s for q, ss in by_seq.items() if q not in val_seq_names for s in ss]
    write_list(splits / "ho3d_train.txt", ho3d_train)
    write_list(splits / "ho3d_val.txt", ho3d_val)
    console.print(f"[cyan]HO3D held-out sequences: {sorted(val_seq_names)}[/cyan]")

    # ---- DexYCB: official s0 manifests, intersect with converted pkls ----
    dex_dir = dexycb_dir or root / "dexycb"
    for split in ("train", "val", "test"):
        want = []
        with open(DEX_MANIFESTS / f"s0_{split}.jsonl") as f:
            for line in f:
                want.append(dex_color_to_stem(json.loads(line)["color_file"]))
        existing = [s for s in want if (dex_dir / f"{s}.pkl").exists()]
        console.print(
            f"[cyan]dexycb s0_{split}: manifest {len(want)}, "
            f"converted in {dex_dir}: {len(existing)}[/cyan]"
        )
        write_list(splits / f"dexycb_{split}.txt", existing)

    # ---- HO3D_v3 official evaluation split ----
    eval_dir = ho3d_eval_dir or root / "ho3d_eval"
    eval_stems = sorted(p.stem for p in eval_dir.glob("*.pkl"))
    write_list(splits / "ho3d_eval.txt", eval_stems)


if __name__ == "__main__":
    tyro.cli(main)
