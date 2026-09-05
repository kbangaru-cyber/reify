"""Report what is actually on disk before training.

    python scripts/audit_data.py

Answers three questions the split lists cannot: how many scenes were really
downloaded, how many are complete, and how many points they hold.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reify.config import load_config  # noqa: E402
from reify.data.scannet import AGG_SUFFIX, MESH_SUFFIX, SEGS_SUFFIX  # noqa: E402


def ply_vertex_count(path: str) -> int:
    """Read the vertex count from the header without parsing the body."""
    with open(path, "rb") as handle:
        for line in handle:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    return 0


def main() -> None:
    cfg = load_config()
    root = Path(cfg.data.root)
    scans, meta = root / "scans", root / "metadata"

    print(f"root: {root}")
    if not scans.is_dir():
        raise SystemExit(f"No scans/ directory under {root}. Check data.root.")

    on_disk = {d.name for d in scans.iterdir() if d.is_dir()}

    for split in ("train", "val"):
        list_path = meta / f"scannetv2_{split}.txt"
        if not list_path.exists():
            print(f"{split:5s}: no split list at {list_path}")
            continue
        listed = [ln.strip() for ln in open(list_path) if ln.strip()]
        present = [s for s in listed if s in on_disk]
        complete = [
            s for s in present
            if all((scans / s / (s + suf)).exists()
                   for suf in (MESH_SUFFIX, SEGS_SUFFIX, AGG_SUFFIX))
        ]
        print(f"{split:5s}: listed {len(listed):5d} | on disk {len(present):5d} "
              f"| complete {len(complete):5d}")
        for name in present[:1]:
            missing = [suf for suf in (MESH_SUFFIX, SEGS_SUFFIX, AGG_SUFFIX)
                       if not (scans / name / (name + suf)).exists()]
            if missing:
                print(f"       example missing files in {name}: {missing}")

    meshes = glob.glob(str(scans / "*" / f"*{MESH_SUFFIX}"))
    if meshes:
        counts = np.array([ply_vertex_count(p) for p in meshes])
        counts = counts[counts > 0]
        print(f"\nmeshes        : {len(counts)}")
        print(f"total points  : {counts.sum():,}")
        print(f"points/scene  : min {counts.min():,} "
              f"median {int(np.median(counts)):,} max {counts.max():,}")
        cap = int(cfg.data.max_points)
        if cap:
            print(f"over max_points={cap:,}: {(counts > cap).sum()} scenes "
                  "will be subsampled during training")

    total = sum(f.stat().st_size for f in scans.rglob("*") if f.is_file())
    print(f"scans on disk : {total / 1e9:.1f} GB")

    cache = Path(cfg.data.cache_dir)
    if cache.is_dir():
        files = list(cache.glob("*.npz"))
        size = sum(f.stat().st_size for f in files)
        print(f"cache         : {len(files)} scenes, {size / 1e9:.2f} GB at {cache}")
    else:
        print(f"cache         : empty ({cache})")


if __name__ == "__main__":
    main()
