"""Report what is actually on disk before training.

    python scripts/audit_data.py            # split counts only, fast
    python scripts/audit_data.py --points   # also read every PLY header
    python scripts/audit_data.py --size     # also measure bytes on disk

The last two touch thousands of files. On local disk that is instant. On a
Google Drive mount every file is a network round trip, so they are opt-in and
run on a thread pool.
"""

from __future__ import annotations

import argparse
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", action="store_true",
                        help="read every PLY header for point totals (slow on Drive)")
    parser.add_argument("--size", action="store_true",
                        help="measure bytes on disk (slow on Drive)")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

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
    print(f"\nmeshes        : {len(meshes)}")

    if args.points and meshes:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            counts = list(tqdm(pool.map(ply_vertex_count, meshes),
                               total=len(meshes), desc="ply headers"))
        counts = np.array([c for c in counts if c > 0])
        print(f"total points  : {counts.sum():,}")
        print(f"points/scene  : min {counts.min():,} "
              f"median {int(np.median(counts)):,} max {counts.max():,}")
        cap = int(cfg.data.max_points)
        if cap:
            print(f"over max_points={cap:,}: {(counts > cap).sum()} scenes "
                  "will be subsampled during training")
    elif meshes:
        print("total points  : not measured (pass --points)")

    if args.size:
        files = [f for f in scans.rglob("*") if f.is_file()]
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            sizes = list(tqdm(pool.map(lambda f: f.stat().st_size, files),
                              total=len(files), desc="sizes"))
        print(f"scans on disk : {sum(sizes) / 1e9:.1f} GB")
    else:
        print("scans on disk : not measured (pass --size)")

    cache = Path(cfg.data.cache_dir)
    if cache.is_dir():
        files = list(cache.glob("*.npz"))
        size = sum(f.stat().st_size for f in files)
        print(f"cache         : {len(files)} scenes, {size / 1e9:.2f} GB at {cache}")
    else:
        print(f"cache         : empty ({cache})")


if __name__ == "__main__":
    main()
