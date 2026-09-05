"""Preprocess every scene into the cache, once.

    python scripts/build_cache.py --split train
    python scripts/build_cache.py --split train --limit 20   # time it first

This is pure CPU and I/O work: read the PLY, voxelize, write an npz. Running it
on an expensive GPU runtime wastes money, because the GPU is idle throughout.
Build the cache on a cheap runtime, point `data.cache_dir` at Drive so it
survives the session, then train on a fast GPU with the cache already warm.

Scenes already cached are skipped, so the script can be interrupted and resumed.
The cache key encodes voxel_size, the caps, normals and whether point level
arrays are included, so a cache built with one setting is never silently reused
with another.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from reify.config import load_config  # noqa: E402
from reify.data.scannet import ScanNetDataset  # noqa: E402


def _noop_collate(batch):
    # __getitem__ has already written the cache entry. Nothing needs to come back,
    # and returning the tensors would only cost memory and pickling time.
    return len(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=0, help="stop after N scenes")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the on-disk completeness check (slow on network mounts)")
    parser.add_argument("--eval-mode", action="store_true",
                        help="build the evaluation cache: caps disabled, point arrays kept")
    args = parser.parse_args()

    cfg = load_config(args.config)
    workers = cfg.data.num_workers if args.workers is None else args.workers

    dataset = ScanNetDataset(
        root=cfg.data.root,
        split=args.split,
        voxel_size=cfg.data.voxel_size,
        max_points=0 if args.eval_mode else cfg.data.max_points,
        max_superpoints=0 if args.eval_mode else cfg.data.max_superpoints,
        use_normals=cfg.data.use_normals,
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.seed,
        return_points=args.eval_mode,
        limit=args.limit,
        verify=not args.no_verify,
    )

    cache = Path(cfg.data.cache_dir)
    already = len(list(cache.glob("*.npz"))) if cache.is_dir() else 0
    print(f"[cache] dir={cache}")
    print(f"[cache] mode={'eval' if args.eval_mode else 'train'} "
          f"scenes={len(dataset)} workers={workers} already_present={already}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=_noop_collate,
    )

    start = time.time()
    for _ in tqdm(loader, total=len(dataset), desc=f"caching {args.split}"):
        pass
    elapsed = time.time() - start

    files = list(cache.glob("*.npz"))
    size = sum(f.stat().st_size for f in files)
    print(f"[cache] done in {elapsed / 60:.1f} min "
          f"({elapsed / max(len(dataset), 1):.2f} s/scene)")
    print(f"[cache] {len(files)} files, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
