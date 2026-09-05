"""ScanNet v2 loading, voxelization and batching.

Differences from the original notebook implementation, all behaviour preserving
except where noted:

  * `_mode_per_group` finishes with a vectorised segment argmax instead of a
    Python loop over runs.
  * Superpoint filtering uses `np.isin` instead of a per-point Python loop.
  * Voxel to superpoint remapping uses `np.searchsorted` instead of a per-voxel
    Python loop. `np.unique` already returns sorted values, so this is exact.
  * Scene ids are filtered against what is actually on disk, so a partial
    download fails at construction with a count rather than mid epoch.
  * Optionally reads the mesh faces to derive per-vertex normals, which take the
    feature width from 6 to 9. This changes the model input, so a checkpoint
    trained one way cannot be loaded the other way.
  * Setting max_points and max_superpoints to 0 disables the caps, which is
    required for evaluation to be comparable with published numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from reify.data.labels import NYU40_TO_SCANNET20

try:  # only needed to parse the label map
    import json
except ImportError:  # pragma: no cover
    raise

MESH_SUFFIX = "_vh_clean_2.ply"
SEGS_SUFFIX = "_vh_clean_2.0.010000.segs.json"
AGG_SUFFIX = ".aggregation.json"


# --------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------

def load_raw_to_nyu40(meta_dir: str) -> dict[str, int]:
    tsv = os.path.join(meta_dir, "scannetv2-labels.combined.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError(
            f"Label map not found at {tsv}. It ships with ScanNet as "
            "scannetv2-labels.combined.tsv"
        )
    df = pd.read_csv(tsv, sep="\t")
    mapping: dict[str, int] = {}
    for col in ("raw_category", "category"):
        if col in df.columns and "nyu40id" in df.columns:
            for raw, nyu in zip(df[col], df["nyu40id"]):
                if isinstance(raw, str):
                    mapping[raw.strip().lower()] = int(nyu)
    if not mapping:
        raise ValueError(f"Could not build a label map from {tsv}")
    return mapping


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def _mode_per_group(inv: np.ndarray, labels: np.ndarray, m: int,
                    ignore_val: int | None = -1) -> np.ndarray:
    """Majority label per group.

    Groups with no valid label get -1. Ties are broken deterministically but
    arbitrarily, which matches the original behaviour.
    """
    labels = labels.astype(np.int64, copy=False)
    inv = inv.astype(np.int64, copy=False)

    if ignore_val is not None:
        keep = labels != ignore_val
        inv_k, lab_k = inv[keep], labels[keep]
    else:
        inv_k, lab_k = inv, labels

    out = np.full((m,), -1, dtype=np.int32)
    if inv_k.size == 0:
        return out

    order = np.lexsort((lab_k, inv_k))
    inv_s, lab_s = inv_k[order], lab_k[order]

    # run-length encode the (group, label) pairs
    same = (inv_s[1:] == inv_s[:-1]) & (lab_s[1:] == lab_s[:-1])
    starts = np.concatenate(([0], np.flatnonzero(~same) + 1))
    ends = np.concatenate((starts[1:], [inv_s.size]))
    run_groups = inv_s[starts]
    run_labels = lab_s[starts]
    run_counts = (ends - starts)

    # vectorised segment argmax: sort by (group, count) and keep the last entry
    # of each group, which is the group's most frequent label.
    order2 = np.lexsort((run_counts, run_groups))
    g = run_groups[order2]
    last = np.concatenate((np.flatnonzero(g[1:] != g[:-1]), [g.size - 1]))
    out[g[last]] = run_labels[order2][last].astype(np.int32)
    return out


def read_mesh(ply_path: str, want_normals: bool):
    """Return xyz, rgb and optionally per-vertex normals from a ScanNet mesh.

    Normals are accumulated from the face cross products, so they are area
    weighted for free. The mesh faces are present in `_vh_clean_2.ply` and were
    simply unused before.
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0

    if not want_normals:
        return xyz, rgb, None

    if "face" not in ply:
        # Not a mesh. Fall back to zero normals rather than failing the scene.
        return xyz, rgb, np.zeros_like(xyz)

    faces = np.vstack([np.asarray(f[0], dtype=np.int64) for f in ply["face"].data])
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # magnitude is twice the triangle area

    normals = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, np.maximum(norm, 1e-8), out=normals)
    return xyz, rgb, normals.astype(np.float32)


def read_raw_labels(scene_dir: str, scene_id: str, raw_to_nyu40: dict[str, int]):
    """Per-point superpoint id, ScanNet20 semantic id and instance id."""
    with open(os.path.join(scene_dir, scene_id + SEGS_SUFFIX), "r") as handle:
        seg_ids = np.asarray(json.load(handle)["segIndices"], dtype=np.int64)

    with open(os.path.join(scene_dir, scene_id + AGG_SUFFIX), "r") as handle:
        groups = json.load(handle).get("segGroups", [])

    max_seg = int(seg_ids.max()) if seg_ids.size else 0
    seg2inst = np.full((max_seg + 1,), -1, dtype=np.int32)
    seg2sem = np.full((max_seg + 1,), -1, dtype=np.int32)

    for inst_id, group in enumerate(groups):
        raw = str(group.get("label", "")).strip().lower()
        nyu40 = raw_to_nyu40.get(raw, 0)
        sc20 = NYU40_TO_SCANNET20.get(int(nyu40), -1)
        segs = np.asarray(group.get("segments", []), dtype=np.int64)
        segs = segs[(segs >= 0) & (segs <= max_seg)]
        seg2inst[segs] = inst_id
        seg2sem[segs] = sc20

    return seg_ids, seg2sem[seg_ids], seg2inst[seg_ids]


def voxelize(xyz, feats_extra, seg_ids, voxel_size):
    """Mean-pool point features into a voxel grid.

    feats_extra is everything appended after xyz, so rgb, or rgb plus normals.
    """
    grid = np.floor(xyz / voxel_size).astype(np.int32)
    vox_u, inv = np.unique(grid, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    v = vox_u.shape[0]

    count = np.maximum(np.bincount(inv, minlength=v), 1).astype(np.float32)[:, None]
    stacked = np.concatenate([xyz, feats_extra], axis=1)

    summed = np.stack(
        [np.bincount(inv, weights=stacked[:, c], minlength=v) for c in range(stacked.shape[1])],
        axis=1,
    ).astype(np.float32)

    vox_feats = summed / count
    vox_seg = _mode_per_group(inv, seg_ids, v, ignore_val=None).astype(np.int64)
    return vox_u.astype(np.int32), vox_feats, vox_seg


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

class ScanNetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        voxel_size: float = 0.02,
        max_points: int = 200_000,
        max_superpoints: int = 4096,
        use_normals: bool = True,
        cache_dir: str | None = None,
        seed: int = 0,
        return_points: bool = False,
    ):
        self.root = str(root)
        self.scans = os.path.join(self.root, "scans")
        self.meta = os.path.join(self.root, "metadata")
        self.split = split
        self.voxel_size = float(voxel_size)
        # 0 or None disables a cap
        self.max_points = int(max_points) or 0
        self.max_superpoints = int(max_superpoints) or 0
        self.use_normals = bool(use_normals)
        self.seed = int(seed)
        self.return_points = bool(return_points)

        list_path = os.path.join(self.meta, f"scannetv2_{split}.txt")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Split list not found: {list_path}")
        listed = [ln.strip() for ln in open(list_path) if ln.strip()]

        # Only keep scenes whose four required files are all present. A partial
        # download should fail here with a count, not at a random training step.
        self.scene_ids = [s for s in listed if self._complete(s)]
        missing = len(listed) - len(self.scene_ids)
        if not self.scene_ids:
            raise RuntimeError(
                f"None of the {len(listed)} scenes listed in {list_path} are complete "
                f"under {self.scans}. Check data.root."
            )
        print(
            f"[data] split={split} listed={len(listed)} usable={len(self.scene_ids)}"
            + (f" skipped={missing} (incomplete on disk)" if missing else "")
        )

        self.raw_to_nyu40 = load_raw_to_nyu40(self.meta)

        self.cache_dir = str(cache_dir) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def in_channels(self) -> int:
        return 9 if self.use_normals else 6

    def _complete(self, scene_id: str) -> bool:
        d = Path(self.scans) / scene_id
        return all(
            (d / (scene_id + suffix)).exists()
            for suffix in (MESH_SUFFIX, SEGS_SUFFIX, AGG_SUFFIX)
        )

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _cache_path(self, scene_id: str) -> str | None:
        if not self.cache_dir:
            return None
        tag = (
            f"vs{self.voxel_size:.3f}_mp{self.max_points}_msp{self.max_superpoints}"
            f"_n{int(self.use_normals)}_p{int(self.return_points)}"
        )
        return os.path.join(self.cache_dir, f"{scene_id}_{tag}.npz")

    def __getitem__(self, idx: int) -> dict:
        scene_id = self.scene_ids[idx]
        cache_path = self._cache_path(scene_id)

        if cache_path and os.path.exists(cache_path):
            try:
                return self._from_npz(scene_id, np.load(cache_path, allow_pickle=False))
            except Exception:
                os.remove(cache_path)  # truncated or stale, rebuild below

        item = self._build(scene_id)

        if cache_path:
            payload = {k: v.numpy() for k, v in item.items() if torch.is_tensor(v)}
            tmp = cache_path + ".tmp.npz"
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, cache_path)  # atomic, so a killed worker cannot leave a partial file

        return item

    def _from_npz(self, scene_id: str, z) -> dict:
        item = {"scene_id": scene_id}
        for key in z.files:
            item[key] = torch.from_numpy(z[key])
        return item

    def _build(self, scene_id: str) -> dict:
        scene_dir = os.path.join(self.scans, scene_id)
        rng = np.random.default_rng(self.seed + hash(scene_id) % (2**31))

        xyz, rgb, normals = read_mesh(
            os.path.join(scene_dir, scene_id + MESH_SUFFIX), self.use_normals
        )
        seg_ids, sem, inst = read_raw_labels(scene_dir, scene_id, self.raw_to_nyu40)

        n = min(len(xyz), len(seg_ids), len(sem), len(inst))
        xyz, rgb, seg_ids, sem, inst = xyz[:n], rgb[:n], seg_ids[:n], sem[:n], inst[:n]
        if normals is not None:
            normals = normals[:n]

        if self.max_points and n > self.max_points:
            sel = rng.choice(n, size=self.max_points, replace=False)
            sel.sort()  # keep points in their original order
            xyz, rgb, seg_ids, sem, inst = xyz[sel], rgb[sel], seg_ids[sel], sem[sel], inst[sel]
            if normals is not None:
                normals = normals[sel]

        seg_u, seg_inv, seg_cnt = np.unique(seg_ids, return_inverse=True, return_counts=True)
        seg_inv = seg_inv.reshape(-1)
        m = seg_u.shape[0]

        if self.max_superpoints and m > self.max_superpoints:
            top = np.argsort(-seg_cnt)[: self.max_superpoints]
            keep = np.isin(seg_ids, seg_u[top])  # vectorised, was a Python loop
            xyz, rgb, seg_ids, sem, inst = xyz[keep], rgb[keep], seg_ids[keep], sem[keep], inst[keep]
            if normals is not None:
                normals = normals[keep]
            seg_u, seg_inv, seg_cnt = np.unique(seg_ids, return_inverse=True, return_counts=True)
            seg_inv = seg_inv.reshape(-1)
            m = seg_u.shape[0]

        sp_sem = _mode_per_group(seg_inv, sem, m, ignore_val=-1)
        sp_inst = _mode_per_group(seg_inv, inst, m, ignore_val=-1)

        extra = rgb if normals is None else np.concatenate([rgb, normals], axis=1)
        coords, vox_feats, vox_seg_raw = voxelize(xyz, extra, seg_ids, self.voxel_size)

        # map raw segment id to contiguous superpoint index. seg_u is sorted, so
        # searchsorted is exact and replaces the per-voxel Python loop.
        pos = np.searchsorted(seg_u, vox_seg_raw)
        pos = np.clip(pos, 0, len(seg_u) - 1)
        vox_sp = np.where(seg_u[pos] == vox_seg_raw, pos, -1).astype(np.int64)

        keep_vox = vox_sp >= 0
        coords, vox_feats, vox_sp = coords[keep_vox], vox_feats[keep_vox], vox_sp[keep_vox]

        item = {
            "scene_id": scene_id,
            "coords": torch.from_numpy(coords.astype(np.int32)),
            "feats": torch.from_numpy(vox_feats.astype(np.float32)),
            "vox_sp": torch.from_numpy(vox_sp),
            "sp_sem": torch.from_numpy(sp_sem.astype(np.int64)),
            "sp_inst": torch.from_numpy(sp_inst.astype(np.int64)),
        }

        if self.return_points:
            # Everything needed to score at the level of the original points.
            item["point_sp"] = torch.from_numpy(seg_inv.astype(np.int64))
            item["point_sem"] = torch.from_numpy(sem.astype(np.int64))
            item["point_inst"] = torch.from_numpy(inst.astype(np.int64))

        return item


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def collate(batch: list[dict]) -> dict:
    """Concatenate scenes, with superpoint ids made global across the batch."""
    b = len(batch)
    m_list = [int(item["sp_sem"].shape[0]) for item in batch]
    offsets = torch.tensor([0] + list(np.cumsum(m_list)), dtype=torch.long)

    coords, feats, vox_sp_global = [], [], []
    for i, item in enumerate(batch):
        v = item["coords"].shape[0]
        idx = torch.full((v, 1), i, dtype=torch.int32)
        coords.append(torch.cat([idx, item["coords"].to(torch.int32)], dim=1))
        feats.append(item["feats"])
        vox_sp_global.append(item["vox_sp"] + int(offsets[i]))

    m_max = max(m_list)
    sp_sem = torch.full((b, m_max), -1, dtype=torch.long)
    sp_inst = torch.full((b, m_max), -1, dtype=torch.long)
    sp_mask = torch.zeros((b, m_max), dtype=torch.bool)
    for i, item in enumerate(batch):
        m = m_list[i]
        sp_sem[i, :m] = item["sp_sem"]
        sp_inst[i, :m] = item["sp_inst"]
        sp_mask[i, :m] = True

    out = {
        "scene_id": [item["scene_id"] for item in batch],
        "coords": torch.cat(coords, dim=0),
        "feats": torch.cat(feats, dim=0),
        "vox_sp_global": torch.cat(vox_sp_global, dim=0).to(torch.long),
        "sp_offsets": offsets,
        "sp_sem": sp_sem,
        "sp_inst": sp_inst,
        "sp_mask": sp_mask,
    }

    # Point-level arrays are ragged, so they stay as a list. Evaluation uses
    # batch size 1, so this costs nothing there.
    if "point_sp" in batch[0]:
        for key in ("point_sp", "point_sem", "point_inst"):
            out[key] = [item[key] for item in batch]

    return out
