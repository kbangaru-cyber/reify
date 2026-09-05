# Running on Colab

Verified target: Python 3.12, `torch 2.9.0+cu126`, one NVIDIA L4.

## 1. Runtime

Set the runtime to a GPU before anything else: **Runtime, Change runtime type, T4 or L4**.

```python
import torch, sys
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu   :", torch.cuda.get_device_name(0))
```

## 2. Mount Drive and clone

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/kbangaru-cyber/reify.git /content/reify
%cd /content/reify
!pip -q install -e .
```

`pip install -e .` puts `reify` on the path. Colab already has a working torch,
so nothing here reinstalls it.

To pull later changes without re-cloning:

```bash
%cd /content/reify && git pull
```

## 3. Point it at your data

Every path lives in `configs/default.yaml`. Override without editing the file:

```python
import os
os.environ["REIFY_DATA__ROOT"] = "/content/drive/MyDrive/Thesis/Datasets/Point_cloud_segmentation_data/scannetv2_raw"
os.environ["REIFY_DATA__CACHE_DIR"] = "/content/reify_cache"
os.environ["REIFY_TRAIN__OUT_DIR"] = "/content/drive/MyDrive/Thesis/Playground/Point_cloud_segmentation/reify_ckpts"
```

The root is the directory that contains `scans/` and `metadata/`.

## 4. Check the data before training

```bash
!python scripts/audit_data.py
```

This reports what is listed, what is on disk, what is complete, and the point
totals. Do this first: the dataset filters to complete scenes, so a partial
download shows up here as a smaller usable count rather than as a crash later.

## 5. Train

```bash
!python scripts/train.py
```

The first epoch is slow because every scene is parsed from the PLY and cached.
Subsequent epochs read the cache. The cache lives on `/content`, which is wiped
when the session ends, so budget for one slow pass per session or move
`REIFY_DATA__CACHE_DIR` onto Drive and accept slower small-file reads.

Useful overrides:

```python
os.environ["REIFY_TRAIN__STEPS"] = "60000"       # ~100 epochs at batch 2
os.environ["REIFY_TRAIN__BATCH_SIZE"] = "4"
os.environ["REIFY_DATA__NUM_WORKERS"] = "4"
os.environ["REIFY_MODEL__BACKBONE"] = "spconv"   # hard error if not installed
```

Resume:

```bash
!python scripts/train.py --resume "$REIFY_TRAIN__OUT_DIR/ckpt_step5000.pt"
```

### On training length

At `batch_size 2`, 6000 steps is about 10 epochs over 1201 scenes. Published
ScanNet results come from several hundred epochs. Treat anything under about
50000 steps as a smoke test, not a result.

## 6. Evaluate

```bash
!python scripts/evaluate.py --ckpt "$REIFY_TRAIN__OUT_DIR/ckpt_step6000.pt" --limit 20
```

Drop `--limit` for the full 312 scene validation split. This writes
`results.md` with semantic mIoU, a per-class IoU table, and panoptic PQ, SQ
and RQ, along with the backbone and scene count that produced them.

Evaluation disables the `max_points` and `max_superpoints` caps and scores at
the level of the original points. That is slower and uses more memory than
training, and it is what makes the number comparable with published mIoU.

### Evaluating the old checkpoint

A checkpoint records its own feature width. `ckpt_step6000.pt` from the original
notebook was trained on 6-dim xyz+rgb, so evaluation will read
`use_normals: false` from it automatically. Do not override that: a 9-dim model
cannot load 6-dim weights.

If the old checkpoint predates this repo and has no metadata, tell it explicitly:

```python
os.environ["REIFY_DATA__USE_NORMALS"] = "False"
```

## 7. Keeping outputs out of git

`.gitignore` already covers `*.pt`, `*.npy`, `*.npz` and cache directories.
Nothing in this repo should ever contain a scene or a checkpoint.

## Troubleshooting

**"None of the N scenes listed are complete"** — `data.root` is wrong, or points
at `scans/` instead of its parent. It must contain both `scans/` and `metadata/`.

**Backbone prints `mlp`** — no sparse convolution library is installed, so every
voxel is encoded independently with no neighbourhood aggregation. That is a
legitimate baseline but it is not the intended model. Set
`REIFY_MODEL__BACKBONE=spconv` to turn the silent fallback into a hard error.

**Out of memory during evaluation** — expected on large scenes with the caps
off. Lower `eval.limit` to score fewer scenes, and say in `results.md` how many
were used. A number over 20 scenes that is honestly labelled beats a number over
312 scenes that never finished.

**First epoch is extremely slow** — that is the cache filling. Watch
`/content/reify_cache` grow. Raising `REIFY_DATA__NUM_WORKERS` helps most here.
