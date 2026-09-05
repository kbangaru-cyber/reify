"""Train the unified segmentation model.

    python scripts/train.py
    python scripts/train.py --config configs/default.yaml --resume /path/ckpt.pt

Every path comes from the config, so a different machine needs one edit or a
few REIFY_ environment variables. The active backbone is printed at startup and
stored in each checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reify.config import load_config  # noqa: E402
from reify.data import ScanNetDataset, collate  # noqa: E402
from reify.models import OneFormer3DLike, compute_losses  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    dataset = ScanNetDataset(
        root=cfg.data.root,
        split="train",
        voxel_size=cfg.data.voxel_size,
        max_points=cfg.data.max_points,
        max_superpoints=cfg.data.max_superpoints,
        use_normals=cfg.data.use_normals,
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.seed,
        verify=cfg.data.verify,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
        persistent_workers=cfg.data.num_workers > 0,
        drop_last=True,
    )

    model = OneFormer3DLike(
        num_classes=cfg.model.num_classes,
        in_channels=dataset.in_channels,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_decoder_layers=cfg.model.num_decoder_layers,
        k_ins=cfg.model.k_ins,
        query_aug_std=cfg.model.query_aug_std,
        backbone=cfg.model.backbone,
        class_agnostic=cfg.model.class_agnostic,
        aux_semantic=cfg.model.aux_semantic,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = max(1, len(dataset) // cfg.train.batch_size)
    print(f"[train] mode={'class-agnostic' if model.class_agnostic else 'class-aware'}"
          f" aux_semantic={model.aux_semantic}")
    print(f"[train] parameters={params:,} scenes={len(dataset)} "
          f"steps/epoch={steps_per_epoch} target={cfg.train.steps} "
          f"({cfg.train.steps / steps_per_epoch:.1f} epochs)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp) and device == "cuda")

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = int(state.get("step", 0))
        print(f"[train] resumed from {args.resume} at step {start_step}")

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    keys = ["loss", "loss_cls", "loss_bce", "loss_dice", "loss_sem", "pos_queries"]
    meters = {k: deque(maxlen=cfg.train.log_every) for k in keys}

    model.train()
    iterator = iter(loader)
    t0 = time.time()

    for step in range(start_step + 1, cfg.train.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(batch)
            stats = compute_losses(batch, out, num_classes=cfg.model.num_classes,
                                   class_agnostic=cfg.model.class_agnostic)

        opt.zero_grad(set_to_none=True)
        scaler.scale(stats["loss"]).backward()
        if cfg.train.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()

        for k in keys:
            # loss still carries grad; the rest are already detached
            meters[k].append(float(stats[k].detach()))

        if step % cfg.train.log_every == 0:
            dt = (time.time() - t0) / cfg.train.log_every
            values = " ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in meters.items())
            print(f"[{step:>7}/{cfg.train.steps}] {values} "
                  f"epoch={step / steps_per_epoch:.2f} sec/step={dt:.3f}")
            t0 = time.time()

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            path = os.path.join(cfg.train.out_dir, f"ckpt_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "backbone": model.backbone_kind,
                    "in_channels": dataset.in_channels,
                    "use_normals": bool(cfg.data.use_normals),
                    "class_agnostic": bool(cfg.model.class_agnostic),
                    "aux_semantic": bool(model.aux_semantic),
                    "config": dict(cfg),
                },
                path,
            )
            print(f"[train] saved {path}")

    print(f"[train] done. backbone={model.backbone_kind}")


if __name__ == "__main__":
    main()
