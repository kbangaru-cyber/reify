"""Evaluate a checkpoint and write results.md.

    python scripts/evaluate.py --ckpt /path/to/ckpt_step6000.pt
    python scripts/evaluate.py --ckpt ... --limit 20      # quick check

Scoring happens at the level of the original points, with the training caps
disabled, so the number is comparable with published ScanNet mIoU. Nothing here
invents a figure: a class with no ground truth in the evaluated scenes is
reported as "not present" rather than counted as zero.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reify.config import load_config  # noqa: E402
from reify.data import ScanNetDataset, collate  # noqa: E402
from reify.data.labels import SCANNET20_NAMES  # noqa: E402
from reify.eval.inference import (  # noqa: E402
    instances_from_outputs,
    panoptic_from_outputs,
    superpoint_to_point,
)
from reify.eval.metrics import AgnosticMeter, ConfusionMatrix, PanopticMeter  # noqa: E402
from reify.models import OneFormer3DLike  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = args.split or cfg.eval.split
    limit = cfg.eval.limit if args.limit is None else args.limit
    out_path = args.out or cfg.eval.results_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device)

    # The checkpoint decides the feature width. A model trained on xyz+rgb
    # cannot be evaluated with normals appended, so follow what it recorded.
    use_normals = bool(state.get("use_normals", cfg.data.use_normals))
    backbone = state.get("backbone", cfg.model.backbone)
    step = state.get("step", "unknown")
    # The checkpoint also decides the head shape, so a class-agnostic model is
    # never scored with class-aware metrics by accident.
    class_agnostic = bool(state.get("class_agnostic", cfg.model.class_agnostic))
    aux_semantic = bool(state.get("aux_semantic", cfg.model.aux_semantic))
    print(f"[eval] checkpoint step={step} backbone={backbone} use_normals={use_normals} "
          f"mode={'class-agnostic' if class_agnostic else 'class-aware'}")

    dataset = ScanNetDataset(
        root=cfg.data.root,
        split=split,
        voxel_size=cfg.data.voxel_size,
        # caps off: this is what makes the number comparable
        max_points=0 if cfg.eval.point_level else cfg.data.max_points,
        max_superpoints=0 if cfg.eval.point_level else cfg.data.max_superpoints,
        use_normals=use_normals,
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.seed,
        verify=cfg.data.verify,
        return_points=True,
    )
    if limit:
        dataset.scene_ids = dataset.scene_ids[:limit]

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate)

    model = OneFormer3DLike(
        num_classes=cfg.model.num_classes,
        in_channels=dataset.in_channels,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_decoder_layers=cfg.model.num_decoder_layers,
        k_ins=cfg.model.k_ins,
        query_aug_std=0.0,
        backbone=backbone,
        class_agnostic=class_agnostic,
        aux_semantic=aux_semantic,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    confusion = ConfusionMatrix(cfg.model.num_classes)
    panoptic = PanopticMeter(cfg.model.num_classes)
    agnostic = AgnosticMeter()
    scenes = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split}"):
            tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(tensors)
            point_sp = batch["point_sp"][0].to(device)
            gt_sem = batch["point_sem"][0].numpy()
            gt_inst = batch["point_inst"][0].numpy()

            if class_agnostic:
                sp_inst, scores = instances_from_outputs(
                    out, tensors["sp_mask"],
                    mask_threshold=cfg.eval.mask_threshold,
                    score_threshold=cfg.eval.score_threshold,
                )
                pred_inst = superpoint_to_point(sp_inst, point_sp).cpu().numpy()
                agnostic.update(pred_inst, gt_inst, scores.cpu().numpy())
                if aux_semantic and out["sem_mask_logits"] is not None:
                    sp_sem = out["sem_mask_logits"][0].argmax(dim=0)
                    confusion.update(
                        superpoint_to_point(sp_sem, point_sp).cpu().numpy(), gt_sem
                    )
            else:
                sp_sem, sp_inst = panoptic_from_outputs(
                    out, tensors["sp_mask"],
                    num_classes=cfg.model.num_classes,
                    mask_threshold=cfg.eval.mask_threshold,
                    score_threshold=cfg.eval.score_threshold,
                )
                pred_sem = superpoint_to_point(sp_sem, point_sp).cpu().numpy()
                pred_inst = superpoint_to_point(sp_inst, point_sp).cpu().numpy()
                confusion.update(pred_sem, gt_sem)
                panoptic.update(pred_sem, pred_inst, gt_sem, gt_inst)
            scenes += 1

    miou = confusion.miou()

    if class_agnostic:
        a = agnostic.summary()
        print(f"\nAP50 {a['AP50']:.1f} | AP25 {a['AP25']:.1f} | recall {a['recall50']:.1f} "
              f"| splits {a['splits']:.2f} | merges {a['merges']:.2f} over {scenes} scenes")
        instance_section = (
            "## Class-agnostic instance grouping\n\n"
            "Labels are ignored entirely. The only question is whether the scene was\n"
            "carved into the right pieces. Walls and floors are scored here like any\n"
            "other object, which the class-aware path cannot do because it treats them\n"
            "as stuff.\n\n"
            + agnostic.as_markdown()
            + "\n\n`splits` above 1 means one object was carved into several. `merges`\n"
              "above 1 means several objects were fused. PQ hides both inside a single\n"
              "number, and they call for opposite fixes.\n"
        )
    else:
        summary = panoptic.summary()
        print(f"\nmIoU {100 * miou:.1f} | PQ {summary['PQ']:.1f} "
              f"SQ {summary['SQ']:.1f} RQ {summary['RQ']:.1f} over {scenes} scenes")
        instance_section = (
            "## Panoptic segmentation\n\n"
            f"**PQ {summary['PQ']:.1f}**, SQ {summary['SQ']:.1f}, RQ {summary['RQ']:.1f}\n\n"
            "Wall and floor are scored as stuff, matching the ScanNet convention that\n"
            "excludes them from instance evaluation because their instance boundaries\n"
            "are not consistently annotated.\n\n"
            + panoptic.as_markdown(SCANNET20_NAMES)
        )

    if confusion.matrix.sum():
        semantic_section = (
            "## Semantic segmentation\n\n"
            f"**mIoU {100 * miou:.1f}**, overall accuracy {100 * confusion.accuracy():.1f}\n\n"
            + confusion.as_markdown(SCANNET20_NAMES)
        )
    else:
        semantic_section = (
            "## Semantic segmentation\n\n"
            "Not measured: this checkpoint carries no semantic branch."
        )

    report = f"""# Results

Generated by `scripts/evaluate.py`. Every figure below was measured. A class with
no ground truth in the evaluated scenes is reported as "not present" rather than
counted as zero.

| | |
|---|---|
| checkpoint | `{Path(args.ckpt).name}` (step {step}) |
| backbone | **{backbone}** |
| mode | **{'class-agnostic' if class_agnostic else 'class-aware'}** |
| input features | {dataset.in_channels}-dim ({'xyz+rgb+normal' if use_normals else 'xyz+rgb'}) |
| split | {split} |
| scenes evaluated | **{scenes}** of {len(dataset.scene_ids) if not limit else limit} |
| scored at | {'original points, caps disabled' if cfg.eval.point_level else 'superpoints'} |
| voxel size | {cfg.data.voxel_size} m |
| generated | {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} |

{semantic_section}

{instance_section}
"""

    Path(out_path).write_text(report, encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
