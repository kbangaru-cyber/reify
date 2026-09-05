"""Interactive 3D comparison of ground truth against prediction.

    from reify.viz import show_scene
    fig, scene_id, info = show_scene(ckpt="/path/ckpt_step3000.pt", split="val")
    fig.show()

Four linked panels: input colour, ground truth semantics, predicted semantics,
predicted instances. The instance panel is the one worth studying, since it shows
directly whether grouping is over-splitting or collapsing objects.

Everything is drawn at voxel resolution. Voxels carry real positions and colours,
whereas points inside one superpoint share a single prediction, so drawing at
voxel level shows the model's actual output without pretending to a resolution it
does not have. This also means the ordinary train and val caches are enough; the
larger evaluation cache is not needed.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from reify.config import load_config
from reify.data.labels import SCANNET20_NAMES
from reify.viz.palette import instance_colors, semantic_colors


def _to_hex(rgb: np.ndarray) -> np.ndarray:
    v = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return np.array([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in v])


def _blank_axis() -> dict:
    return {"showgrid": False, "zeroline": False, "showticklabels": False,
            "title": "", "showbackground": False}


def _name(class_id) -> str:
    class_id = int(class_id)
    if 0 <= class_id < len(SCANNET20_NAMES):
        return SCANNET20_NAMES[class_id]
    return "ignored"


def show_scene(
    ckpt: str,
    scene_id: str | None = None,
    split: str = "val",
    config: str | None = None,
    max_points: int = 30_000,
    seed: int | None = None,
    point_size: float = 1.8,
    height: int = 900,
):
    """Render one scene. Returns (figure, scene_id, summary dict)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from reify.data.scannet import ScanNetDataset, collate
    from reify.eval.inference import instances_from_outputs, panoptic_from_outputs
    from reify.models import OneFormer3DLike

    cfg = load_config(config, verbose=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device)
    use_normals = bool(state.get("use_normals", cfg.data.use_normals))
    backbone = state.get("backbone", cfg.model.backbone)
    class_agnostic = bool(state.get("class_agnostic", cfg.model.class_agnostic))
    aux_semantic = bool(state.get("aux_semantic", cfg.model.aux_semantic))

    # Same settings the training cache was built with, so this is a cache hit.
    dataset = ScanNetDataset(
        root=cfg.data.root,
        split=split,
        voxel_size=cfg.data.voxel_size,
        max_points=cfg.data.max_points,
        max_superpoints=cfg.data.max_superpoints,
        use_normals=use_normals,
        cache_dir=cfg.data.cache_dir,
        seed=cfg.data.seed,
        verify=cfg.data.verify,
    )

    if scene_id is None:
        scene_id = random.Random(seed).choice(dataset.scene_ids)
    if scene_id not in dataset.scene_ids:
        raise ValueError(f"{scene_id} is not in the {split} split")

    batch = collate([dataset[dataset.scene_ids.index(scene_id)]])
    tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

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

    with torch.no_grad():
        out = model(tensors)
        if class_agnostic:
            sp_inst_pred, _ = instances_from_outputs(
                out, tensors["sp_mask"],
                mask_threshold=cfg.eval.mask_threshold,
                score_threshold=cfg.eval.score_threshold,
            )
            if out["sem_mask_logits"] is not None:
                sp_sem_pred = out["sem_mask_logits"][0].argmax(dim=0)
            else:
                sp_sem_pred = torch.full_like(sp_inst_pred, -1)
        else:
            sp_sem_pred, sp_inst_pred = panoptic_from_outputs(
                out, tensors["sp_mask"],
                num_classes=cfg.model.num_classes,
                mask_threshold=cfg.eval.mask_threshold,
                score_threshold=cfg.eval.score_threshold,
            )

    # Voxel geometry, and superpoint labels broadcast onto the voxels that own them.
    feats = batch["feats"].numpy()
    xyz, rgb = feats[:, :3], np.clip(feats[:, 3:6], 0, 1)
    vox_sp = batch["vox_sp_global"].numpy()

    gt_sem = batch["sp_sem"][0].numpy()[vox_sp]
    gt_inst = batch["sp_inst"][0].numpy()[vox_sp]
    pred_sem = sp_sem_pred.cpu().numpy()[vox_sp]
    pred_inst = sp_inst_pred.cpu().numpy()[vox_sp]

    n = xyz.shape[0]
    keep = np.arange(n)
    if n > max_points:
        keep = np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))

    panels = [
        ("Input colour", _to_hex(rgb[keep])),
        ("Ground truth instances", _to_hex(instance_colors(gt_inst[keep]))),
        ("Predicted instances", _to_hex(instance_colors(pred_inst[keep]))),
        (
            "Predicted semantics" if (pred_sem >= 0).any() else "Ground truth semantics",
            _to_hex(semantic_colors(pred_sem[keep] if (pred_sem >= 0).any() else gt_sem[keep])),
        ),
    ]
    hover = np.array([
        f"gt: {_name(g)}<br>pred: {_name(p)}<br>instance: {i}"
        for g, p, i in zip(gt_sem[keep], pred_sem[keep], pred_inst[keep])
    ])

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}] * 2] * 2,
        subplot_titles=[title for title, _ in panels],
        horizontal_spacing=0.02, vertical_spacing=0.06,
    )
    for k, (_, colors) in enumerate(panels):
        fig.add_trace(
            go.Scatter3d(
                x=xyz[keep, 0], y=xyz[keep, 1], z=xyz[keep, 2],
                mode="markers",
                marker={"size": point_size, "color": colors},
                text=hover, hoverinfo="text", showlegend=False,
            ),
            row=k // 2 + 1, col=k % 2 + 1,
        )

    labelled = gt_sem >= 0
    accuracy = float((pred_sem[labelled] == gt_sem[labelled]).mean()) if labelled.any() else float("nan")
    summary = {
        "scene_id": scene_id,
        "voxels": int(n),
        "shown": int(len(keep)),
        "voxel_accuracy": accuracy,
        "gt_instances": int(np.unique(gt_inst[gt_inst >= 0]).size),
        "pred_instances": int(np.unique(pred_inst[pred_inst >= 0]).size),
        "backbone": backbone,
        "step": state.get("step", "unknown"),
    }

    camera = {"eye": {"x": 1.4, "y": 1.4, "z": 1.1}}
    scenes = {
        f"scene{'' if i == 0 else i + 1}": {
            "aspectmode": "data", "camera": camera,
            "xaxis": _blank_axis(), "yaxis": _blank_axis(), "zaxis": _blank_axis(),
        }
        for i in range(4)
    }
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        title=(
            f"{scene_id} | step {summary['step']} | backbone {backbone} | "
            f"voxel accuracy {100 * accuracy:.1f}% | "
            f"{summary['pred_instances']} instances predicted, "
            f"{summary['gt_instances']} in ground truth"
        ),
        **scenes,
    )
    return fig, scene_id, summary
