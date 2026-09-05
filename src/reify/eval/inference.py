"""Turn model outputs into a panoptic labelling.

Things come from the instance queries, stuff from the semantic head, which is
the standard fusion. Queries are applied highest score first and may not claim a
superpoint that an earlier query already took, so the result has no overlaps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from reify.data.labels import STUFF_CLASSES


@torch.no_grad()
def panoptic_from_outputs(
    out: dict,
    sp_mask: torch.Tensor,
    num_classes: int = 20,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.25,
    stuff: tuple[int, ...] = STUFF_CLASSES,
):
    """Return (sem, inst) per superpoint for one scene, as (M,) int64 tensors.

    Instance id -1 marks stuff or unassigned superpoints.
    """
    valid = sp_mask[0]
    m = int(valid.sum())

    sem = out["sem_mask_logits"][0, :, :m].argmax(dim=0)  # (M,)
    inst = torch.full((m,), -1, dtype=torch.long, device=sem.device)

    cls_prob = F.softmax(out["inst_logits"][0], dim=-1)[:, :num_classes]  # drop no-object
    mask_prob = torch.sigmoid(out["inst_mask_logits"][0, :, :m])          # (K,M)

    best_prob, best_cls = cls_prob.max(dim=-1)
    binary = mask_prob > mask_threshold
    counts = binary.sum(dim=1)
    # mask quality: mean probability inside the thresholded region
    quality = (mask_prob * binary).sum(dim=1) / counts.clamp_min(1)
    score = best_prob * quality
    score = torch.where(counts > 0, score, torch.zeros_like(score))

    keep = (score >= score_threshold) & (counts > 0)
    for cls in stuff:
        keep &= best_cls != cls  # stuff never comes from an instance query

    order = torch.argsort(score, descending=True)
    next_id = 0
    for q in order.tolist():
        if not keep[q]:
            continue
        take = binary[q] & (inst < 0)
        if int(take.sum()) == 0:
            continue
        sem[take] = best_cls[q]
        inst[take] = next_id
        next_id += 1

    return sem, inst


def superpoint_to_point(values: torch.Tensor, point_sp: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-superpoint labelling back onto the original points."""
    safe = point_sp.clamp(0, values.shape[0] - 1)
    out = values[safe]
    return torch.where(point_sp >= 0, out, torch.full_like(out, -1))


@torch.no_grad()
def instances_from_outputs(
    out: dict,
    sp_mask: torch.Tensor,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.25,
):
    """Class-agnostic instance labelling for one scene.

    Returns (instance_ids, scores) where instance_ids is (M,) int64 with -1 for
    unassigned superpoints, and scores holds one confidence per emitted instance
    in id order, which class-agnostic average precision needs.

    Unlike the panoptic path there is no stuff, so nothing is excluded by class:
    walls and floors compete for superpoints on the same footing as furniture.
    That is the only way this pipeline can ever learn to separate one wall from
    the next, since the class-aware path discards them as stuff.
    """
    valid = sp_mask[0]
    m = int(valid.sum())
    inst = torch.full((m,), -1, dtype=torch.long, device=sp_mask.device)

    # index 0 is object, index 1 is no-object
    objectness = F.softmax(out["inst_logits"][0], dim=-1)[:, 0]
    mask_prob = torch.sigmoid(out["inst_mask_logits"][0, :, :m])

    binary = mask_prob > mask_threshold
    counts = binary.sum(dim=1)
    quality = (mask_prob * binary).sum(dim=1) / counts.clamp_min(1)
    score = torch.where(counts > 0, objectness * quality, torch.zeros_like(objectness))

    keep = (score >= score_threshold) & (counts > 0)
    order = torch.argsort(score, descending=True)

    scores: list[float] = []
    next_id = 0
    for q in order.tolist():
        if not keep[q]:
            continue
        take = binary[q] & (inst < 0)
        if int(take.sum()) == 0:
            continue
        inst[take] = next_id
        scores.append(float(score[q]))
        next_id += 1

    return inst, torch.tensor(scores, device=inst.device)
