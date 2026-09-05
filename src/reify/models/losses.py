"""Losses for the unified semantic and instance head.

Ported from the original notebook. The matching is disentangled in the
OneFormer3D sense: an instance query can only be matched to the ground truth
instance that owns the superpoint the query was sampled from, so there is no
Hungarian assignment over the full set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_cost(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1)
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def bce_cost(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)


def compute_losses(batch: dict, out: dict, num_classes: int = 20,
                   lambda_cls: float = 0.5) -> dict:
    device = batch["sp_inst"].device
    sp_inst, sp_sem, sp_mask = batch["sp_inst"], batch["sp_sem"], batch["sp_mask"]
    b, m_max = sp_inst.shape

    inst_logits = out["inst_logits"]
    inst_mask_logits = out["inst_mask_logits"]
    sem_mask_logits = out["sem_mask_logits"]
    src_sp = out["src_sp"]
    k = inst_logits.shape[1]
    no_object = num_classes

    # semantic: cross entropy per superpoint
    loss_sem = F.cross_entropy(
        sem_mask_logits.transpose(1, 2).reshape(-1, num_classes),
        sp_sem.reshape(-1),
        ignore_index=-1,
    )

    tgt_cls = torch.full((b, k), no_object, dtype=torch.long, device=device)
    tgt_mask = torch.zeros((b, k, m_max), dtype=torch.float32, device=device)
    positive = torch.zeros((b, k), dtype=torch.bool, device=device)
    cls_prob = F.softmax(inst_logits, dim=-1)

    for i in range(b):
        valid = sp_mask[i]
        inst_i, sem_i_all = sp_inst[i][valid], sp_sem[i][valid]
        m = int(inst_i.shape[0])
        if m == 0:
            continue

        gt_ids = torch.unique(inst_i[inst_i >= 0])
        if gt_ids.numel() == 0:
            continue

        src_local = src_sp[i].clamp(0, m - 1)
        src_inst = inst_i[src_local]
        pred_mask = inst_mask_logits[i, :, :m]

        for raw_id in gt_ids.tolist():
            gt = (inst_i == raw_id).float().unsqueeze(0)

            sem_in = sem_i_all[inst_i == raw_id]
            sem_in = sem_in[sem_in >= 0]
            if sem_in.numel() == 0:
                continue
            vals, counts = torch.unique(sem_in, return_counts=True)
            gt_class = int(vals[torch.argmax(counts)].item())

            cand = torch.where(src_inst == raw_id)[0]
            if cand.numel() == 0:
                continue

            expanded = gt.expand(cand.numel(), -1)
            cost = (
                -lambda_cls * cls_prob[i, cand, gt_class]
                + bce_cost(pred_mask[cand], expanded)
                + dice_cost(pred_mask[cand], expanded)
            )
            best = cand[torch.argmin(cost)]
            tgt_cls[i, best] = gt_class
            tgt_mask[i, best, :m] = gt.squeeze(0)
            positive[i, best] = True

    loss_cls = F.cross_entropy(inst_logits.reshape(-1, num_classes + 1), tgt_cls.reshape(-1))

    if positive.any():
        valid_pos = sp_mask.unsqueeze(1).expand(b, k, m_max)[positive]
        pred = inst_mask_logits[positive][valid_pos]
        tgt = tgt_mask[positive][valid_pos]
        loss_bce = F.binary_cross_entropy_with_logits(pred, tgt)
        probs = torch.sigmoid(pred)
        loss_dice = 1.0 - (2.0 * (probs * tgt).sum() + 1.0) / (probs.sum() + tgt.sum() + 1.0)
    else:
        loss_bce = torch.zeros((), device=device)
        loss_dice = torch.zeros((), device=device)

    return {
        "loss": loss_cls + loss_bce + loss_dice + loss_sem,
        "loss_cls": loss_cls.detach(),
        "loss_bce": loss_bce.detach(),
        "loss_dice": loss_dice.detach(),
        "loss_sem": loss_sem.detach(),
        "pos_queries": positive.sum().detach(),
    }
