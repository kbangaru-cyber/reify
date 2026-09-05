"""Segmentation metrics.

Semantic scoring accumulates a confusion matrix and reports per-class IoU plus
mIoU. Panoptic scoring follows the standard definition: a predicted segment
matches a ground truth segment of the same class when their IoU exceeds 0.5,
which is a unique match because the threshold is above one half.

    PQ = sum(IoU over true positives) / (TP + 0.5 * FP + 0.5 * FN)
    SQ = sum(IoU over true positives) / TP
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN)

Points labelled -1 are ignored throughout.
"""

from __future__ import annotations

import numpy as np

from reify.data.labels import NUM_CLASSES, SCANNET20_NAMES, STUFF_CLASSES


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        keep = (target >= 0) & (target < self.num_classes)
        pred, target = pred[keep], target[keep]
        pred = np.clip(pred, 0, self.num_classes - 1)
        flat = target.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        self.matrix += np.bincount(
            flat, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def iou_per_class(self) -> np.ndarray:
        tp = np.diag(self.matrix).astype(np.float64)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def miou(self) -> float:
        iou = self.iou_per_class()
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")

    def accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.diag(self.matrix).sum() / total) if total else float("nan")

    def as_markdown(self, names: list[str] | None = None) -> str:
        names = names or SCANNET20_NAMES
        iou = self.iou_per_class()
        support = self.matrix.sum(axis=1)
        lines = ["| class | IoU | points |", "|---|---:|---:|"]
        for i, name in enumerate(names[: self.num_classes]):
            value = "not present" if np.isnan(iou[i]) else f"{100 * iou[i]:.1f}"
            lines.append(f"| {name} | {value} | {support[i]:,} |")
        lines.append(f"| **mIoU** | **{100 * self.miou():.1f}** | |")
        return "\n".join(lines)


class PanopticMeter:
    """Accumulates PQ, SQ and RQ over scenes.

    Segments are (class, instance id) pairs. Stuff classes are matched as a
    single segment per class per scene, which is the usual convention and the
    reason ScanNet excludes wall and floor from instance scoring.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, stuff: tuple[int, ...] = STUFF_CLASSES,
                 iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.stuff = set(stuff)
        self.iou_threshold = iou_threshold
        self.iou_sum = np.zeros(num_classes, dtype=np.float64)
        self.tp = np.zeros(num_classes, dtype=np.int64)
        self.fp = np.zeros(num_classes, dtype=np.int64)
        self.fn = np.zeros(num_classes, dtype=np.int64)

    @staticmethod
    def _segments(sem: np.ndarray, inst: np.ndarray, stuff: set[int]) -> dict:
        """Map (class, id) to the point indices of that segment."""
        out: dict[tuple[int, int], np.ndarray] = {}
        valid = sem >= 0

        for cls in np.unique(sem[valid]):
            cls = int(cls)
            mask = valid & (sem == cls)
            if cls in stuff:
                out[(cls, -1)] = np.flatnonzero(mask)
                continue
            ids = inst[mask]
            for seg_id in np.unique(ids[ids >= 0]):
                out[(cls, int(seg_id))] = np.flatnonzero(mask & (inst == seg_id))
        return out

    def update(self, pred_sem: np.ndarray, pred_inst: np.ndarray,
               gt_sem: np.ndarray, gt_inst: np.ndarray) -> None:
        gt_segments = self._segments(gt_sem, gt_inst, self.stuff)
        pred_segments = self._segments(pred_sem, pred_inst, self.stuff)

        matched_pred: set[tuple[int, int]] = set()
        matched_gt: set[tuple[int, int]] = set()

        # Group by class so a prediction can only match ground truth of that class.
        by_class: dict[int, list] = {}
        for key, idx in pred_segments.items():
            by_class.setdefault(key[0], []).append((key, idx))

        for gt_key, gt_idx in gt_segments.items():
            cls = gt_key[0]
            gt_set = np.zeros(gt_sem.shape[0], dtype=bool)
            gt_set[gt_idx] = True
            for pred_key, pred_idx in by_class.get(cls, []):
                if pred_key in matched_pred:
                    continue
                inter = int(gt_set[pred_idx].sum())
                if inter == 0:
                    continue
                union = len(gt_idx) + len(pred_idx) - inter
                iou = inter / union
                if iou > self.iou_threshold:
                    self.tp[cls] += 1
                    self.iou_sum[cls] += iou
                    matched_pred.add(pred_key)
                    matched_gt.add(gt_key)
                    break

        for key in gt_segments:
            if key not in matched_gt:
                self.fn[key[0]] += 1
        for key in pred_segments:
            if key not in matched_pred:
                self.fp[key[0]] += 1

    def _per_class(self):
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        with np.errstate(invalid="ignore", divide="ignore"):
            pq = np.where(denom > 0, self.iou_sum / denom, np.nan)
            sq = np.where(self.tp > 0, self.iou_sum / np.maximum(self.tp, 1), np.nan)
            rq = np.where(denom > 0, self.tp / denom, np.nan)
        return pq, sq, rq

    def summary(self) -> dict[str, float]:
        pq, sq, rq = self._per_class()

        def mean(values: np.ndarray) -> float:
            # SQ is undefined when a class has no true positive, so an all-nan
            # column is a real answer rather than an error worth warning about.
            valid = values[~np.isnan(values)]
            return float(100 * valid.mean()) if valid.size else float("nan")

        return {"PQ": mean(pq), "SQ": mean(sq), "RQ": mean(rq)}

    def as_markdown(self, names: list[str] | None = None) -> str:
        names = names or SCANNET20_NAMES
        pq, sq, rq = self._per_class()
        lines = ["| class | PQ | SQ | RQ | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
        for i, name in enumerate(names[: self.num_classes]):
            if np.isnan(pq[i]):
                lines.append(f"| {name} | not present | | | 0 | 0 | 0 |")
                continue
            lines.append(
                f"| {name} | {100 * pq[i]:.1f} | {100 * sq[i]:.1f} | {100 * rq[i]:.1f} "
                f"| {self.tp[i]} | {self.fp[i]} | {self.fn[i]} |"
            )
        s = self.summary()
        lines.append(
            f"| **mean** | **{s['PQ']:.1f}** | **{s['SQ']:.1f}** | **{s['RQ']:.1f}** | | | |"
        )
        return "\n".join(lines)
