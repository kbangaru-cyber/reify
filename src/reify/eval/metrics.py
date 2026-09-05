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


class AgnosticMeter:
    """Class-agnostic instance grouping quality.

    Ignores labels entirely and asks one question: did the model carve the scene
    into the right pieces? Reports

      AP50 / AP25   average precision at IoU 0.5 and 0.25, ranked by score
      recall        fraction of ground truth objects matched at IoU 0.5
      splits        predicted segments per matched ground truth object, so a
                    value above 1 means one object was carved into several
      merges        ground truth objects per matched prediction, above 1 means
                    several objects were fused into one

    Splits and merges are the diagnostic that PQ hides: PQ collapses both
    failures into one number, but they call for opposite fixes.
    """

    def __init__(self, iou_thresholds: tuple[float, ...] = (0.25, 0.5)):
        self.iou_thresholds = iou_thresholds
        self._scores: list[float] = []
        self._matched: dict[float, list[bool]] = {t: [] for t in iou_thresholds}
        self.n_gt = 0
        self._split_num = 0
        self._split_den = 0
        self._merge_num = 0
        self._merge_den = 0

    @staticmethod
    def _masks(inst: np.ndarray) -> list[np.ndarray]:
        return [np.flatnonzero(inst == i) for i in np.unique(inst[inst >= 0])]

    def update(self, pred_inst: np.ndarray, gt_inst: np.ndarray,
               scores: np.ndarray | None = None) -> None:
        gt = self._masks(gt_inst)
        pred = self._masks(pred_inst)
        self.n_gt += len(gt)
        if not pred:
            return

        if scores is None or len(scores) != len(pred):
            scores = np.ones(len(pred), dtype=np.float64)

        n = gt_inst.shape[0]
        iou = np.zeros((len(pred), len(gt)), dtype=np.float64)
        occupancy = np.zeros(n, dtype=np.int64)
        for j, g in enumerate(gt):
            occupancy[:] = 0
            occupancy[g] = 1
            for i, p in enumerate(pred):
                inter = int(occupancy[p].sum())
                if inter:
                    iou[i, j] = inter / (len(p) + len(g) - inter)

        order = np.argsort(-np.asarray(scores))
        for threshold in self.iou_thresholds:
            claimed = set()
            for i in order:
                j = int(np.argmax(iou[i])) if iou.shape[1] else -1
                hit = j >= 0 and iou[i, j] > threshold and j not in claimed
                if hit:
                    claimed.add(j)
                self._matched[threshold].append(bool(hit))
                if threshold == self.iou_thresholds[-1]:
                    self._scores.append(float(scores[i]))

        # Over- and under-segmentation, counted at IoU 0.25 so that partial
        # overlaps still register rather than vanishing.
        overlap = iou > 0.25
        for j in range(len(gt)):
            hits = int(overlap[:, j].sum())
            if hits:
                self._split_num += hits
                self._split_den += 1
        for i in range(len(pred)):
            hits = int(overlap[i].sum())
            if hits:
                self._merge_num += hits
                self._merge_den += 1

    def _ap(self, threshold: float) -> float:
        flags = np.asarray(self._matched[threshold], dtype=bool)
        if flags.size == 0 or self.n_gt == 0:
            return float("nan")
        order = np.argsort(-np.asarray(self._scores))
        flags = flags[order]
        tp = np.cumsum(flags)
        fp = np.cumsum(~flags)
        recall = tp / self.n_gt
        precision = tp / np.maximum(tp + fp, 1)
        # interpolate precision to be monotonically decreasing, then integrate
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))

    def summary(self) -> dict[str, float]:
        flags = np.asarray(self._matched.get(0.5, []), dtype=bool)
        return {
            "AP25": 100 * self._ap(0.25) if 0.25 in self.iou_thresholds else float("nan"),
            "AP50": 100 * self._ap(0.5) if 0.5 in self.iou_thresholds else float("nan"),
            "recall50": 100 * float(flags.sum()) / self.n_gt if self.n_gt else float("nan"),
            "splits": self._split_num / self._split_den if self._split_den else float("nan"),
            "merges": self._merge_num / self._merge_den if self._merge_den else float("nan"),
            "gt_objects": float(self.n_gt),
            "pred_objects": float(len(self._scores)),
        }

    def as_markdown(self) -> str:
        s = self.summary()
        return "\n".join([
            "| metric | value | reading |",
            "|---|---:|---|",
            f"| AP50 | {s['AP50']:.1f} | grouping quality at IoU 0.5 |",
            f"| AP25 | {s['AP25']:.1f} | at the looser IoU 0.25 |",
            f"| recall @ 0.5 | {s['recall50']:.1f} | share of real objects found |",
            f"| splits | {s['splits']:.2f} | predictions per object; >1 means over-segmented |",
            f"| merges | {s['merges']:.2f} | objects per prediction; >1 means fused |",
            f"| objects | {s['gt_objects']:.0f} ground truth, {s['pred_objects']:.0f} predicted | |",
        ])
