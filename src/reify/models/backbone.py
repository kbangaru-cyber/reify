"""Voxel backbone, with the choice made explicit.

The original code silently fell back to a per-point MLP when MinkowskiEngine
failed to import, which meant a run could quietly train with no neighbourhood
aggregation at all. Here:

  * `backbone: auto` picks the best available and prints what it chose.
  * naming one explicitly (`mlp`, `spconv`, `minkowski`) makes a failed import a
    hard error, so a run never silently downgrades.

Whatever is chosen, the name is printed at construction and stored on the module
as `.kind` so training and evaluation can record it alongside every result.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _try_minkowski():
    try:
        import MinkowskiEngine as ME  # noqa: F401
        return ME
    except Exception:
        return None


def _try_spconv():
    try:
        import spconv.pytorch as spconv  # noqa: F401
        return spconv
    except Exception:
        return None


def resolve_backbone(requested: str = "auto") -> str:
    """Return the backbone kind to build, or raise if the request cannot be met."""
    requested = (requested or "auto").lower()

    if requested == "auto":
        if _try_spconv() is not None:
            return "spconv"
        if _try_minkowski() is not None:
            return "minkowski"
        print(
            "[backbone] auto: no sparse convolution library found, using the MLP "
            "fallback. This encodes every voxel independently, with no "
            "neighbourhood aggregation. Set model.backbone explicitly to make "
            "this a hard error instead."
        )
        return "mlp"

    if requested == "mlp":
        return "mlp"
    if requested == "spconv":
        if _try_spconv() is None:
            raise RuntimeError(
                "model.backbone=spconv but 'spconv' could not be imported. "
                "Install a build matching your CUDA runtime, for example "
                "'pip install spconv-cu120', or set model.backbone=auto."
            )
        return "spconv"
    if requested == "minkowski":
        if _try_minkowski() is None:
            raise RuntimeError(
                "model.backbone=minkowski but MinkowskiEngine could not be "
                "imported. It does not build against most recent CUDA versions. "
                "Prefer spconv, or set model.backbone=auto."
            )
        return "minkowski"

    raise ValueError(f"Unknown backbone {requested!r}. Use auto, mlp, spconv or minkowski.")


class Backbone(nn.Module):
    """Maps voxel features to per-voxel embeddings."""

    def __init__(self, in_ch: int = 6, out_ch: int = 128, kind: str = "auto"):
        super().__init__()
        self.kind = resolve_backbone(kind)
        self.in_ch = in_ch
        self.out_ch = out_ch

        if self.kind == "mlp":
            # Kept identical to the original fallback so old checkpoints load.
            self.net = nn.Sequential(
                nn.Linear(in_ch, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_ch),
                nn.ReLU(inplace=True),
            )
        elif self.kind == "minkowski":
            ME = _try_minkowski()
            self.net = nn.Sequential(
                ME.MinkowskiConvolution(in_ch, 32, kernel_size=3, stride=1, dimension=3),
                ME.MinkowskiBatchNorm(32),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(64, out_ch, kernel_size=3, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(out_ch),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(out_ch, out_ch, kernel_size=3, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(out_ch),
                ME.MinkowskiReLU(inplace=True),
            )
        else:  # spconv
            spconv = _try_spconv()
            self.net = spconv.SparseSequential(
                spconv.SubMConv3d(in_ch, 32, 3, bias=False, indice_key="sub0"),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(32, 64, 3, bias=False, indice_key="sub1"),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                spconv.SubMConv3d(64, out_ch, 3, bias=False, indice_key="sub2"),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        print(f"[backbone] kind={self.kind} in_ch={in_ch} out_ch={out_ch}")

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """coords is (V,4) as (batch, x, y, z). Returns (V, out_ch)."""
        if self.kind == "mlp":
            return self.net(feats)

        if self.kind == "minkowski":
            ME = _try_minkowski()
            return self.net(ME.SparseTensor(features=feats, coordinates=coords)).F

        # spconv requires non-negative indices inside a declared spatial shape,
        # and voxel coordinates from floor(xyz / voxel_size) are often negative.
        spconv = _try_spconv()
        idx = coords.clone().to(torch.int32)
        xyz = idx[:, 1:]
        xyz = xyz - xyz.min(dim=0, keepdim=True).values
        spatial_shape = (xyz.max(dim=0).values + 1).tolist()
        idx = torch.cat([idx[:, :1], xyz], dim=1).contiguous()
        batch_size = int(coords[:, 0].max().item()) + 1

        tensor = spconv.SparseConvTensor(feats, idx, spatial_shape, batch_size)
        return self.net(tensor).features
