"""Data loading.

`labels` is pure Python and always cheap to import. The dataset pulls in torch
and plyfile, so it is loaded lazily: importing metrics or labels must not
require the full training stack.
"""

from reify.data.labels import (
    NUM_CLASSES,
    SCANNET20_NAMES,
    STUFF_CLASSES,
    THING_CLASSES,
)

__all__ = [
    "NUM_CLASSES",
    "SCANNET20_NAMES",
    "STUFF_CLASSES",
    "THING_CLASSES",
    "ScanNetDataset",
    "collate",
]

_LAZY = {"ScanNetDataset": "reify.data.scannet", "collate": "reify.data.scannet"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
