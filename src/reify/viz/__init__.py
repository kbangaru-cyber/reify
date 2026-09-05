from reify.viz.palette import (
    SCANNET20_RGB,
    SCANNET20_RGB_0_255,
    UNKNOWN_COLOR,
    instance_colors,
    semantic_colors,
)

__all__ = [
    "SCANNET20_RGB",
    "SCANNET20_RGB_0_255",
    "UNKNOWN_COLOR",
    "instance_colors",
    "semantic_colors",
    "show_scene",
]

_LAZY = {"show_scene": "reify.viz.plotly_view"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
