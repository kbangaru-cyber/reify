"""Configuration loading.

One YAML file holds every tunable value and every path. Any leaf can be
overridden from the environment so that a Colab session needs no file edits:

    REIFY_DATA__ROOT=/path/to/scannetv2_raw
    REIFY_TRAIN__STEPS=60000

The separator between nesting levels is a double underscore.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import yaml

_PREFIX = "REIFY_"
_SEP = "__"


class Config(dict):
    """A dict that also supports attribute access, one level deep per section."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _coerce(text: str) -> Any:
    """Turn an environment string into an int, float, bool, None or str."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _apply_env(cfg: dict) -> list[str]:
    """Override cfg in place from REIFY_ environment variables. Returns what changed."""
    applied = []
    for key, raw in os.environ.items():
        if not key.startswith(_PREFIX):
            continue
        path = key[len(_PREFIX):].lower().split(_SEP)
        node = cfg
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node = None
                break
            node = node[part]
        if node is None or path[-1] not in node:
            # Unknown key. Ignore rather than inventing config silently.
            continue
        node[path[-1]] = _coerce(raw)
        applied.append(f"{'.'.join(path)}={raw}")
    return applied


def load_config(path: str | os.PathLike | None = None, verbose: bool = True) -> Config:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    applied = _apply_env(cfg)
    if verbose and applied:
        print(f"[config] {path.name} with environment overrides: {', '.join(applied)}")
    elif verbose:
        print(f"[config] {path.name}")

    return Config(cfg)
