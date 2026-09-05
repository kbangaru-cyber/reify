"""Colours for ScanNet20.

The semantic palette is the standard published mapping, carried over verbatim so
that renders stay comparable with earlier ones. Do not change these values.
"""

from __future__ import annotations

import numpy as np

SCANNET20_RGB_0_255 = np.array([
    (174, 199, 232),  # 0  wall
    (152, 223, 138),  # 1  floor
    (31, 119, 180),   # 2  cabinet
    (255, 187, 120),  # 3  bed
    (188, 189, 34),   # 4  chair
    (140, 86, 75),    # 5  sofa
    (255, 152, 150),  # 6  table
    (214, 39, 40),    # 7  door
    (197, 176, 213),  # 8  window
    (148, 103, 189),  # 9  bookshelf
    (196, 156, 148),  # 10 picture
    (23, 190, 207),   # 11 counter
    (247, 182, 210),  # 12 desk
    (219, 219, 141),  # 13 curtain
    (255, 127, 14),   # 14 refrigerator
    (158, 218, 229),  # 15 shower curtain
    (44, 160, 44),    # 16 toilet
    (112, 128, 144),  # 17 sink
    (227, 119, 194),  # 18 bathtub
    (82, 84, 163),    # 19 other furniture
], dtype=np.float32)

SCANNET20_RGB = (SCANNET20_RGB_0_255 / 255.0).astype(np.float32)
UNKNOWN_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float32)


def semantic_colors(class_ids: np.ndarray) -> np.ndarray:
    """(N,) class ids to (N,3) float RGB. Ignored points (-1) go grey."""
    ids = np.asarray(class_ids)
    out = np.tile(UNKNOWN_COLOR, (ids.shape[0], 1))
    valid = (ids >= 0) & (ids < len(SCANNET20_RGB))
    out[valid] = SCANNET20_RGB[ids[valid]]
    return out


def instance_colors(instance_ids: np.ndarray, seed: int = 0) -> np.ndarray:
    """(N,) instance ids to (N,3) float RGB.

    Colours are drawn from a hashed permutation so neighbouring ids look
    different, which is what makes over-segmentation visible at a glance.
    Stuff and unassigned points (-1) go grey.
    """
    ids = np.asarray(instance_ids)
    out = np.tile(UNKNOWN_COLOR, (ids.shape[0], 1))
    valid = ids >= 0
    if not valid.any():
        return out

    rng = np.random.default_rng(seed)
    n = int(ids[valid].max()) + 1
    # Golden-ratio hue stepping keeps consecutive ids far apart on the wheel.
    hues = (np.arange(n) * 0.61803398875 + rng.random()) % 1.0
    table = _hsv_to_rgb(hues, 0.65, 0.95)
    out[valid] = table[ids[valid]]
    return out


def _hsv_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1).astype(np.float32)
