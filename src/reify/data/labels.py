"""ScanNet20 label definitions.

NYU40 ids map onto the 20 evaluated ScanNet classes. Anything not in the map
becomes -1 and is ignored by both the loss and the metrics.
"""

NYU40_TO_SCANNET20 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 14: 12, 16: 13, 24: 14, 28: 15, 33: 16, 34: 17,
    36: 18, 39: 19,
}

NUM_CLASSES = 20

SCANNET20_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "other furniture",
]

# The official ScanNet instance benchmark scores 18 classes. Wall and floor are
# excluded because their instance boundaries are not consistently annotated.
# Panoptic scoring here follows that split: these two are treated as stuff.
STUFF_CLASSES = (0, 1)
THING_CLASSES = tuple(c for c in range(NUM_CLASSES) if c not in STUFF_CLASSES)
