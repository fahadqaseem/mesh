"""Farthest-point sampling (FPS): pick k points that spread evenly over a cloud."""
from __future__ import annotations

import numpy as np


def farthest_point_sample(points: np.ndarray, k: int, start: int | None = None) -> np.ndarray:
    """Indices of k points chosen greedily, each the farthest from those already chosen.

    Deterministic: starts at `start`, or by default at the point farthest from the cloud's centroid.
    Invariant to rotation, translation and uniform scale of the cloud. O(N k) time, O(N) memory.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if k >= n:
        return np.arange(n)
    if start is None:
        start = int(np.argmax(((pts - pts.mean(0)) ** 2).sum(1)))
    idx = np.empty(k, dtype=np.int64)
    idx[0] = start
    d = ((pts - pts[start]) ** 2).sum(1)
    for i in range(1, k):
        j = int(np.argmax(d))
        idx[i] = j
        np.minimum(d, ((pts - pts[j]) ** 2).sum(1), out=d)
    return idx
