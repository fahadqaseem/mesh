"""Gap filling: add points from a second cloud only where the first cloud's surface is missing."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def gap_fill_mask(base: np.ndarray, extra: np.ndarray, lo: float, hi: float = np.inf, min_nbrs: int = 0,
                  radius: float = 0.02) -> np.ndarray:
    """Which extra points to add: those lying in a gap of the base cloud, in a distance band.

    All distances are fractions of the base cloud's largest bbox extent (no GT needed).
    An extra point is kept when its distance to the nearest base point is in (lo, hi] (far enough to be a gap, close
    enough not to be a floater) and, if min_nbrs > 0, at least min_nbrs other extra points lie within `radius`
    (isolated stray samples are dropped).
    """
    ext = float(np.ptp(base, axis=0).max())
    d = cKDTree(base).query(extra)[0] / ext
    keep = (d > lo) & (d <= hi)
    if min_nbrs > 0 and keep.any():
        cand = np.flatnonzero(keep)
        counts = cKDTree(extra).query_ball_point(extra[cand], radius * ext, return_length=True) - 1
        keep[cand[counts < min_nbrs]] = False
    return keep
