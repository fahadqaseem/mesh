import numpy as np

from photo2mesh.points.fill import gap_fill_mask


def test_gap_fill_keeps_band_and_drops_isolated():
    rng = np.random.default_rng(0)
    # base: unit square in z=0 with a hole at x > 0.8
    base = np.c_[rng.uniform(0, 0.8, 20000), rng.uniform(0, 1, 20000), np.zeros(20000)]
    # extra: points filling the hole (dense), plus a few isolated floaters far above
    hole = np.c_[rng.uniform(0.85, 1.0, 3000), rng.uniform(0, 1, 3000), np.zeros(3000)]
    on_base = np.c_[rng.uniform(0, 0.8, 1000), rng.uniform(0, 1, 1000), np.zeros(1000)]
    floaters = np.c_[rng.uniform(0, 0.8, 5), rng.uniform(0, 1, 5), np.full(5, 0.5)]
    extra = np.vstack([hole, on_base, floaters])
    keep = gap_fill_mask(base, extra, lo=0.03, hi=0.3, min_nbrs=3, radius=0.02)
    assert keep[:3000][hole[:, 0] > 0.83].mean() > 0.95          # the hole is filled (past the band's inner edge)
    assert not keep[3000:4000].any()                              # nothing added where the base already is
    assert not keep[4000:].any()                                  # floaters dropped (too far, and isolated)
