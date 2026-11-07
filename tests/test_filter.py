import numpy as np

from photo2mesh.points.filter import confidence_mask


def test_floor_confidence_is_dropped_even_when_it_is_the_percentile():
    # 60% background at exactly the VGGT floor (1.0), 40% object with real confidence
    conf = np.concatenate([np.ones(600), np.linspace(5, 90, 400)])
    keep = confidence_mask(conf, percentile=50, min_conf=1.0)
    assert not keep[:600].any()
    assert keep[600:].all()


def test_percentile_still_applies_above_floor():
    conf = np.linspace(2, 100, 1000)
    keep = confidence_mask(conf, percentile=50, min_conf=1.0)
    assert abs(keep.mean() - 0.5) < 0.01


def test_depth_edge_mask_drops_only_the_discontinuity():
    from photo2mesh.points.filter import depth_edge_mask
    depth = np.ones((1, 20, 20), np.float32)
    depth[:, :, 10:] = 2.0                                   # a step: object at 1, background at 2
    keep = depth_edge_mask(depth, rel_thresh=0.05, dilate=1)
    assert not keep[0, :, 8:12].any()                        # step columns (+1 px dilation) dropped
    assert keep[0, :, :8].all() and keep[0, :, 12:].all()    # flat parts kept
    tilted = np.tile(np.linspace(1.0, 1.1, 20, dtype=np.float32), (1, 20, 1))   # smooth slope: no edges
    assert depth_edge_mask(tilted, rel_thresh=0.05, dilate=1).all()


def test_foreground_erosion():
    from photo2mesh.points.filter import depth_edge_mask
    fg = np.zeros((1, 20, 20), bool)
    fg[:, 5:15, 5:15] = True
    keep = depth_edge_mask(np.ones((1, 20, 20), np.float32), rel_thresh=0.05, dilate=0, foreground=fg, erode=2)
    assert keep.sum() == 6 * 6
