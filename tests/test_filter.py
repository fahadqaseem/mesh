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
