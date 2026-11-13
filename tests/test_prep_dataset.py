import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prep_dataset import find_models, pick, scene_name  # noqa: E402


def test_shapenet_layout_and_pick(tmp_path):
    for syn, n in [("02691156", 3), ("03001627", 2), ("04379243", 1)]:
        for i in range(n):
            p = tmp_path / syn / f"model{i}" / "models"
            p.mkdir(parents=True)
            (p / "model_normalized.obj").write_text("v 0 0 0\n")
    models = find_models("shapenet", tmp_path)
    assert len(models) == 6 and models[0][0] == "02691156"
    a, b = pick(models, 3, seed=0), pick(models, 3, seed=0)
    assert a == b and len({m[0] for m in a}) == 3          # deterministic, one per category first
    assert scene_name("toys4k", "air plane", "plane_010") == "toys4k_air-plane_plane-010"
