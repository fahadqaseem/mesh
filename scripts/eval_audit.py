"""Evaluation consistency audit of the 16-19 Nov reports (Prof. Zhong, 17 Nov, item 5).

Usage: python scripts/eval_audit.py      -> prints a markdown table (pasted into docs/eval_protocol.md section 6)
For every cloud a report scores, it checks from the files on disk:
  - alignment: outputs/<run>/align/sim3.json comes from scripts/align.py with no "icp" key (camera alignment only);
  - count and seed: outputs/<run>/eval/equal_count.json has n = 1,000,000 and seed 0 (n_source = points before the draw);
  - GT: the GT mesh scored against is data/scenes/<s>/gt_normalized.ply; for runs aligned with data/scenes_low/<s> the
    GT mesh and the first 16 render cameras are checked to be byte- / value-identical;
  - signed offset present (100k-point subsample, seed 0) where the report has that column.
Meshes (seeded 4K) are checked for their MeshFlow json and for a 4,096-point input taken from the aligned 1M TSDF cloud.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_points_queue import DEV, SET_A, SET_B  # noqa: E402

O = Path("outputs")
LOW = {"_lowA", "_lowB", "_v12r3"}
REPORTS = [
    ("2026-11-17_equal_count", [DEV] + SET_A + SET_B, ["_dense", "_dfill"], True, False),
    ("2026-11-17_conf_ablation", SET_B, ["_dense", "_c50", "_c30", "_c10", "_c10e0", "_dfill"], True, False),
    ("2026-11-18_depth_scale", SET_B, ["_dense", "_dsg", "_dsv", "_lowA", "_lowB", "_v16"], True, True),
    ("2026-11-18_depth_scale (pencilpot tuning)", [DEV], ["_ds", "_ds_global_e0.005", "_ds_view_e0.005", "_ds_global_e0.01",
                                                         "_ds_view_e0.01", "_ds_global_e0.02", "_ds_view_e0.02",
                                                         "_ds_fixed_1.005", "_ds_fixed_1.01", "_ds_fixed_1.0125",
                                                         "_ds_fixed_1.015", "_ds_fixed_1.02"], True, True),
    ("2026-11-19_views_focal", SET_B, ["_dense", "_lowB", "_v12", "_v12r3", "_v16", "_dsg", "_jf", "_vfocal"], True, True),
    ("2026-11-19_views_focal (pencilpot tuning)", [DEV], ["_ds", "_ds_global_e0.005", "_jf_e0.005", "_jf_e0.01", "_vfocal"], True, True),
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def check_points(shapes, sufs, need_offset):
    bad, n_src, rms = [], [], []
    for s in shapes:
        for x in sufs:
            run = O / f"{s}{x}"
            sim = json.loads((run / "align" / "sim3.json").read_text())
            if "icp" in sim:
                bad.append(f"{run.name}: ICP in alignment")
            rms.append(sim["camera_rms"])
            f = run / "eval" / "equal_count.json"
            if not f.exists():
                bad.append(f"{run.name}: no equal_count.json")
                continue
            e = json.loads(f.read_text())
            if e["n"] != 1_000_000 or e["seed"] != 0:
                bad.append(f"{run.name}: n {e['n']} seed {e['seed']}")
            n_src.append(e["n_source"])
            if need_offset and not (run / "eval" / "signed_offset.json").exists():
                bad.append(f"{run.name}: no signed offset")
            if x in LOW:
                a, b = Path("data/scenes") / s, Path("data/scenes_low") / s
                if md5(a / "gt_normalized.ply") != md5(b / "gt_normalized.ply") or \
                        np.abs(np.load(a / "gt_cam_poses.npy") - np.load(b / "gt_cam_poses.npy")[:16]).max() > 0:
                    bad.append(f"{run.name}: scenes_low GT differs")
    return bad, n_src, rms


def main() -> None:
    print("| report | clouds checked | alignment | points scored | n before the draw | GT | problems |")
    print("|---|---:|---|---|---|---|---|")
    for name, shapes, sufs, _, off in REPORTS:
        bad, n_src, rms = check_points(shapes, sufs, off)
        n = len(shapes) * len(sufs)
        print(f"| {name} | {n} | cameras only (rms {min(rms):.3f}-{max(rms):.3f}) | 1,000,000, seed 0 | "
              f"{min(n_src):,}-{max(n_src):,} | data/scenes{' (+ scenes_low, identical)' if any(x in LOW for x in sufs) else ''} | "
              f"{'; '.join(bad) or 'none'} |")
    # seeded 4K: meshes
    bad = []
    for s in SET_B:
        for x in ["_k4random", "_k4fps", "_k4wfps"]:
            run = O / f"{s}{x}"
            if not (run / "mesh" / "meshflow_s42.json").exists():
                bad.append(f"{run.name}: no mesh json")
            sel = json.loads((run / "align" / "select.json").read_text())
            if sel.get("n", sel.get("k", 4096)) != 4096:
                bad.append(f"{run.name}: {sel}")
    print(f"| 2026-11-19_seeded_4k | {3 * len(SET_B)} meshes | inputs from the camera-aligned `<s>_dense`; mesh mapped back by "
          f"MeshFlow's exact inverse normalisation | mesh protocol: 5,000 / 100,000 surface samples | 4,096 input points | "
          f"data/scenes | {'; '.join(bad) or 'none'} |")
    # outliers report: union clouds built in the script
    rms_g = [json.loads((O / f"{s}_gskd" / "align" / "sim3.json").read_text()) for s in [DEV] + SET_A + SET_B]
    rms_d = [json.loads((O / f"{s}_dense" / "align" / "sim3.json").read_text()) for s in [DEV] + SET_A + SET_B]
    same = all(np.allclose(a["R"], b["R"]) and np.isclose(a["s"], b["s"]) for a, b in zip(rms_g, rms_d))
    icp = any("icp" in a for a in rms_g)
    print(f"| 2026-11-18_outliers | {len(rms_g)} filled clouds per filter | cameras only{' (ICP!)' if icp else ''}; kernel "
          f"samples moved with `<s>_gskd`'s sim3, {'identical to' if same else 'DIFFERENT from'} `<s>_dense`'s | all points "
          f"of TSDF 1M + kept gap points (not redrawn to 1M) | 1,000,000 + gap points | data/scenes | see exceptions |")


if __name__ == "__main__":
    main()
