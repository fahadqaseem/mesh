"""Score depth-scale runs: protocol v1 at exactly 1M points + outliers (equal_count.score) + signed offset (conf_report).

Usage: python scripts/ds_score.py <shape> <suffix> [<suffix> ...]   e.g. pencilpot_299291 _dense _ds_global_e0.01
Prints one line per run and leaves outputs/<shape><suffix>/eval/{equal_count.json, signed_offset.json}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conf_report import cloud, normalised, signed_offset  # noqa: E402
from equal_count import score  # noqa: E402


def main() -> None:
    shape = sys.argv[1]
    _, g, c, s = normalised(shape)
    for suf in sys.argv[2:]:
        r = score(shape, suf, 1_000_000, 0)
        p = cloud(shape, suf, c, s)
        off = signed_offset(p[np.random.default_rng(0).choice(len(p), 100_000, replace=False)], g)
        (Path("outputs") / f"{shape}{suf}" / "eval" / "signed_offset.json").write_text(json.dumps({"signed_offset": off}))
        print(f"{shape}{suf}: CD {r['CD']:.3f}  F1@0.01 {r['F1@0.01']:.3f}  R_all@0.02 {r['R_all@0.02']:.3f}  "
              f"out@0.01 {r['out@0.01']:.1f}  out@0.02 {r['out@0.02']:.1f}  offset {off:+.4f}", flush=True)


if __name__ == "__main__":
    main()
