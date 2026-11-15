"""Queue for 8 vs 12 vs 16 VGGT views on all shapes (run after the guarded pilot on pencilpot + 3 set B shapes).

Usage: python scripts/make_views_queue.py   -> outputs/views_queue/queue.tsv (run with scripts/overnight.sh)
Each job is scripts/views_run.sh (VGGT under scripts/mem_guard.py, which kills the run if memory pressure turns red,
then align + points_eval + cloud_stats). 12 views = the 8 default views plus 1, 5, 10, 14 (6 per ring); 16 = all.
The 8-view rows are the dense default runs (<s>_dense). Step 1: set A; step 2: the other 9 set B shapes.
"""
from __future__ import annotations

from pathlib import Path

from make_points_queue import SET_A, SET_B

PILOT = ["thingi_103356", "thingi_77949", "thingi_593381"]
V12 = "0,1,2,4,5,6,9,10,11,13,14,15"


def main() -> None:
    jobs = []
    for step, shapes in [(1, SET_A), (2, [s for s in SET_B if s not in PILOT])]:
        for s in shapes:
            jobs += [(step, f"{s}_v12/run", f"scripts/views_run.sh {s} _v12 {V12}"),
                     (step, f"{s}_v16/run", f"scripts/views_run.sh {s} _v16 all")]
    out = Path("outputs/views_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{j}\t{c}\n" for st, j, c in jobs))
    print(f"[queue] {len(jobs)} jobs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
