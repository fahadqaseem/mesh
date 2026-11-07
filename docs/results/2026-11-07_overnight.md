# Overnight results, 2026-11-07

Generated 2026-11-07 01:52 by `scripts/overnight_report.py` from `outputs/` (not in git).

Protocol v1 (`docs/eval_protocol.md`): camera-only alignment, no ICP; GT in [-0.5, 0.5]; CD/HD x100 at 5k samples
(MeshFlow definition, not squared); NC and F1 at 100k samples; size = mesh max extent / GT max extent (1 = right size).
Pipeline default v2: 8 views (4 per ring), known render focal, multi-view consistency (>= 2 views), 2 px foreground
erosion, 200k point cap; MeshFlow hybrid (DiT on MPS, extraction on CPU), 28 steps, guidance 1.0.
Lower CD/HD is better; higher NC/F1 is better.

Shape sets: **dev** = pencilpot_299291 (used to tune the filter thresholds; shown, but excluded from every mean). **Set A** = the other 7 original shapes. **Set B** = the 12 step-5 shapes, picked automatically and never looked at before this run: the clean test set.

## Headline (mean ± std (min-max) over shapes; dev shape excluded)

| stage | set | n | CD | HD | NC | F1@0.01 | F1@0.02 | V | size_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pipeline mesh, known focal (seed 42) | A | 7 | 10.32 ± 3.73 (4.94-15.16) | 28.0 ± 9.9 (15.1-41.2) | 0.769 ± 0.071 (0.642-0.841) | 0.298 ± 0.154 (0.103-0.508) | 0.473 ± 0.181 (0.231-0.723) | 1,682 ± 1,202 (240-3,904) | 1.05 ± 0.02 (1.03-1.09) |
| Pipeline mesh, known focal (seed 42) | B test | 0 | - | - | - | - | - | - | - |
| Pipeline mesh, known focal (mean of 3 seeds per shape) | A | 0 | - | - | - | - | - | - | - |
| Pipeline mesh, VGGT focal (uncalibrated, seed 42) | A | 7 | 11.77 ± 4.70 (6.65-18.73) | 30.9 ± 15.0 (9.3-55.2) | 0.774 ± 0.093 (0.587-0.860) | 0.235 ± 0.164 (0.072-0.515) | 0.406 ± 0.182 (0.152-0.680) | 739 ± 450 (202-1,513) | 1.10 ± 0.05 (1.02-1.14) |
| Upper bound: MeshFlow on clean GT samples (seed 42) | A | 7 | 2.74 ± 0.63 (1.55-3.60) | 4.2 ± 1.1 (2.3-5.4) | 0.908 ± 0.102 (0.685-0.977) | 0.815 ± 0.149 (0.627-0.975) | 0.979 ± 0.040 (0.890-1.000) | 256 ± 162 (53-473) | 1.02 ± 0.01 (1.01-1.03) |
| Upper bound: MeshFlow on clean GT samples (seed 42) | B test | 0 | - | - | - | - | - | - | - |
| Points stage: VGGT cloud before meshing (#V = points) | A | 7 | 9.32 ± 3.46 (4.62-13.26) | 27.1 ± 10.6 (13.1-40.3) | 0.800 ± 0.051 (0.692-0.839) | 0.414 ± 0.181 (0.176-0.645) | 0.559 ± 0.153 (0.320-0.727) | 179,489 ± 37,467 (97,891-200,000) | - |
| Points stage: VGGT cloud before meshing (#V = points) | B test | 0 | - | - | - | - | - | - | - |

CD floor (what a perfect mesh scores under this protocol): set A 2.26 ± 0.53 (1.27-2.85), set B -.

Seed spread (set A, 0 shapes with all 3 seeds; average std across seeds vs std across shapes): pending.

Grey vs textured (0 of 3 shapes, dev shape excluded; textured minus grey): points F1@0.02 -; mesh CD -; mesh F1@0.02 -; VGGT focal ratio grey - vs textured 0.942 ± 0.052 (0.882-0.979).

## Job status

- step 1: 72 of 72 jobs run (ok 72)
- step 2: 16 of 16 jobs run (ok 16)
- step 3: 0 of 16 jobs run (not started)
- step 4: 0 of 27 jobs run (not started)
- step 5: 0 of 84 jobs run (not started)

No failed or skipped jobs so far.

## Step 1: full pipeline, original 8 shapes (textured, 8 views, MeshFlow seed 42)

### Mesh (headline: known focal)

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s | CD floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 10.25 | 31.2 | 0.776 | 0.148 | 0.305 | 918 | 1,709 | 0.503 | 1.08 | 52 | 2.79 |
| cylinder_472094 | 4.94 | 15.1 | 0.751 | 0.508 | 0.723 | 3,904 | 18,832 | 0.591 | 1.05 | 53 | 1.27 |
| shackbot_94155 | 15.16 | 39.9 | 0.827 | 0.113 | 0.258 | 2,261 | 5,798 | 0.521 | 1.07 | 51 | 2.80 |
| nsphere_186199 | 12.94 | 29.2 | 0.820 | 0.103 | 0.231 | 1,243 | 2,001 | 0.466 | 1.03 | 52 | 2.06 |
| wheel_76903 | 9.39 | 26.7 | 0.841 | 0.328 | 0.510 | 1,283 | 3,229 | 0.533 | 1.04 | 51 | 2.38 |
| presents_98021 | 13.07 | 41.2 | 0.717 | 0.327 | 0.473 | 240 | 1,262 | 0.377 | 1.09 | 51 | 2.85 |
| monsterring_1356640 | 10.44 | 26.0 | 0.784 | 0.448 | 0.641 | 781 | 3,181 | 0.464 | 1.03 | 51 | 2.27 |
| cryptex_105686 | 6.32 | 17.8 | 0.642 | 0.256 | 0.472 | 2,065 | 6,031 | 0.498 | 1.04 | 51 | 2.15 |
| **mean ± std (min-max), set A, n=7** | 10.32 ± 3.73 (4.94-15.16) | 28.0 ± 9.9 (15.1-41.2) | 0.769 ± 0.071 (0.642-0.841) | 0.298 ± 0.154 (0.103-0.508) | 0.473 ± 0.181 (0.231-0.723) | 1,682 ± 1,202 (240-3,904) | 5,762 ± 6,033 (1,262-18,832) | 0.493 ± 0.067 (0.377-0.591) | 1.05 ± 0.02 (1.03-1.09) | 52 ± 1 (51-53) | |

### Mesh, uncalibrated variant (VGGT's own focal, same VGGT pass)

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s | VGGT focal ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 12.50 | 32.7 | 0.792 | 0.047 | 0.157 | 2,369 | 9,339 | 0.426 | 1.10 | 54 | 0.924 |
| cylinder_472094 | 7.74 | 22.1 | 0.786 | 0.402 | 0.558 | 1,513 | 8,328 | 0.480 | 1.12 | 50 | 0.912 |
| shackbot_94155 | 18.73 | 55.2 | 0.854 | 0.108 | 0.241 | 630 | 1,962 | 0.568 | 1.14 | 51 | 0.882 |
| nsphere_186199 | 16.84 | 34.1 | 0.821 | 0.072 | 0.152 | 955 | 2,866 | 0.478 | 1.14 | 51 | 0.837 |
| wheel_76903 | 9.64 | 25.9 | 0.860 | 0.146 | 0.456 | 580 | 1,595 | 0.568 | 1.04 | 51 | 0.964 |
| presents_98021 | 13.75 | 43.7 | 0.738 | 0.242 | 0.426 | 310 | 941 | 0.363 | 1.10 | 51 | 0.979 |
| monsterring_1356640 | 9.06 | 25.9 | 0.774 | 0.515 | 0.680 | 202 | 444 | 0.428 | 1.02 | 51 | 1.008 |
| cryptex_105686 | 6.65 | 9.3 | 0.587 | 0.163 | 0.327 | 980 | 4,824 | 0.414 | 1.10 | 51 | 0.922 |
| **mean ± std (min-max), set A, n=7** | 11.77 ± 4.70 (6.65-18.73) | 30.9 ± 15.0 (9.3-55.2) | 0.774 ± 0.093 (0.587-0.860) | 0.235 ± 0.164 (0.072-0.515) | 0.406 ± 0.182 (0.152-0.680) | 739 ± 450 (202-1,513) | 2,994 ± 2,755 (444-8,328) | 0.471 ± 0.077 (0.363-0.568) | 1.10 ± 0.05 (1.02-1.14) | 51 ± 0 (50-51) | |

### Points stage (VGGT cloud vs GT, known focal; V = points kept)

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | normal_outward | V |
|---|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 8.79 | 30.2 | 0.833 | 0.245 | 0.478 | 0.919 | 194,579 |
| cylinder_472094 | 4.62 | 14.6 | 0.839 | 0.604 | 0.727 | 0.902 | 172,890 |
| shackbot_94155 | 13.26 | 38.5 | 0.838 | 0.221 | 0.394 | 0.941 | 185,642 |
| nsphere_186199 | 12.49 | 30.7 | 0.805 | 0.176 | 0.320 | 0.946 | 97,891 |
| wheel_76903 | 8.04 | 26.8 | 0.822 | 0.513 | 0.659 | 0.950 | 200,000 |
| presents_98021 | 11.81 | 40.3 | 0.796 | 0.378 | 0.538 | 0.921 | 200,000 |
| monsterring_1356640 | 9.71 | 25.4 | 0.808 | 0.645 | 0.688 | 0.950 | 200,000 |
| cryptex_105686 | 5.28 | 13.1 | 0.692 | 0.362 | 0.589 | 0.867 | 200,000 |
| **mean ± std (min-max), set A, n=7** | 9.32 ± 3.46 (4.62-13.26) | 27.1 ± 10.6 (13.1-40.3) | 0.800 ± 0.051 (0.692-0.839) | 0.414 ± 0.181 (0.176-0.645) | 0.559 ± 0.153 (0.320-0.727) | 0.925 ± 0.031 (0.867-0.950) | 179,489 ± 37,467 (97,891-200,000) |

### Points stage, uncalibrated variant

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | normal_outward | V |
|---|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 11.41 | 32.8 | 0.803 | 0.053 | 0.152 | 0.889 | 200,000 |
| cylinder_472094 | 6.61 | 21.0 | 0.779 | 0.480 | 0.607 | 0.900 | 200,000 |
| shackbot_94155 | 19.23 | 46.0 | 0.818 | 0.064 | 0.154 | 0.945 | 200,000 |
| nsphere_186199 | 16.08 | 34.1 | 0.792 | 0.076 | 0.174 | 0.930 | 200,000 |
| wheel_76903 | 8.55 | 24.2 | 0.817 | 0.251 | 0.630 | 0.950 | 200,000 |
| presents_98021 | 12.34 | 41.2 | 0.793 | 0.332 | 0.523 | 0.914 | 200,000 |
| monsterring_1356640 | 8.36 | 25.6 | 0.796 | 0.651 | 0.719 | 0.948 | 200,000 |
| cryptex_105686 | 5.93 | 10.5 | 0.626 | 0.146 | 0.427 | 0.722 | 200,000 |
| **mean ± std (min-max), set A, n=7** | 11.01 ± 5.05 (5.93-19.23) | 28.9 ± 12.3 (10.5-46.0) | 0.775 ± 0.067 (0.626-0.818) | 0.286 ± 0.219 (0.064-0.651) | 0.462 ± 0.223 (0.154-0.719) | 0.901 ± 0.081 (0.722-0.950) | 200,000 ± 0 (200,000-200,000) |

## Step 2: upper bound, MeshFlow on clean GT surface samples (seed 42)

Not a pipeline result: shows what MeshFlow does with perfect input.

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s | CD floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 3.19 | 5.1 | 0.955 | 0.820 | 0.991 | 361 | 746 | 0.611 | 1.02 | 51 | 2.79 |
| cylinder_472094 | 1.55 | 2.3 | 0.970 | 0.975 | 1.000 | 53 | 99 | 0.443 | 1.03 | 51 | 1.27 |
| shackbot_94155 | 3.04 | 5.4 | 0.971 | 0.899 | 0.995 | 468 | 946 | 0.668 | 1.01 | 50 | 2.80 |
| nsphere_186199 | 2.49 | 3.4 | 0.930 | 0.867 | 0.994 | 177 | 374 | 0.582 | 1.01 | 50 | 2.06 |
| wheel_76903 | 2.62 | 4.2 | 0.977 | 0.967 | 1.000 | 166 | 333 | 0.688 | 1.01 | 50 | 2.38 |
| presents_98021 | 3.60 | 5.3 | 0.908 | 0.627 | 0.890 | 165 | 347 | 0.552 | 1.02 | 50 | 2.85 |
| monsterring_1356640 | 2.95 | 4.1 | 0.919 | 0.631 | 0.997 | 289 | 584 | 0.594 | 1.01 | 50 | 2.27 |
| cryptex_105686 | 2.96 | 4.8 | 0.685 | 0.737 | 0.975 | 473 | 2,179 | 0.473 | 1.03 | 50 | 2.15 |
| **mean ± std (min-max), set A, n=7** | 2.74 ± 0.63 (1.55-3.60) | 4.2 ± 1.1 (2.3-5.4) | 0.908 ± 0.102 (0.685-0.977) | 0.815 ± 0.149 (0.627-0.975) | 0.979 ± 0.040 (0.890-1.000) | 256 ± 162 (53-473) | 695 ± 706 (99-2,179) | 0.571 ± 0.091 (0.443-0.688) | 1.02 ± 0.01 (1.01-1.03) | 50 ± 0 (50-51) | |

## Step 3: seed variance (MeshFlow seeds 42, 7, 123 on each step-1 cloud)

Per shape: mean ± std over seeds.

| shape | seeds | CD | HD | NC | F1@0.01 | F1@0.02 | V | size_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291 (dev, not in means) | 1 | 10.25 | 31.2 | 0.776 | 0.148 | 0.305 | 918 | 1.08 |
| cylinder_472094 | 1 | 4.94 | 15.1 | 0.751 | 0.508 | 0.723 | 3,904 | 1.05 |
| shackbot_94155 | 1 | 15.16 | 39.9 | 0.827 | 0.113 | 0.258 | 2,261 | 1.07 |
| nsphere_186199 | 1 | 12.94 | 29.2 | 0.820 | 0.103 | 0.231 | 1,243 | 1.03 |
| wheel_76903 | 1 | 9.39 | 26.7 | 0.841 | 0.328 | 0.510 | 1,283 | 1.04 |
| presents_98021 | 1 | 13.07 | 41.2 | 0.717 | 0.327 | 0.473 | 240 | 1.09 |
| monsterring_1356640 | 1 | 10.44 | 26.0 | 0.784 | 0.448 | 0.641 | 781 | 1.03 |
| cryptex_105686 | 1 | 6.32 | 17.8 | 0.642 | 0.256 | 0.472 | 2,065 | 1.04 |
| **set A: mean ± std of per-shape means, n=0** | | - | - | - | - | - | - | - |

Seed spread (set A, 0 shapes with all 3 seeds; average std across seeds vs std across shapes): pending.

## Step 4: grey vs textured (8 views, default pipeline)

Points stage, then MeshFlow mean over seeds 42/7/123. Focal ratio = VGGT's predicted focal / true focal.

| shape | pts F1@0.02 grey | pts F1@0.02 tex | pts CD grey | pts CD tex | focal ratio grey | focal ratio tex | mesh CD grey | mesh CD tex | mesh F1@0.02 grey | mesh F1@0.02 tex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shackbot_94155 | - | 0.394 | - | 13.26 | - | 0.882 | - | 15.16 | - | 0.258 |
| presents_98021 | - | 0.538 | - | 11.81 | - | 0.979 | - | 13.07 | - | 0.473 |
| wheel_76903 | - | 0.659 | - | 8.04 | - | 0.964 | - | 9.39 | - | 0.510 |
| pencilpot_299291 (dev, not in means) | - | 0.478 | - | 8.79 | - | 0.924 | - | 10.25 | - | 0.305 |

Grey vs textured (0 of 3 shapes, dev shape excluded; textured minus grey): points F1@0.02 -; mesh CD -; mesh F1@0.02 -; VGGT focal ratio grey - vs textured 0.942 ± 0.052 (0.882-0.979).

## Step 5: clean test set, 12 new Thingi10K shapes (textured, 8 views, seed 42)

### Mesh

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s | CD floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| thingi_1146183 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_84620 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_77949 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_593381 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_73155 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_40550 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_290506 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_105924 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_103356 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_59706 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_697228 | - | - | - | - | - | - | - | - | - | - | - |
| thingi_94668 | - | - | - | - | - | - | - | - | - | - | - |
| **mean ± std (min-max), set B (test), n=0** | - | - | - | - | - | - | - | - | - | - | |

### Points stage

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | normal_outward | V |
|---|---:|---:|---:|---:|---:|---:|---:|
| thingi_1146183 | - | - | - | - | - | - | - |
| thingi_84620 | - | - | - | - | - | - | - |
| thingi_77949 | - | - | - | - | - | - | - |
| thingi_593381 | - | - | - | - | - | - | - |
| thingi_73155 | - | - | - | - | - | - | - |
| thingi_40550 | - | - | - | - | - | - | - |
| thingi_290506 | - | - | - | - | - | - | - |
| thingi_105924 | - | - | - | - | - | - | - |
| thingi_103356 | - | - | - | - | - | - | - |
| thingi_59706 | - | - | - | - | - | - | - |
| thingi_697228 | - | - | - | - | - | - | - |
| thingi_94668 | - | - | - | - | - | - | - |
| **mean ± std (min-max), set B (test), n=0** | - | - | - | - | - | - | - |

### Upper bound (MeshFlow on clean GT samples)

| shape | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| thingi_1146183 | - | - | - | - | - | - | - | - | - | - |
| thingi_84620 | - | - | - | - | - | - | - | - | - | - |
| thingi_77949 | - | - | - | - | - | - | - | - | - | - |
| thingi_593381 | - | - | - | - | - | - | - | - | - | - |
| thingi_73155 | - | - | - | - | - | - | - | - | - | - |
| thingi_40550 | - | - | - | - | - | - | - | - | - | - |
| thingi_290506 | - | - | - | - | - | - | - | - | - | - |
| thingi_105924 | - | - | - | - | - | - | - | - | - | - |
| thingi_103356 | - | - | - | - | - | - | - | - | - | - |
| thingi_59706 | - | - | - | - | - | - | - | - | - | - |
| thingi_697228 | - | - | - | - | - | - | - | - | - | - |
| thingi_94668 | - | - | - | - | - | - | - | - | - | - |
| **mean ± std (min-max), set B (test), n=0** | - | - | - | - | - | - | - | - | - | - |

## All non-dev shapes (set A + set B), pipeline mesh, seed 42

| n | CD | HD | NC | F1@0.01 | F1@0.02 | V | F | G | size_ratio | total_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 10.32 ± 3.73 (4.94-15.16) | 28.0 ± 9.9 (15.1-41.2) | 0.769 ± 0.071 (0.642-0.841) | 0.298 ± 0.154 (0.103-0.508) | 0.473 ± 0.181 (0.231-0.723) | 1,682 ± 1,202 (240-3,904) | 5,762 ± 6,033 (1,262-18,832) | 0.493 ± 0.067 (0.377-0.591) | 1.05 ± 0.02 (1.03-1.09) | 52 ± 1 (51-53) |
