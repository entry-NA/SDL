# Current Pseudo-Label State Audit

Generated from the current filesystem and code-adjacent artifacts.

## Label Directories

| Directory | PNG count | Empty | Full | Mean foreground ratio |
|---|---:|---:|---:|---:|
| `naive_sam2_labels` | 4028 | 0 | 0 | 0.378531 |
| `refined_pseudo_labels` | 4040 | 12 | 0 | 0.394467 |
| `refined_pseudo_labels_broken` | 4040 | 12 | 0 | 0.185986 |
| `refined_pseudo_labels_vloose` | 4026 | 0 | 0 | 0.269878 |

## Compared With `naive_sam2_labels`

- `refined_pseudo_labels`: common=4028, equal=40, mean pixel difference=0.155815, mean IoU=0.662107.
- `refined_pseudo_labels_broken`: common=4028, equal=1, mean pixel difference=0.308248, mean IoU=0.290159.
- `refined_pseudo_labels_vloose`: common=4026, equal=0, mean pixel difference=0.308038, mean IoU=0.364996.

## Missing Referenced Assets

- `raw_sam2_outputs`: MISSING
- `refined_pseudo_labels_backup`: MISSING
- `refined_pseudo_labels_backup_old`: MISSING

## Recorded Gate Statistics

- Entries: 4040
- Gate counts: `{'full': 3862, 'fusion': 174, 'fallback': 4}`
- Scores greater than 1: 3804
- Scores greater than 999: 0
- S-score summary: `{'min': 0.131329, 'p05': 0.89218025, 'median': 4.8462985, 'p95': 10.539090099999997, 'max': 20.664036, 'mean': 5.16162118490099}`

## Gating Algebra Finding

For `S > 1`, the current formula `S*SAM + (1-S)*coarse` gives the coarse mask a negative coefficient. After thresholding, pixels present only in the coarse mask are always removed, so the operation is not a convex fusion.

## Interpretation Rule

This report describes the current disk state only. It does not prove which label directory produced a historical checkpoint unless that run recorded an immutable configuration and input manifest.
