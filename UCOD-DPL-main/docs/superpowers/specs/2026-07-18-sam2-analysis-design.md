# SAM2 Refinement Analysis — Design Spec

**Date**: 2026-07-18
**Status**: Design Approved
**Purpose**: Quantitative evidence for two innovation points in the SAM2-boundary-refinement paper

---

## 1. Deliverables

| # | File | Type | Runtime | Dependencies |
|---|------|------|---------|--------------|
| 1 | `scripts/offline_sam2_refine.py` | MODIFY | ~37 min | SAM2 installed |
| 2 | `experiments/analyze_label_quality.py` | CREATE | ~5 min | Existing 4040 PNG + 4040 pkl + GT |
| 3 | `experiments/analyze_offset_causality.py` | CREATE | ~5 min | Naive SAM2 labels + refined labels + per-image stats |

### Output files from deliverable 1:
- `refined_pseudo_labels/` — 4040 PNG (overwrite existing)
- `naive_sam2_labels/` — 4040 PNG (new)
- `refined_pseudo_labels/per_image_stats.json` — 4040 records (new)

### Output files from deliverable 2:
- `experiments/output/label_quality_report.md`
- `experiments/output/label_quality_per_image.csv`
- `experiments/output/figures/label_quality_*.png`

### Output files from deliverable 3:
- `experiments/output/offset_causality_report.md`
- `experiments/output/offset_per_image.csv`
- `experiments/output/figures/offset_*.png`

---

## 2. Script 1: `offline_sam2_refine.py` Modification

### 2.1 New CLI

```
python scripts/offline_sam2_refine.py --mode full   # existing behavior + per-image stats
python scripts/offline_sam2_refine.py --mode naive  # naive SAM2 only
python scripts/offline_sam2_refine.py --mode both   # default: both in one pass
```

### 2.2 Internal Flow (mode=both)

Naive SAM2 branch reuses the **exact same ablation override logic** that produced the v2 ablation training labels (lines 313-327 of original script). No new prompt definition — zero divergence from the trained v2.

```
for each image:
    1. Load image + 16x16 pkl → upsample to original resolution
       Skip and flag if coarse mask is empty (all zeros)
    2. Run SAM2 inference once → produce 3 candidate masks (shared)
    3. [Branch A — Naive SAM2, runs with ablation CFG overrides]
       a. Prompt: 1 random positive point from inside coarse mask
          + tight bbox (expand_base=0, expand_coeff=0)
          + ZERO negative points (n_neg_safe=0, n_neg_cautious=0)
       b. Stage 3: iou_lower=0.0, iou_upper=1.01
          → all 3 masks scored by IoU(mask_i, coarse) × SAM2_score_i, no exclusion
       c. Stage 4: alpha=0, beta=1, gamma=0
          → S = IoU_ori only (single-dimension, coarse-agreement-based gating)
       d. Save naive/{name}.png
    4. [Branch B — Full pipeline, default CFG, independent]
       a. Stage 1: Adaptive prompt (expanded bbox, hierarchical negatives, n_pos=5)
       b. Stage 2: [reuse masks from step 2] — no second SAM2 call
       c. Stage 3: Truncated multi-mask selection (3-zone thresholds)
       d. Stage 4: Edge-aware 3-factor confidence gating → M_final, S_score, gate_decision
       e. Save refined/{name}.png
       f. Collect stats: S_score, IoU_ori, EdgeAlign, IoU_pred, gate_decision, LocalSAM_triggered
```

Key: Naive SAM2's prompt is NOT centroid + corners. It is the exact CFG overrides from the original `ablation_flags_naive_sam2.json`: 1 random positive point + tight bbox + 0 negatives. This ensures the analysis labels are byte-identical to what trained the v2 ablation model.

Critical optimization: SAM2 inference (stage 2) is shared — run once, feed the same 3 candidate masks into both branches. This keeps total time at ~37 minutes instead of 2×.

### 2.3 `per_image_stats.json` Schema

```json
{
  "image_name": {
    "S_score": 0.92,
    "IoU_ori": 0.88,
    "EdgeAlign": 0.73,
    "IoU_pred": 0.95,
    "gate_decision": "full",
    "LocalSAM_triggered": false,
    "coarse_area_ratio": 0.05,
    "coarse_centroid": [320, 240],
    "selected_mask_idx": 1
  }
}
```

`gate_decision` values: `"full"` (S > 0.8), `"fusion"` (0.2 ≤ S ≤ 0.8), `"fallback"` (S < 0.2)

---

## 3. Script 2: `analyze_label_quality.py` (Innovation Point 1)

### 3.1 Comparison: Main Table (Binary-only)

| Label | Source |
|-------|--------|
| Coarse (binary) | 16×16 pkl → bilinear upsample → threshold 0.5 |
| Refined (ours) | Full 4-stage SAM2 pipeline → PNG |

**Metrics** (7): BF-score (width=3), Boundary Recall (R_b), Boundary Precision (P_b), IoU, S-measure, MAE, E-measure

### 3.2 Comparison: Supplementary Table (Soft label vs binary)

| Label | Source |
|-------|--------|
| Coarse (soft float) | 16×16 pkl → bilinear upsample → keep float [0,1] |
| Coarse (binary 0.5) | Same → threshold 0.5 |
| Refined (ours) | Full 4-stage SAM2 pipeline → PNG |

**Metrics** (3): MAE, S-measure, Transition Zone Ratio (pixels in [0.1, 0.9])

Purpose: Prove that even retaining float values, 16×16 upsampled labels suffer from inherent boundary blur, and SAM2 refinement is necessary.

### 3.3 Statistics Per Metric

- Mean ± 95% CI (bootstrap 1000)
- Median
- Per-stratum breakdown

### 3.4 Stratification Dimensions (4)

1. **All** (4040 images)
2. **By dataset**: TR-CAMO (1000) vs TR-COD10K (3040)
3. **By target area ratio** (GT foreground pixels / total pixels): tertiles (small / medium / large). Computed in Script 2 from GT masks — NOT from coarse labels, which would misclassify under-segmented large targets as small.
4. **By coarse label quality**: IoU(coarse_binary, GT) tertiles (low / medium / high)

### 3.5 Output Figures

- `hist_bfscore_refined_vs_coarse.png` — overlaid histograms
- `scatter_iou_coarse_vs_refined.png` — per-image scatter, color-coded by area tertile
- `boxplot_stratified_*.png` — one boxplot per stratification dimension

---

## 4. Script 3: `analyze_offset_causality.py` (Innovation Point 2)

### 4.1 Prerequisites

Must run Script 1 (`--mode both`) first to produce:
- `naive_sam2_labels/` — 4040 PNG
- `refined_pseudo_labels/` — 4040 PNG
- `per_image_stats.json`

### 4.2 Layer 1: Quantify Spatial Offset

Compute per image:
- **Centroid distance**: Euclidean distance between coarse mask centroid (largest connected component) and GT centroid (largest connected component), normalized by image diagonal
- **IoU(coarse_binary, GT)**: overall quality metric capturing both translation and shape distortion

Output: scatter plot (centroid distance vs IoU) + histogram of centroid distances

### 4.3 Layer 2: Offset → Naive SAM2 Failure

Stratify by IoU(coarse_binary, GT) tertiles (low/medium/high, ~1347 each).

**Balance check**: Per group, report mean ± std of:
- GT boundary complexity (Canny edge count normalized by object area)
- Target area ratio

Per group, compare:
| Label | Metrics |
|-------|---------|
| Coarse (binary) | BF-score, IoU, MAE |
| Naive SAM2 | BF-score, IoU, MAE |
| Δ (Naive − Coarse) | signed difference |

Expected evidence: high-offset (low IoU) group shows negative or near-zero Δ; low-offset group shows positive Δ.

### 4.4 Layer 3: Our Mechanism Repairs the Damage

Same stratification, add refined labels:

| Label | Metrics |
|-------|---------|
| Coarse (binary) | BF-score, IoU, MAE |
| Naive SAM2 | BF-score, IoU, MAE |
| **Refined (ours)** | **BF-score, IoU, MAE** |

Expected evidence: in high-offset group, refined significantly better than naive, not worse than coarse.

Additional: cross-tabulate `gate_decision` × offset tertile using per-image stats. Expected: fusion+fallback cases concentrated in high-offset group.

### 4.5 Output Figures

- `scatter_centroid_vs_iou.png` — per-image scatter
- `bar_grouped_offset_strata.png` — 3-group × 3-label grouped bar chart
- `pie_gate_by_offset.png` — pie charts of gate_decision per offset tertile

---

## 5. Constraints

- **Environment**: `C:\Anaconda\envs\test01`, Python 3.9
- **GPU**: 8GB VRAM
- **SAM2 model**: `sam2.1_hiera_tiny`
- **Working directory**: `C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main`
- **Data paths**:
  - Coarse labels: `C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K\data_{i}.pkl`
  - Images: `C:\Users\23991\Desktop\RefCOD (1)\RefCOD\{TR-CAMO,TR-COD10K}\im\*.jpg`
  - GT: `C:\Users\23991\Desktop\RefCOD (1)\RefCOD\{TR-CAMO,TR-COD10K}\gt\*.png`

## 6. Metrics Implementation Notes

- **BF-score**: boundary F-measure with 3-pixel tolerance. Steps: (1) extract boundary pixels from pred and GT binary masks via `cv2.findContours`; (2) compute `cv2.distanceTransform` on GT boundary map → distance field; (3) for each pred boundary pixel, look up distance in the GT distance field — match counted if distance ≤ 3. Compute Precision = matched_pred_pixels / total_pred_boundary_pixels, Recall = matched_gt_pixels / total_gt_boundary_pixels (reverse the distance transform direction), F1 = 2*P*R/(P+R). Use `cv2.DIST_L2` with `maskSize=3` for distance transform. This is O(N) per image, not O(N²).
- **Boundary Recall (R_b)**: fraction of GT boundary pixels that have a pred boundary pixel within 3px. Computed via distanceTransform of pred boundary, query GT boundary pixels.
- **Boundary Precision (P_b)**: fraction of pred boundary pixels that have a GT boundary pixel within 3px. Computed via distanceTransform of GT boundary, query pred boundary pixels.
- **S-measure**: reuse from `engine/utils/metrics/metric.py` (project's existing COD evaluation code, verified correct by CVPR 2025 paper).
- **E-measure**: reuse from `engine/utils/metrics/metric.py` (same as above).
- **MAE**: per-pixel |pred − gt|, for binary masks both in [0,1]
- **Transition Zone Ratio**: fraction of pixels with value ∈ [0.1, 0.9], computed on the float [0,1] soft label matrix
- **Centroid**: compute from largest connected component (use `cv2.connectedComponentsWithStats`). For multi-region GT, use the largest component's centroid.

## 7. Edge Cases

- **Empty coarse mask**: skip image, log warning, exclude from all aggregate statistics. Empty coarse masks have no foreground → no centroid, no IoU, and SAM2 has nothing to refine.
- **Empty refined/naive mask**: if SAM2 outputs all-zero mask (failed to find target), IoU with GT = 0, BF-score = 0, MAE = GT foreground ratio. These are valid data points — they represent SAM2 failure, which is exactly what the analysis needs to capture.
- **Multi-region GT**: for centroid distance, use only the largest connected component of GT. For IoU and other metrics, use the full GT mask.

## 8. Index Mapping

The `plable/TR-CAMO+TR-COD10K/index.json` maps `str(index)` → relative pkl filename (e.g., `"0": "data_0.pkl"`). Image names are derived from the pkl basename (e.g., `data_0.jpg` → lookup in TR-CAMO/im/ and TR-COD10K/im/). GT names match image names but with `.png` extension and are located in the corresponding `gt/` subdirectory.

Script 2 and 3 must build a unified mapping at startup: `image_name → {pkl_path, image_path, gt_path, refined_png_path, naive_png_path}`.

## 9. Ablation Variant Mapping (3 Variants)

The original 6-variant design is reduced to 3:

| # | Variant | Prompt | Stage 3/4 | Status |
|---|---------|--------|-----------|--------|
| 1 | Baseline | N/A | N/A (original 16×16 pkl) | Already exists |
| 2 | Naive SAM2 | 1 random pos point + tight bbox + 0 negatives | IoU_ori-only gating | Generated by Script 1 |
| 3 | Full model | 5 pos points + expanded bbox + 3 safe + 2 cautious negs | 3-zone selection + 3-factor edge gating | Generated by Script 1 |

The key causal narrative is: v1 (baseline) → v2 (naive SAM2 risks over-segmentation, especially at high offset) → v3 (our gating mechanism repairs the damage).

**Evidence chain integrity**: Script 1's naive SAM2 uses the exact same CFG overrides as the original `ablation_flags_naive_sam2.json`. The labels produced are functionally identical to what trained the v2 ablation model. To verify: after Script 1 runs, diff the first 100 naive labels against stored v2 ablation labels (if available), or compare the gate_decision distribution (should match the reported 7.8% full / 77.7% fusion / 14.5% fallback).
