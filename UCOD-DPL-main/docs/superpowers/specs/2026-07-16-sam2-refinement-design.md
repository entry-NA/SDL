# SAM2-Guided Pseudo-Label Boundary Refinement for UCOD-DPL

**Date**: 2026-07-16
**Status**: Design Approved
**Paper Target**: UCOD-DPL + SAM2 refinement = improved unsupervised COD

## 1. Overview

### 1.1 Problem

UCOD-DPL generates coarse pseudo-labels from DINOv2 attention maps (16x16, thresholded binary masks). These labels suffer from blurred boundaries and spatial drift. The APM module dynamically fuses them with teacher predictions, but the "source quality" bottleneck remains — garbage in, garbage out.

### 1.2 Solution

Use **frozen SAM2 (sam2.1_hiera_tiny)** as an offline boundary refiner before training. The coarse pseudo-label provides spatial guidance ("where the object is"), SAM2 provides boundary precision ("where the exact edge is"). A multi-check gating system prevents SAM2 from introducing over-segmentation errors.

### 1.3 Core Claim

> SAM2 as a frozen boundary specialist, guarded by confidence-weighted fallback, systematically improves unsupervised COD pseudo-label quality — without modifying the training architecture.

---

## 2. Architecture

### 2.1 System Overview

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  plable/      │     │  offline_sam2_      │     │  UCOD-DPL Train  │
│  4040 pkl     │────▶│  refine.py          │────▶│  (modified load) │
│  16x16 masks  │     │  4-stage pipeline   │     │                  │
└──────────────┘     └────────────────────┘     └────────┬─────────┘
      │                    │                             │
      │ RefCOD/im/         │ RefCOD/im/                  │
      │ original images    │ original images             │
      └────────────────────┴─────────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  Evaluation      │
                                                 │  COD10K/CAMO/    │
                                                 │  CHAMELEON/NC4K  │
                                                 └──────────────────┘
```

### 2.2 Three Modules

| Module | File | Purpose |
|--------|------|---------|
| 1. Offline Refinement | `scripts/offline_sam2_refine.py` | Run once before training. Input: original images + 16x16 pkl pseudo-labels. Output: refined PNG masks at original resolution |
| 2. Training Integration | `data/datasets/cache_manager.py` (modify) + `engine/runner/loop_UCOD_DPL.py` (guard) | Load refined PNGs at 68x68 directly, bypass 16x16 bottleneck |
| 3. Experiment Suite | `experiments/` | Main comparison table, ablation study, visualization |

---

## 3. Module 1: offline_sam2_refine.py

### 3.1 Input/Output

**Input:**
- Original RGB image at native resolution from `RefCOD (1)/RefCOD/{TR-CAMO,TR-COD10K}/im/*.jpg`
- Coarse pseudo-label from `plable/TR-CAMO+TR-COD10K/data_{i}.pkl` — torch.Tensor [1, 16, 16], binary {0,1}
- Index mapping from `plable/TR-CAMO+TR-COD10K/index.json` — `{index: filename}`

**Output:**
- `./datasets/cache/refined_pseudo_labels/{image_name}.png` — single-channel binary PNG at original image resolution

### 3.2 Four-Stage Pipeline

#### Stage 1: Adaptive Prompt Generation with Hierarchical Negative Sampling

```
1. Load coarse mask (16x16), bilinear-upsample to original image resolution
2. Compute bounding box from upsampled coarse mask
3. Adaptive expansion ratio:
   R = 0.30 - 0.15 * min(1, A / (H*W*0.01))
   where A = foreground area, H*W = image area
4. Expand bbox by ratio R (produce "expanded bbox")
5. Sample positive points: uniform-random N=5 from inside coarse mask
6. Sample negative points (hierarchical):
   - Layer 1 (safe): from outside the expanded bbox — guaranteed background
   - Layer 2 (cautious): from inside expanded bbox but outside coarse mask,
     only points farther than 0.5*sqrt(A) from coarse mask centroid
```

**Rationale**: Small objects have larger relative positioning error in DINOv2 attention maps, so expansion compensates. Layered negatives prevent false-negative contamination when the coarse mask is spatially shifted.

#### Stage 2: SAM2 Multi-Mask Inference

```
1. If coarse_mask_area < H*W*0.01 → trigger Local-SAM:
   - Crop the expanded bbox region from original image
   - Resize crop to 256x256
   - Feed to SAM2 with positive + negative points
   - Resize output mask back to original crop coordinates
   - Paste back into full-resolution mask
2. Otherwise → standard SAM2 inference at original resolution
3. SAM2 returns 3 candidate masks + 3 IoU prediction scores
```

**Rationale**: Small targets in a full image context are extremely hard for SAM2. Local-SAM magnification gives SAM2 a clear view. This is the "pre-hoc" analogue to UCOD-DPL's "post-hoc" Look-Twice mechanism.

#### Stage 3: Truncated Multi-Mask Selection

```
Algorithm: Truncated Multi-Mask Acceptance

Input:  M_sam[1..3], S_sam[1..3], M_coarse
Output: M_selected

for i in 1..3:
    iou[i] = IoU(M_sam[i], M_coarse)

for i in 1..3:
    if iou[i] < 0.25:
        candidate[i] = -inf           // SAM diverged, exclude
    elif iou[i] > 0.90:
        return M_sam[i]               // coarse label already good, take with confidence
    else:
        candidate[i] = iou[i] * S_sam[i]

return M_sam[argmax(candidate)]
```

**Rationale**: The upper bound exemption (0.90+) prevents "killing" good refinements where SAM only micro-adjusts boundaries. The lower bound (0.25) catches catastrophic SAM failures. The middle range uses joint scoring.

#### Stage 4: Edge-Aware Confidence Gating and Fallback

```
1. Compute edge alignment:
   E_I = Canny(I, sigma=1.5)          // image edge map
   M_boundary = dilate(M) - erode(M)   // morphological gradient, 3x3 kernel
   EdgeAlign = |M_boundary ∩ E_I| / |M_boundary|

2. Compute confidence score:
   S = 0.3 * IoU_pred + 0.4 * IoU(M_sam, M_coarse) + 0.3 * EdgeAlign

3. Three-tier fallback:
   if S < 0.2:        M_final = M_coarse                              // full fallback
   elif S > 0.8:      M_final = M_selected                            // full adoption
   else:              M_final = (S * M_selected + (1-S) * M_coarse) > 0.5  // soft fusion
```

**Rationale**: EdgeAlign catches SAM2 "hallucinations" — when SAM outputs a confident mask whose boundary cuts through flat texture regions. The three-factor S score balances SAM's self-assessed confidence, agreement with the coarse label, and boundary plausibility.

---

## 4. Module 2: Training Integration

### 4.1 Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `data/datasets/cache_manager.py` | `PseudoLabelCache.read_file()` — add PNG read path | ~10 |
| `engine/runner/loop_UCOD_DPL.py` | `_process_batch()` — guard against redundant upsample | ~3 |

### 4.2 cache_manager.py — Detailed Change

In `PseudoLabelCache` (or equivalent class), modify `read_file()`:

```python
def read_file(self, index):
    png_path = os.path.join(self.cache_dir, f"{self.index_map[str(index)]}.png")
    if os.path.exists(png_path):
        from PIL import Image
        import numpy as np
        img = Image.open(png_path).convert('L')
        img = img.resize((self.feature_size, self.feature_size), Image.LANCZOS)
        pseudo_label = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
        # Soft boundaries: keep 0~1 float values, no binarization
        return pseudo_label
    else:
        # Fallback: read original pkl
        return self._read_pkl(index)
```

**Key decisions:**
- LANCZOS resampling preserves SAM2 boundary information better than bilinear/bicubic
- Keep soft boundaries (float 0~1) — BCE loss expects probabilities, and soft edges provide smoother gradient signals
- Fallback to pkl ensures compatibility if refined labels don't exist

### 4.3 loop_UCOD_DPL.py — Guard Change

In `_process_batch()` (line ~154):

```python
# Old:
# pseudo_labels = F.interpolate(pseudo_labels, size=(h,w), mode='bilinear')

# New:
h = w = self.cfg.model_cfg.feature_size
if pseudo_labels.shape[-1] != h:
    pseudo_labels = F.interpolate(pseudo_labels, size=(h, w), mode='bilinear')
```

### 4.4 Config Change

In `configs/uscod/UCOD-DPL_dinov2.py`:

```python
dataset_cfg=dict(
    cache_dir='./datasets/cache/refined_pseudo_labels',  # point to refined PNGs
    ...
)
```

---

## 5. Module 3: Experiment Suite

### 5.1 Main Experiment — Table 1

**Datasets**: COD10K, CAMO, CHAMELEON, NC4K
**Metrics**: S-measure (S_m), E-measure (E_m), weighted F-measure (F_β^w), MAE

**Baseline comparison:**

| Method | Type | COD10K | CAMO | CHAMELEON | NC4K |
|--------|------|--------|------|-----------|------|
| UGTR | Unsup | ref | ref | ref | ref |
| SEARCH | Unsup | ref | ref | ref | ref |
| UCOD-DPL (original) | Unsup | reproduce | reproduce | reproduce | reproduce |
| UCOD-DPL + SAM2 refine (Ours) | Unsup | **ours** | **ours** | **ours** | **ours** |
| DualUCOD | Unsup (SOTA) | paper | paper | paper | paper |

### 5.2 Ablation Study — Table 2

**Dataset**: COD10K. Each row adds one mechanism:

| Row | Configuration | S_m | E_m | F_β^w | MAE |
|-----|---------------|-----|-----|-------|-----|
| 1 | UCOD-DPL baseline (16x16 pkl pseudo-label) | - | - | - | - |
| 2 | + Naive SAM2 (direct mask prompt, no gating) | - | - | - | - |
| 3 | + Adaptive Prompt + Hierarchical Neg Sampling | - | - | - | - |
| 4 | + Truncated Multi-Mask Selection | - | - | - | - |
| 5 | + Edge-Aware Confidence Gating | - | - | - | - |
| 6 | Full model (including Local-SAM) | - | - | - | - |

**Purpose**: Prove each of the 4 mechanisms independently contributes, and naive SAM2 without gating can hurt.

### 5.3 Visualization — Figure 4

**Layout**: 5 columns x 3-4 rows (one per representative case)

| Column | Content |
|--------|---------|
| (a) Original Image | RGB image |
| (b) Coarse Pseudo-label | 16x16 → upsample, blurred boundary |
| (c) Raw SAM2 output | Without gating, may show over-segmentation |
| (d) Ours (refined) | After full 4-stage pipeline |
| (e) Ground Truth | For reference |

**Case selection**:
- Row 1: Simple large target (shows SAM2 handles easy case)
- Row 2: Small target (shows Local-SAM benefit)
- Row 3: Low-contrast target (shows gating prevents over-segmentation)
- Row 4: Multi-object scene (shows robustness)

---

## 6. File Structure

```
UCOD-DPL-main/
├── scripts/
│   └── offline_sam2_refine.py          # SAM2 refinement pipeline (NEW)
├── experiments/
│   ├── run_main_experiment.sh          # Train + evaluate all datasets (NEW)
│   ├── run_ablation.sh                 # Ablation study variants (NEW)
│   └── plot_figures.py                 # Generate Figure 4 (NEW)
├── data/datasets/
│   └── cache_manager.py                # Modified: PNG read support
├── engine/runner/
│   └── loop_UCOD_DPL.py                # Modified: upsample guard
├── configs/uscod/
│   └── UCOD-DPL_dinov2.py              # Modified: cache_dir
└── datasets/cache/
    └── refined_pseudo_labels/          # Output directory (NEW, populated by script)
```

---

## 7. Constraints and Assumptions

### 7.1 Hardware
- **GPU**: 8GB VRAM (tested with sam2.1_hiera_tiny, ~4GB during inference)
- **Time**: ~40 minutes for 4040-image refinement (single GPU, ~0.6s/image)
- **Disk**: ~50MB for 4040 PNG masks at original resolution

### 7.2 Dependencies
- `sam2` (Meta official package, pip install)
- sam2.1_hiera_tiny checkpoint (~80MB download)
- Existing: torch, PIL, numpy, opencv-python

### 7.3 Key Parameters (Fixed)

| Parameter | Value | Stage |
|-----------|-------|-------|
| α (SAM confidence weight) | 0.3 | 4 |
| β (coarse agreement weight) | 0.4 | 4 |
| γ (edge alignment weight) | 0.3 | 4 |
| IoU lower threshold | 0.25 | 3 |
| IoU upper threshold | 0.90 | 3 |
| S lower threshold | 0.2 | 4 |
| S upper threshold | 0.8 | 4 |
| Small target area threshold | H*W*0.01 | 2 |
| Expand ratio base | 0.30 | 1 |
| Expand ratio coefficient | 0.15 | 1 |
| Positive points (N) | 5 | 1 |
| SAM2 image size | original resolution | 2 |

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SAM2 crashes on edge cases (all-black images, extreme low contrast) | Low | Low | S-score gating catches these, falls back to coarse label |
| LANCZOS downsampling from original→68x68 loses boundary precision | Medium | Medium | 68x68 is 18x more pixels than 16x16; verify with boundary F-measure |
| sam2.1_hiera_tiny too weak for complex camouflage | Low | Medium | Ablation will reveal; can upgrade to small/base if needed |
| PNG reading too slow in dataloader | Low | Low | PNG decode + resize is <1ms; 4040 images cached in RAM by dataloader |
| Double-upsample (LANCZOS to 68 + bilinear in train loop) | Low | Low | Guard condition prevents redundant upsample |

---

## 9. Success Criteria

1. **Primary**: S-measure on COD10K exceeds UCOD-DPL baseline by >= 0.5 points
2. **Boundary**: Boundary F-measure improvement >= 2 points (the main claim)
3. **Ablation**: Each of the 4 gating mechanisms contributes positively when added
4. **Visual**: Figure 4 clearly shows boundary improvement over 16x16 coarse label
5. **Training stability**: Loss curve no worse than original UCOD-DPL (no divergence from bad SAM2 masks)
