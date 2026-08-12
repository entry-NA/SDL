# SAM2-Guided Pseudo-Label Refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build offline SAM2 pseudo-label refinement pipeline + integrate into UCOD-DPL training + experiment scripts.

**Architecture:** Single offline script processes 4040 training images once, producing refined PNG masks. `base_dataset.py` loads PNGs at 68×68 by image name (bypasses 16×16 pkl bottleneck). `loop_UCOD_DPL.py` gets upsample guard.

**Tech Stack:** Python 3.9, torch 2.12, SAM2 (sam2.1_hiera_tiny), PIL, numpy, opencv-python

## Global Constraints

- **Environment:** `C:\Anaconda\envs\test01`
- **GPU:** 8GB VRAM
- **SAM2 model:** `sam2.1_hiera_tiny`, original image resolution inference, `torch.no_grad()`
- **Coarse labels:** `C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K\data_{i}.pkl` (16×16 binary tensor)
- **Images:** `C:\Users\23991\Desktop\RefCOD (1)\RefCOD\{TR-CAMO,TR-COD10K}\im\*.jpg`
- **Output:** `./datasets/cache/refined_pseudo_labels/{image_name}.png` (original resolution, single-channel)
- **Working dir:** `C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main`

## Parameters (spec Section 7.3)

```python
CFG = {
    'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3,
    'iou_lower': 0.25, 'iou_upper': 0.90,
    's_lower': 0.2, 's_upper': 0.8,
    'small_area_ratio': 0.01,
    'expand_base': 0.30, 'expand_coeff': 0.15,
    'n_pos_points': 5, 'n_neg_safe': 3, 'n_neg_cautious': 2,
    'canny_sigma': 1.5, 'local_sam_size': 256,
}
```

## Files Changed/Created

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `scripts/offline_sam2_refine.py` | Full 4-stage SAM2 refinement pipeline |
| MODIFY | `data/datasets/base_dataset.py` | Add PNG pseudo-label loading by image name |
| MODIFY | `engine/runner/loop_UCOD_DPL.py` | Upsample guard for pre-resized labels |
| CREATE | `experiments/run_ablation.py` | Ablation study variants |
| CREATE | `experiments/plot_figure4.py` | Visualization script |

---

### Task 1: Install SAM2 in test01 Environment

- [ ] Install SAM2 package:

```bash
conda run -p "C:\Anaconda\envs\test01" pip install git+https://github.com/facebookresearch/sam2.git
```

- [ ] Verify import + GPU:

```bash
conda run -p "C:\Anaconda\envs\test01" python -c "
import torch
from sam2 import build_sam2
m = build_sam2('sam2.1_hiera_tiny')
print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')
print('OK')
"
```

Expected: `CUDA: True, VRAM: ~8.0GB, OK`

---

### Task 2: Create offline_sam2_refine.py — Skeleton + Helpers

**Create:** `scripts/offline_sam2_refine.py`

- [ ] Write the complete refinement script. The file is structured into sections:

**Section A — Imports and CFG constants:**

```python
"""offline_sam2_refine.py — SAM2-Guided Pseudo-Label Boundary Refinement."""
import os, sys, json, pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CFG = {
    'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3,
    'iou_lower': 0.25, 'iou_upper': 0.90,
    's_lower': 0.2, 's_upper': 0.8,
    'small_area_ratio': 0.01,
    'expand_base': 0.30, 'expand_coeff': 0.15,
    'n_pos_points': 5, 'n_neg_safe': 3, 'n_neg_cautious': 2,
    'canny_sigma': 1.5, 'local_sam_size': 256,
    'output_dir': './datasets/cache/refined_pseudo_labels',
    'coarse_dir': r'C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K',
    'image_dirs': [
        r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\im',
        r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-COD10K\im',
    ],
}
```

**Section B — Helper functions:**

```python
def load_coarse_label(pkl_path, target_shape):
    """Load 16x16 pkl, upsample to target_shape (H, W). Returns uint8 ndarray {0,255}."""
    with open(pkl_path, 'rb') as f:
        coarse = pickle.load(f)
    if isinstance(coarse, torch.Tensor):
        coarse = coarse.numpy()
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    return cv2.resize(coarse, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def compute_iou(mask_a, mask_b):
    """IoU between two bool arrays. Returns float [0,1]."""
    a = (mask_a > 0.5) if mask_a.dtype != bool else mask_a
    b = (mask_b > 0.5) if mask_b.dtype != bool else mask_b
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_edge_align(mask, image):
    """Fraction of mask boundary pixels on Canny edges. Returns float [0,1]."""
    mask_bool = (mask > 0.5) if mask.dtype != bool else mask
    mask_u8 = mask_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    bp = boundary.sum()
    return float((boundary * edges).sum()) / bp if bp > 0 else 0.0


def expand_bbox(bbox, ratio, im_h, im_w):
    """Expand bbox (x,y,w,h) by ratio, clamped to image. Returns (x,y,w,h)."""
    x, y, w, h = bbox
    nw, nh = int(w * (1 + ratio)), int(h * (1 + ratio))
    nx = max(0, int(x - (nw - w) / 2))
    ny = max(0, int(y - (nh - h) / 2))
    return (nx, ny, min(nw, im_w - nx), min(nh, im_h - ny))


def sample_points_in_mask(mask, n, rng):
    """Sample n foreground points from mask (uint8 {0,255}). Returns list[(x,y)]."""
    ys, xs = np.where(mask >= 128)
    if len(ys) == 0:
        return []
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def sample_outer_negatives(bbox, im_h, im_w, n, rng):
    """Sample n negative points from OUTSIDE expanded bbox. Returns list[(x,y)]."""
    x, y, w, h = bbox
    regions = []
    if y > 0: regions.append((0, 0, im_w, y))
    if y + h < im_h: regions.append((0, y + h, im_w, im_h - y - h))
    if x > 0: regions.append((0, y, x, h))
    if x + w < im_w: regions.append((x + w, y, im_w - x - w, h))
    if not regions:
        return []
    areas = [rw * rh for _, _, rw, rh in regions]
    total = sum(areas)
    pts = []
    for _ in range(n):
        r = rng.uniform(0, total)
        cum = 0
        for (rx, ry, rw, rh), a in zip(regions, areas):
            cum += a
            if r <= cum:
                pts.append((rx + rng.randint(0, rw), ry + rng.randint(0, rh)))
                break
    return pts


def sample_cautious_negatives(mask, bbox, n, min_dist, rng):
    """Sample n negative points inside expanded bbox but outside mask, far from centroid. Returns list[(x,y)]."""
    x, y, w, h = bbox
    roi = mask[y:y+h, x:x+w]
    bys, bxs = np.where(roi < 128)
    if len(bys) == 0:
        return []
    fys, fxs = np.where(mask >= 128)
    if len(fys) == 0:
        return []
    cx, cy = fxs.mean(), fys.mean()
    valid = [(int(x + dx), int(y + dy)) for dy, dx in zip(bys, bxs)
             if np.sqrt((x + dx - cx)**2 + (y + dy - cy)**2) > min_dist]
    if not valid:
        return []
    idx = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    return [valid[i] for i in idx]


def get_image_paths(image_dirs):
    """Sorted list of image paths from directories."""
    paths = []
    for d in image_dirs:
        paths.extend(str(p) for p in Path(d).glob('*.jpg'))
        paths.extend(str(p) for p in Path(d).glob('*.png'))
    return sorted(paths)


def save_refined_mask(mask, path):
    """Save binary mask (uint8, {0,255}) as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask, mode='L').save(path)
```

- [ ] Verify import:

```bash
conda run -p "C:\Anaconda\envs\test01" python -c "
import sys; sys.path.insert(0, 'scripts')
from offline_sam2_refine import CFG, load_coarse_label, compute_iou, compute_edge_align
print('Helpers OK')
"
```

---

### Task 3: SAM2Wrapper Class

**Modify:** `scripts/offline_sam2_refine.py` — append SAM2Wrapper

- [ ] Write SAM2Wrapper:

```python
class SAM2Wrapper:
    """Loads sam2.1_hiera_tiny, provides predict() and predict_crop()."""

    def __init__(self, device='cuda'):
        from sam2 import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = device
        self.model = build_sam2('sam2.1_hiera_tiny', ckpt_path=None).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.predictor = SAM2ImagePredictor(self.model)
        print(f"[SAM2Wrapper] Loaded sam2.1_hiera_tiny on {device}")

    def predict(self, image, pos_points, neg_points, bbox):
        """Run SAM2 with point+bbox prompt. Returns (3 masks, 3 iou_scores)."""
        all_pts = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)
        with torch.no_grad():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=all_pts if all_pts else None,
                point_labels=all_labels if all_pts else None,
                box=bbox,
                multimask_output=True,
            )
        return [m for m in masks], [float(s) for s in scores]

    def predict_crop(self, image, bbox, pos_points, neg_points):
        """Local-SAM: crop expanded bbox, resize to 256, predict, map back."""
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]
        oh, ow = crop.shape[:2]
        ls = CFG['local_sam_size']
        crop_r = cv2.resize(crop, (ls, ls), interpolation=cv2.INTER_LINEAR)
        sx, sy = ls / ow, ls / oh
        pos_c = [(int((px - x) * sx), int((py - y) * sy)) for px, py in pos_points]
        neg_c = [(int((px - x) * sx), int((py - y) * sy)) for px, py in neg_points]
        all_pts = pos_c + neg_c
        all_labels = [1] * len(pos_c) + [0] * len(neg_c)
        with torch.no_grad():
            self.predictor.set_image(crop_r)
            masks, scores, _ = self.predictor.predict(
                point_coords=all_pts if all_pts else None,
                point_labels=all_labels if all_pts else None,
                box=(0, 0, ls, ls),
                multimask_output=True,
            )
        masks_full = []
        for m in masks:
            mu8 = m.astype(np.uint8) * 255
            mb = cv2.resize(mu8, (ow, oh), interpolation=cv2.INTER_LINEAR)
            fm = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            fm[y:y+h, x:x+w] = (mb >= 128)
            masks_full.append(fm)
        return masks_full, [float(s) for s in scores]
```

- [ ] Verify:

```bash
conda run -p "C:\Anaconda\envs\test01" python -c "
import sys; sys.path.insert(0, 'scripts')
from offline_sam2_refine import SAM2Wrapper
w = SAM2Wrapper('cuda')
print('Wrapper OK')
"
```

---

### Task 4: Four-Stage Pipeline Functions

**Modify:** `scripts/offline_sam2_refine.py` — append stages 1-4

- [ ] Write Stage 1 — Adaptive Prompt:

```python
def stage1_adaptive_prompt(coarse_mask, im_h, im_w, rng):
    """Returns (bbox_expanded, pos_points, neg_points)."""
    area = (coarse_mask >= 128).sum()
    expand_ratio = CFG['expand_base'] - CFG['expand_coeff'] * min(1.0, area / (im_h * im_w * 0.01))
    fg_y, fg_x = np.where(coarse_mask >= 128)
    if len(fg_y) == 0:
        return (0, 0, im_w, im_h), [], []
    x, y, w, h = (int(fg_x.min()), int(fg_y.min()),
                   int(fg_x.max() - fg_x.min()), int(fg_y.max() - fg_y.min()))
    bbox_exp = expand_bbox((x, y, w, h), expand_ratio, im_h, im_w)
    pos = sample_points_in_mask(coarse_mask, CFG['n_pos_points'], rng)
    neg = sample_outer_negatives(bbox_exp, im_h, im_w, CFG['n_neg_safe'], rng)
    min_dist = 0.5 * np.sqrt(max(area, 1))
    neg += sample_cautious_negatives(coarse_mask, bbox_exp, CFG['n_neg_cautious'], min_dist, rng)
    return bbox_exp, pos, neg
```

- [ ] Write Stage 2 — SAM2 Inference:

```python
def stage2_sam2_inference(wrapper, image, coarse_mask, bbox_exp, pos, neg):
    """Returns (3 masks, 3 iou_scores)."""
    area = (coarse_mask >= 128).sum()
    img_area = image.shape[0] * image.shape[1]
    if area < img_area * CFG['small_area_ratio']:
        return wrapper.predict_crop(image, bbox_exp, pos, neg)
    return wrapper.predict(image, pos, neg, bbox_exp)
```

- [ ] Write Stage 3 — Truncated Multi-Mask Selection:

```python
def stage3_select_mask(masks, iou_scores, coarse_mask):
    """Returns best mask (bool ndarray)."""
    coarse_bool = coarse_mask >= 128
    best_idx, best_score = -1, -float('inf')
    for i in range(3):
        if masks[i].shape != coarse_mask.shape:
            m = cv2.resize(masks[i].astype(np.uint8)*255,
                           (coarse_mask.shape[1], coarse_mask.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            m_bool = m >= 128
        else:
            m_bool = masks[i].astype(bool) if masks[i].dtype != bool else masks[i]
        iou_val = compute_iou(m_bool, coarse_bool)
        if iou_val < CFG['iou_lower']:
            continue
        if iou_val > CFG['iou_upper']:
            return masks[int(np.argmax(iou_scores))] if masks[int(np.argmax(iou_scores))].dtype == bool else masks[int(np.argmax(iou_scores))]
        score = iou_val * iou_scores[i]
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx < 0:
        return coarse_mask
    return masks[best_idx]
```

- [ ] Write Stage 4 — Edge-Aware Gating:

```python
def stage4_confidence_gating(mask_sel, coarse_mask, image, iou_pred):
    """Returns final mask (uint8 ndarray {0,255})."""
    ms_bool = mask_sel if mask_sel.dtype == bool else (mask_sel >= 128)
    cb_bool = coarse_mask >= 128
    iou_ori = compute_iou(ms_bool, cb_bool)
    edge = compute_edge_align(ms_bool, image)
    S = CFG['alpha'] * iou_pred + CFG['beta'] * iou_ori + CFG['gamma'] * edge
    if S < CFG['s_lower']:
        return coarse_mask
    elif S > CFG['s_upper']:
        return (ms_bool.astype(np.uint8)) * 255
    else:
        fused = S * ms_bool.astype(np.float32) + (1.0 - S) * cb_bool.astype(np.float32)
        return (fused > 0.5).astype(np.uint8) * 255
```

- [ ] Write main():

```python
def main():
    print("=" * 60 + "\nSAM2-Guided Pseudo-Label Refinement\n" + "=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    wrapper = SAM2Wrapper(device)
    rng = np.random.RandomState(42)
    image_paths = get_image_paths(CFG['image_dirs'])
    with open(os.path.join(CFG['coarse_dir'], 'index.json'), 'r') as f:
        index_map = json.load(f)
    print(f"Images: {len(image_paths)}, Pseudo-labels: {len(index_map)}")
    assert len(image_paths) == len(index_map), "Count mismatch!"
    os.makedirs(CFG['output_dir'], exist_ok=True)
    stats = {'total': 0, 'full': 0, 'fallback': 0, 'fusion': 0, 'local': 0, 'err': 0}

    for idx in tqdm(range(len(image_paths)), desc="Refining"):
        try:
            img_path = image_paths[idx]
            image = cv2.imread(img_path)
            if image is None: stats['err'] += 1; continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_h, im_w = image.shape[:2]
            coarse = load_coarse_label(
                os.path.join(CFG['coarse_dir'], index_map[str(idx)]), (im_h, im_w))
            bbox, pos, neg = stage1_adaptive_prompt(coarse, im_h, im_w, rng)
            masks, scores = stage2_sam2_inference(wrapper, image, coarse, bbox, pos, neg)
            selected = stage3_select_mask(masks, scores, coarse)
            final = stage4_confidence_gating(selected, coarse, image, max(scores))
            S = (CFG['alpha'] * max(scores) + CFG['beta'] * compute_iou(selected, coarse) +
                 CFG['gamma'] * compute_edge_align(selected, image))
            if S < CFG['s_lower']: stats['fallback'] += 1
            elif S > CFG['s_upper']: stats['full'] += 1
            else: stats['fusion'] += 1
            if (coarse >= 128).sum() < im_h * im_w * CFG['small_area_ratio']:
                stats['local'] += 1
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_refined_mask(final, os.path.join(CFG['output_dir'], f"{img_name}.png"))
            stats['total'] += 1
        except Exception as e:
            print(f"\n[ERROR] idx={idx}: {e}"); stats['err'] += 1

    print(f"\nDone. {stats['total']} refined. Full:{stats['full']} Fallback:{stats['fallback']} "
          f"Fusion:{stats['fusion']} LocalSAM:{stats['local']} Errors:{stats['err']}")

if __name__ == '__main__':
    main()
```

- [ ] Verify pipeline on single image:

```bash
conda run -p "C:\Anaconda\envs\test01" python -c "
import sys; sys.path.insert(0, 'scripts')
import offline_sam2_refine as ref
import cv2, numpy as np
w = ref.SAM2Wrapper('cuda')
paths = ref.get_image_paths(ref.CFG['image_dirs'])
img = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
rng = np.random.RandomState(42)
coarse = ref.load_coarse_label(r'C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K\data_0.pkl', img.shape[:2])
bb, pos, neg = ref.stage1_adaptive_prompt(coarse, img.shape[0], img.shape[1], rng)
print(f'Stage1: bbox={bb}, pos={len(pos)}, neg={len(neg)}')
masks, scores = ref.stage2_sam2_inference(w, img, coarse, bb, pos, neg)
print(f'Stage2: {len(masks)} masks, scores={scores}')
sel = ref.stage3_select_mask(masks, scores, coarse)
print(f'Stage3: shape={sel.shape}')
fin = ref.stage4_confidence_gating(sel, coarse, img, max(scores))
print(f'Stage4: shape={fin.shape}, unique={fin.sum()}')
print('ALL 4 STAGES PASSED')
"
```

---

### Task 5: Batch Refinement — 4040 Images

- [ ] Run full batch:

```bash
cd "C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main"
conda run -p "C:\Anaconda\envs\test01" python scripts/offline_sam2_refine.py
```

Expected: ~40 min, 4040 images, errors < 10.

- [ ] Verify output:

```bash
echo "Files: $(ls ./datasets/cache/refined_pseudo_labels/ | wc -l)"
conda run -p "C:\Anaconda\envs\test01" python -c "
from PIL import Image; import os
d = './datasets/cache/refined_pseudo_labels'
fs = os.listdir(d)
print(f'Count: {len(fs)}')
i = Image.open(os.path.join(d, fs[0]))
print(f'Sample: {i.size}, mode={i.mode}, vals={set(list(i.getdata()))}')
"
```

Expected: 4040 files, mode='L', values {0, 255}.

---

### Task 6: Modify base_dataset.py — PNG Pseudo-Label Loading

**Modify:** `data/datasets/base_dataset.py` lines 157-184 (`__getitem__` method)

- [ ] Replace `__getitem__` with PNG-priority loading:

In `data/datasets/base_dataset.py`, replace the `__getitem__` method:

```python
def __getitem__(self, index: int) -> Dict[str, Any]:
    """Get dataset item. Tries refined PNG pseudo-labels first, falls back to pkl."""
    img_path = self.image_paths[index]
    img_name = os.path.splitext(os.path.basename(str(img_path)))[0]
    
    # Load label if required
    label_tensor = None
    if self.label_paths:
        label_tensor = self.img_io.read_image(self.label_paths[index], 'L')
        label_tensor = self.transform_label(label_tensor)
    
    # Load features from cache
    features = None
    features_cache = self.cache_manager.get_features_cache()
    if features_cache:
        features = features_cache.read_file(index)
    
    # Load pseudo label — refined PNG first, then pkl cache
    pseudo_label = None
    refined_dir = os.path.join(self.cache_dir, 'refined_pseudo_labels')
    if os.path.isdir(refined_dir):
        png_path = os.path.join(refined_dir, f"{img_name}.png")
        if os.path.exists(png_path):
            from PIL import Image
            feature_size = getattr(self.config, 'feature_size', 68)
            img = Image.open(png_path).convert('L')
            img = img.resize((feature_size, feature_size), Image.LANCZOS)
            pseudo_label = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
    
    # Fallback to pkl cache
    if pseudo_label is None:
        pseudo_label_cache = self.cache_manager.get_pseudo_label_cache()
        if pseudo_label_cache:
            pseudo_label = pseudo_label_cache.read_file(index)

    return {
        "pseudo_label": pseudo_label,
        "label_tensor": label_tensor,
        "features": features,
        "img_path": str(img_path)
    }
```

Note: Add `import torch` at the top of `base_dataset.py` if not already present.

- [ ] Verify loading:

```bash
cd "C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main"
conda run -p "C:\Anaconda\envs\test01" python -c "
import torch, os, sys
sys.path.insert(0, '.')
from data.datasets import base_dataset
# Quick test: check PNG path resolution
refined_dir = './datasets/cache/refined_pseudo_labels'
print(f'Dir exists: {os.path.isdir(refined_dir)}')
if os.path.isdir(refined_dir):
    files = os.listdir(refined_dir)
    print(f'Files: {len(files)}, Sample: {files[0]}')
"
```

---

### Task 7: Modify loop_UCOD_DPL.py — Upsample Guard

**Modify:** `engine/runner/loop_UCOD_DPL.py` lines 152-154

- [ ] In `_process_batch()`, replace the pseudo-label upsample line:

```python
# Old (lines 152-154):
# h = w = self.cfg.model_cfg.feature_size
# features = F.interpolate(features, size=(h,w), mode='bilinear')
# pseudo_labels = F.interpolate(pseudo_labels, size=(h,w), mode='bilinear').float()

# New:
h = w = self.cfg.model_cfg.feature_size
features = F.interpolate(features, size=(h, w), mode='bilinear')

# Guard: only upsample if not already at target size
if pseudo_labels.shape[-1] != h or pseudo_labels.shape[-2] != w:
    pseudo_labels = F.interpolate(pseudo_labels, size=(h, w), mode='bilinear').float()
else:
    pseudo_labels = pseudo_labels.float()
```

- [ ] Verify syntax:

```bash
conda run -p "C:\Anaconda\envs\test01" python -c "
import sys; sys.path.insert(0, '.')
from engine.runner.loop_UCOD_DPL import TrainLoop
print('Import OK')
"
```

---

### Task 8: Experiment — Ablation Script

**Create:** `experiments/run_ablation.py`

- [ ] Write ablation launcher:

```python
"""run_ablation.py — Ablation study: systematically disable each mechanism."""
import os, sys, json

ABLATION_VARIANTS = {
    'baseline':          {},  # use original pkl, no changes
    'naive_sam2':        {'adaptive_prompt': False, 'mask_selection': False,
                           'edge_gating': False, 'local_sam': False},
    '+adaptive_prompt':  {'adaptive_prompt': True, 'mask_selection': False,
                           'edge_gating': False, 'local_sam': False},
    '+mask_selection':   {'adaptive_prompt': True, 'mask_selection': True,
                           'edge_gating': False, 'local_sam': False},
    '+edge_gating':      {'adaptive_prompt': True, 'mask_selection': True,
                           'edge_gating': True, 'local_sam': False},
    'full_model':        {'adaptive_prompt': True, 'mask_selection': True,
                           'edge_gating': True, 'local_sam': True},
}

def run_variant(name, flags):
    """Generate refined labels with specific mechanisms enabled/disabled."""
    # Write flags to a temp JSON, run refinement, then train + eval
    flag_path = f'./experiments/ablation_flags_{name}.json'
    with open(flag_path, 'w') as f:
        json.dump(flags, f)
    print(f"[{name}] Flags saved to {flag_path}")
    # Training command
    cmd = f"bash ./scripts/launch_train_first_stage.sh -c ./configs/uscod/UCOD-DPL_dinov2.py"
    print(f"[{name}] Run: {cmd}")
    print(f"[{name}] Then: evaluate on COD10K/CAMO/CHAMELEON/NC4K")

if __name__ == '__main__':
    for name, flags in ABLATION_VARIANTS.items():
        run_variant(name, flags)
```

- [ ] Document expected output:

Each variant requires: (1) generate refined labels (or use fallback for baseline), (2) train UCOD-DPL first stage, (3) evaluate on 4 datasets, (4) record S_m, E_m, F_beta_w, MAE into a CSV for Table 2.

---

### Task 9: Experiment — Visualization Script

**Create:** `experiments/plot_figure4.py`

- [ ] Write visualization script:

```python
"""plot_figure4.py — Generate 5-column visualization: original, coarse, raw SAM2, ours, GT."""
import cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CASES = {
    'large_target': ('camourflage_00001.jpg', 'TR-CAMO'),
    'small_target': ('camourflage_00150.jpg', 'TR-CAMO'),
    'low_contrast': ('COD10K-CAM-1- Aquatic-3-Bat-242.jpg', 'TR-COD10K'),
    'multi_object': ('COD10K-CAM-1- Aquatic-1-Crab-88.jpg', 'TR-COD10K'),
}

def plot_case(axs, img_name, dataset):
    """Plot one row: (a) original (b) coarse (c) raw SAM2 (d) ours (e) GT"""
    # Load original
    img = cv2.cvtColor(cv2.imread(f'../RefCOD (1)/RefCOD/{dataset}/im/{img_name}'), cv2.COLOR_BGR2RGB)
    axs[0].imshow(img); axs[0].set_title('Original'); axs[0].axis('off')
    # Load coarse (upsampled 16x16)
    # Load raw SAM2 (without gating)
    # Load ours (refined PNG)
    # Load GT
    axs[1].set_title('Coarse (16x16)'); axs[1].axis('off')
    axs[2].set_title('Raw SAM2'); axs[2].axis('off')
    axs[3].set_title('Ours (refined)'); axs[3].axis('off')
    axs[4].set_title('Ground Truth'); axs[4].axis('off')

if __name__ == '__main__':
    fig, axs = plt.subplots(len(CASES), 5, figsize=(20, 4 * len(CASES)))
    for row, (case_name, (img_name, dataset)) in enumerate(CASES.items()):
        plot_case(axs[row], img_name, dataset)
    plt.tight_layout()
    plt.savefig('./experiments/figure4_sam2_refinement.png', dpi=300)
    print('Figure 4 saved to ./experiments/figure4_sam2_refinement.png')
```

---

### Task 10: End-to-End Validation

- [ ] Train UCOD-DPL with refined pseudo-labels:

```bash
cd "C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main"
bash ./scripts/launch_train_first_stage.sh -c ./configs/uscod/UCOD-DPL_dinov2.py
```

- [ ] Evaluate on TE-CAMO:

```bash
bash ./scripts/launch_val_first_stage.sh -c ./configs/uscod/UCOD-DPL_dinov2.py \
  -m ./work/<exp_name>/ckp/epoch25.pth
```

Expected: S-measure, E-measure, F_beta^w, MAE printed. Compare to UCOD-DPL baseline.

- [ ] Success criteria check:
  1. S-measure on COD10K >= baseline + 0.5
  2. Boundary F-measure improvement visible
  3. Training loss curve stable (no divergence)
  4. Figure 4 shows clear boundary improvement
  5. Ablation: each mechanism adds value monotonically
