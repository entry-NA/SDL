"""offline_sam2_refine.py — SAM2-Guided Pseudo-Label Boundary Refinement."""
import os, sys, json, pickle, gc
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CFG = {
    'alpha': 0.3, 'beta': 0.4, 'gamma': 50.0,
    'iou_lower': 0.25, 'iou_upper': 0.90,
    's_lower': 0.20, 's_upper': 999.0,
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
    sigma = CFG['canny_sigma']
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    edges = cv2.Canny(blurred, 50, 150)
    bp = boundary.sum()
    return float((boundary * edges).sum()) / bp / 255.0 if bp > 0 else 0.0


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


class SAM2Wrapper:
    """Loads sam2.1_hiera_tiny, provides predict() and predict_crop()."""

    def __init__(self, device='cuda'):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from huggingface_hub import hf_hub_download
        self.device = device
        ckpt_path = hf_hub_download(
            repo_id='facebook/sam2.1-hiera-tiny',
            filename='sam2.1_hiera_tiny.pt'
        )
        self.model = build_sam2(
            config_file='configs/sam2.1/sam2.1_hiera_t.yaml',
            ckpt_path=ckpt_path,
            device=device
        )
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
            mu8 = (m * 255).astype(np.uint8)
            mb = cv2.resize(mu8, (ow, oh), interpolation=cv2.INTER_LINEAR)
            fm = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
            fm[y:y+h, x:x+w] = (mb >= 128)
            masks_full.append(fm)
        return masks_full, [float(s) for s in scores]


# ==============================================================================
# Stage 1: Adaptive Prompt
# ==============================================================================
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


# ==============================================================================
# Stage 2: SAM2 Inference
# ==============================================================================
def stage2_sam2_inference(wrapper, image, coarse_mask, bbox_exp, pos, neg):
    """Returns (3 masks, 3 iou_scores)."""
    area = (coarse_mask >= 128).sum()
    img_area = image.shape[0] * image.shape[1]
    if area < img_area * CFG['small_area_ratio']:
        return wrapper.predict_crop(image, bbox_exp, pos, neg)
    return wrapper.predict(image, pos, neg, bbox_exp)


# ==============================================================================
# Stage 3: Truncated Multi-Mask Selection
# ==============================================================================
def stage3_select_mask(masks, iou_scores, coarse_mask, image):
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
            return masks[i]  # exemption: coarse label reliable, return this mask directly
        edge_align = compute_edge_align(m_bool, image)
        score = 0.3 * iou_scores[i] + 0.4 * iou_val + 0.3 * edge_align
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx < 0:
        return coarse_mask
    return masks[best_idx]


# ==============================================================================
# Stage 4: Edge-Aware Confidence Gating
# ==============================================================================
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


# ==============================================================================
# Main Pipeline Entry Point
# ==============================================================================
def parse_args():
    """Parse CLI arguments while preserving the historical path defaults."""
    args = {
        'flags': None,
        'output_dir': None,
        'mode': 'both',
        'dataset_dir': None,
        'coarse_dir': None,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--flags' and i + 1 < len(argv):
            args['flags'] = argv[i + 1]
            i += 2
        elif argv[i] == '--output_dir' and i + 1 < len(argv):
            args['output_dir'] = argv[i + 1]
            i += 2
        elif argv[i] == '--mode' and i + 1 < len(argv):
            args['mode'] = argv[i + 1]
            i += 2
        elif argv[i] == '--dataset_dir' and i + 1 < len(argv):
            args['dataset_dir'] = argv[i + 1]
            i += 2
        elif argv[i] == '--coarse_dir' and i + 1 < len(argv):
            args['coarse_dir'] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


def _apply_naive_cfg_overrides():
    """Apply the exact same overrides used for v2 naive SAM2 ablation.
    Must match experiments/ablation_flags_naive_sam2.json exactly."""
    CFG['n_pos_points'] = 1
    CFG['n_neg_safe'] = 0
    CFG['n_neg_cautious'] = 0
    CFG['expand_base'] = 0.0
    CFG['expand_coeff'] = 0.0
    CFG['iou_lower'] = 0.0
    CFG['iou_upper'] = 1.01
    CFG['alpha'] = 0.0
    CFG['beta'] = 1.0
    CFG['gamma'] = 0.0
    CFG['small_area_ratio'] = 0.0


def main():
    cli_args = parse_args()
    mode = cli_args['mode']

    # Save original CFG BEFORE any overrides
    full_cfg = dict(CFG)

    # Apply ablation flags if provided (backward compatible)
    if cli_args['flags']:
        with open(cli_args['flags'], 'r') as f:
            ablation_flags = json.load(f)
        if ablation_flags.get('use_original_pkl'):
            print("Ablation: using original pkl — skipping refinement")
            return
        if not ablation_flags.get('adaptive_prompt', True):
            CFG['n_pos_points'] = 1
            CFG['n_neg_safe'] = 0
            CFG['n_neg_cautious'] = 0
            CFG['expand_base'] = 0.0
            CFG['expand_coeff'] = 0.0
        if not ablation_flags.get('mask_selection', True):
            CFG['iou_lower'] = 0.0
            CFG['iou_upper'] = 1.01
        if not ablation_flags.get('edge_gating', True):
            CFG['alpha'] = 0.0
            CFG['beta'] = 1.0
            CFG['gamma'] = 0.0
        if not ablation_flags.get('local_sam', True):
            CFG['small_area_ratio'] = 0.0

    if cli_args['output_dir']:
        CFG['output_dir'] = cli_args['output_dir']
    if cli_args['dataset_dir']:
        dataset_dir = Path(cli_args['dataset_dir'])
        CFG['image_dirs'] = [
            str(dataset_dir / 'TR-CAMO' / 'im'),
            str(dataset_dir / 'TR-COD10K' / 'im'),
        ]
    if cli_args['coarse_dir']:
        CFG['coarse_dir'] = cli_args['coarse_dir']

    print("=" * 60)
    print(f"SAM2-Guided Pseudo-Label Refinement (mode={mode})")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    wrapper = SAM2Wrapper(device)
    rng = np.random.RandomState(42)
    image_paths = get_image_paths(CFG['image_dirs'])
    with open(os.path.join(CFG['coarse_dir'], 'index.json'), 'r') as f:
        index_map = json.load(f)
    print(f"Images: {len(image_paths)}, Pseudo-labels: {len(index_map)}")
    assert len(image_paths) == len(index_map), "Count mismatch!"

    # Output directories
    refined_dir = CFG['output_dir']
    naive_dir = os.path.join(os.path.dirname(refined_dir), 'naive_sam2_labels')
    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(naive_dir, exist_ok=True)

    stats = {'total': 0, 'full': 0, 'fallback': 0, 'fusion': 0, 'local': 0, 'err': 0,
             'naive': 0, 'empty_coarse': 0}
    per_image_stats = {}

    for idx in tqdm(range(len(image_paths)), desc="Refining"):
        try:
            img_path = image_paths[idx]
            image = cv2.imread(img_path)
            if image is None: stats['err'] += 1; continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_h, im_w = image.shape[:2]
            coarse = load_coarse_label(
                os.path.join(CFG['coarse_dir'], index_map[str(idx)]), (im_h, im_w))

            # --- Branch: Naive SAM2 (only if mode is 'naive' or 'both') ---
            if mode in ('naive', 'both'):
                _apply_naive_cfg_overrides()
                bbox_n, pos_n, neg_n = stage1_adaptive_prompt(coarse, im_h, im_w, rng)
                if len(pos_n) > 0:
                    masks_n, scores_n = stage2_sam2_inference(wrapper, image, coarse,
                                                              bbox_n, pos_n, neg_n)
                    selected_n = stage3_select_mask(masks_n, scores_n, coarse, image)
                    final_n = stage4_confidence_gating(selected_n, coarse, image, max(scores_n))
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_refined_mask(final_n, os.path.join(naive_dir, f"{img_name}.png"))
                    stats['naive'] += 1
                # Restore full CFG
                CFG.update(full_cfg)

            # --- Branch: Full pipeline (only if mode is 'full' or 'both') ---
            if mode in ('full', 'both'):
                bbox, pos, neg = stage1_adaptive_prompt(coarse, im_h, im_w, rng)
                if len(pos) == 0:
                    stats['empty_coarse'] += 1
                masks, scores = stage2_sam2_inference(wrapper, image, coarse, bbox, pos, neg)
                selected = stage3_select_mask(masks, scores, coarse, image)
                final = stage4_confidence_gating(selected, coarse, image, max(scores))
                S = (CFG['alpha'] * max(scores) + CFG['beta'] * compute_iou(selected, coarse) +
                     CFG['gamma'] * compute_edge_align(selected, image))
                if S < CFG['s_lower']: gate = 'fallback'
                elif S > CFG['s_upper']: gate = 'full'
                else: gate = 'fusion'
                if gate == 'fallback': stats['fallback'] += 1
                elif gate == 'full': stats['full'] += 1
                else: stats['fusion'] += 1
                local_triggered = (coarse >= 128).sum() < im_h * im_w * CFG['small_area_ratio']
                if local_triggered: stats['local'] += 1

                img_name = os.path.splitext(os.path.basename(img_path))[0]
                save_refined_mask(final, os.path.join(refined_dir, f"{img_name}.png"))
                stats['total'] += 1

                # Collect per-image stats
                coarse_area = (coarse >= 128).sum()
                fg_coarse = coarse >= 128
                cc = cv2.connectedComponentsWithStats(fg_coarse.astype(np.uint8), connectivity=8)
                if cc[0] > 1:
                    largest_area_idx = np.argmax(cc[2][1:, cv2.CC_STAT_AREA]) + 1
                    centroid = tuple(cc[3][largest_area_idx])
                else:
                    centroid = None
                per_image_stats[img_name] = {
                    'S_score': round(float(S), 6),
                    'IoU_ori': round(float(compute_iou(selected, coarse)), 6),
                    'EdgeAlign': round(float(compute_edge_align(selected, image)), 6),
                    'IoU_pred': round(float(max(scores)), 6),
                    'gate_decision': str(gate),
                    'LocalSAM_triggered': bool(local_triggered),
                    'coarse_area_ratio': round(float(coarse_area) / (im_h * im_w), 6),
                    'coarse_centroid': [float(c) for c in centroid] if centroid else None,
                    'selected_mask_idx': int(np.argmax(scores)),
                    'image_shape': [int(im_h), int(im_w)],
                }
        except Exception as e:
            print(f"\n[ERROR] idx={idx}: {e}")
            import traceback; traceback.print_exc()
            stats['err'] += 1
        finally:
            # Prevent GPU/CPU memory fragmentation buildup
            if idx % 500 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save per-image stats
    stats_path = os.path.join(refined_dir, 'per_image_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(per_image_stats, f, indent=2)
    print(f"Per-image stats saved to {stats_path}")

    total = stats['total']
    print(f"\nDone. mode={mode}")
    if mode in ('full', 'both'):
        print(f"  Full pipeline: {total} refined, Full:{stats['full']} ({100*stats['full']/max(total,1):.1f}%) "
              f"Fusion:{stats['fusion']} ({100*stats['fusion']/max(total,1):.1f}%) "
              f"Fallback:{stats['fallback']} ({100*stats['fallback']/max(total,1):.1f}%) "
              f"LocalSAM:{stats['local']} Empty:{stats['empty_coarse']}")
    if mode in ('naive', 'both'):
        print(f"  Naive pipeline: {stats['naive']} masks")
    print(f"  Errors: {stats['err']}")


if __name__ == '__main__':
    main()
