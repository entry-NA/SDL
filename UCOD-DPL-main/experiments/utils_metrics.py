"""Shared utilities for SAM2 refinement analysis scripts."""
import os, sys, json, pickle
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

# Project root for importing engine.utils.metrics
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_EPS = np.spacing(1)

# ---- BF-score (Boundary F-measure with tolerance) ----

def _extract_boundary(mask_bool):
    """Extract boundary pixels from a boolean mask using morphological gradient."""
    mask_u8 = mask_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask_u8, kernel)
    eroded = cv2.erode(mask_u8, kernel)
    return (dilated - eroded) > 0  # bool ndarray


def _boundary_precision_recall(pred_boundary, gt_boundary, width):
    """Compute boundary precision and recall using distance transform.

    Precision: fraction of pred boundary pixels within `width` px of any GT boundary pixel.
    Recall: fraction of GT boundary pixels within `width` px of any pred boundary pixel.

    Uses cv2.distanceTransform for O(N) computation.
    """
    pred_b = pred_boundary.astype(np.uint8)
    gt_b = gt_boundary.astype(np.uint8)

    # Precision: distance from pred boundary to GT boundary
    dt_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, maskSize=3)
    matched_pred = (dt_gt[pred_boundary] <= width).sum()
    total_pred = pred_boundary.sum()
    precision = float(matched_pred / total_pred) if total_pred > 0 else 0.0

    # Recall: distance from GT boundary to pred boundary
    dt_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, maskSize=3)
    matched_gt = (dt_pred[gt_boundary] <= width).sum()
    total_gt = gt_boundary.sum()
    recall = float(matched_gt / total_gt) if total_gt > 0 else 0.0

    return precision, recall


def compute_bfscore(pred, gt, width=3):
    """Compute Boundary F-measure with `width`-pixel tolerance.

    Args:
        pred: uint8 or bool ndarray, binary mask {0,255} or {0,1} or bool
        gt: uint8 or bool ndarray, binary mask {0,255} or {0,1} or bool
        width: tolerance in pixels (default 3)

    Returns:
        dict: {'r_b': recall, 'p_b': precision, 'bf': f1_score}
    """
    pred_bool = (pred > 127) if pred.dtype == np.uint8 else pred.astype(bool)
    gt_bool = (gt > 127) if gt.dtype == np.uint8 else gt.astype(bool)

    pred_boundary = _extract_boundary(pred_bool)
    gt_boundary = _extract_boundary(gt_bool)

    precision, recall = _boundary_precision_recall(pred_boundary, gt_boundary, width)

    if precision + recall > 0:
        bf = 2 * precision * recall / (precision + recall)
    else:
        bf = 0.0

    return {'r_b': recall, 'p_b': precision, 'bf': bf}


# ---- Data Loading ----

COARSE_DIR = r'C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K'
IMAGE_DIRS = [
    r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\im',
    r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-COD10K\im',
]
REFINED_DIR = './datasets/cache/refined_pseudo_labels_backup'
NAIVE_DIR = './datasets/cache/refined_pseudo_labels_naive'
OUTPUT_DIR = './experiments/output'


def _get_image_paths():
    """Sorted list of image paths from directories."""
    paths = []
    for d in IMAGE_DIRS:
        for ext in ('*.jpg', '*.png'):
            paths.extend(str(p) for p in Path(d).glob(ext))
    return sorted(paths)


def _find_gt_path(img_path):
    """Given an image path, find corresponding GT path."""
    img_name = os.path.basename(img_path)
    name_no_ext = os.path.splitext(img_name)[0]
    # GT is always .png
    for im_dir in IMAGE_DIRS:
        gt_dir = im_dir.replace('\\im', '\\gt')
        gt_path = os.path.join(gt_dir, f"{name_no_ext}.png")
        if os.path.exists(gt_path):
            return gt_path
    return None


def build_unified_index():
    """Build unified per-image mapping with all resolved paths.

    Returns:
        list[dict]: each dict has keys:
            img_name, img_path, gt_path, pkl_path, refined_path, naive_path,
            dataset ('TR-CAMO' or 'TR-COD10K')
    """
    image_paths = _get_image_paths()
    with open(os.path.join(COARSE_DIR, 'index.json'), 'r') as f:
        index_map = json.load(f)

    assert len(image_paths) == len(index_map), \
        f"Count mismatch: {len(image_paths)} images vs {len(index_map)} pkls"

    results = []
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        pkl_filename = index_map[str(idx)]
        pkl_path = os.path.join(COARSE_DIR, pkl_filename)
        gt_path = _find_gt_path(img_path)
        refined_path = os.path.join(REFINED_DIR, f"{img_name}.png")
        naive_path = os.path.join(NAIVE_DIR, f"{img_name}.png")

        dataset = 'TR-CAMO' if 'TR-CAMO' in img_path else 'TR-COD10K'

        results.append({
            'img_name': img_name,
            'img_path': img_path,
            'gt_path': gt_path,
            'pkl_path': pkl_path,
            'refined_path': refined_path,
            'naive_path': naive_path,
            'dataset': dataset,
            'index': idx,
        })
    return results


# ---- Mask Loading ----

def load_mask_binary(path_or_array, target_shape=None):
    """Load a binary mask from file or return array as-is. Returns bool ndarray or None."""
    if isinstance(path_or_array, str):
        if path_or_array.endswith('.pkl'):
            with open(path_or_array, 'rb') as f:
                coarse = pickle.load(f)
            if isinstance(coarse, torch.Tensor):
                coarse = coarse.numpy()
            arr = coarse.squeeze()
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = cv2.imread(path_or_array, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                return None
    else:
        arr = path_or_array

    if target_shape is not None:
        if arr.shape[:2] != target_shape:
            arr = cv2.resize(arr, (target_shape[1], target_shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return (arr > 127)


def load_coarse_soft(pkl_path, target_shape):
    """Load 16x16 pkl, upsample to target_shape, keep float [0,1] values."""
    with open(pkl_path, 'rb') as f:
        coarse = pickle.load(f)
    if isinstance(coarse, torch.Tensor):
        coarse = coarse.numpy()
    coarse = coarse.squeeze().astype(np.float32)
    coarse = cv2.resize(coarse, (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    return np.clip(coarse, 0.0, 1.0)


def load_coarse_binary(pkl_path, target_shape):
    """Load 16x16 pkl, upsample, threshold at 0.5. Returns bool ndarray."""
    soft = load_coarse_soft(pkl_path, target_shape)
    return soft > 0.5


# ---- Statistics ----

def bootstrap_ci(values, n_bootstrap=1000, alpha=0.05):
    """Compute mean with 95% CI via bootstrap."""
    values = np.asarray(values)
    n = len(values)
    means = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        sample = values[rng.choice(n, size=n, replace=True)]
        means[i] = sample.mean()
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return values.mean(), lower, upper


def transition_zone_ratio(soft_mask):
    """Fraction of pixels with value in [0.1, 0.9] — the 'blur zone'."""
    in_zone = np.logical_and(soft_mask >= 0.1, soft_mask <= 0.9)
    return float(in_zone.sum() / soft_mask.size)


# ---- Metric Computation ----

def compute_all_binary_metrics(pred_bool, gt_bool):
    """Compute all 7 metrics for a binary prediction vs GT.

    Args:
        pred_bool: bool ndarray, model prediction
        gt_bool: bool ndarray, ground truth

    Returns:
        dict with keys: bf, r_b, p_b, iou, sm, mae, e_max, e_mean
    """
    from engine.utils.metrics.metric import Smeasure, Emeasure, MAEmeasure, IOUmeasure

    pred_u8 = pred_bool.astype(np.uint8) * 255
    gt_u8 = gt_bool.astype(np.uint8) * 255

    # BF-score (our implementation)
    bf_results = compute_bfscore(pred_u8, gt_u8, width=3)

    # IoU
    iou_measure = IOUmeasure()
    iou_measure.step(pred=pred_u8, gt=gt_u8)
    iou = iou_measure.get_results()['miou']

    # S-measure
    sm_measure = Smeasure()
    sm_measure.step(pred=pred_u8, gt=gt_u8)
    sm = sm_measure.get_results()['sm']

    # E-measure
    em_measure = Emeasure()
    em_measure.step(pred=pred_u8, gt=gt_u8)
    em_results = em_measure.get_results()['em']
    e_max = em_results['curve'].max()
    e_mean = em_results['curve'].mean()

    # MAE
    mae_measure = MAEmeasure()
    mae_measure.step(pred=pred_u8, gt=gt_u8)
    mae = mae_measure.get_results()['mae']

    return {
        'bf': bf_results['bf'],
        'r_b': bf_results['r_b'],
        'p_b': bf_results['p_b'],
        'iou': float(iou),
        'sm': float(sm),
        'mae': float(mae),
        'e_max': float(e_max),
        'e_mean': float(e_mean),
    }


# ---- Centroid Utilities ----

def largest_cc_centroid(mask_bool):
    """Centroid of largest connected component. Returns (cx, cy) or None if empty."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    return tuple(centroids[largest_idx])  # (cx, cy)


# ---- Report Writer ----

class MarkdownReport:
    """Helper to build a markdown report incrementally."""

    def __init__(self, title):
        self.lines = [f"# {title}", "", f"*Generated: {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}*", ""]

    def add_heading(self, text, level=2):
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")

    def add_text(self, text):
        self.lines.append(text)
        self.lines.append("")

    def add_table(self, headers, rows):
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            self.lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self.lines.append("")

    def add_code_block(self, code, lang=''):
        self.lines.append(f"```{lang}")
        self.lines.append(code)
        self.lines.append("```")
        self.lines.append("")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"Report saved to {path}")


# ---- Narrow-Band Localized Metrics ----

def compute_localized_metrics(pred, gt, band_width=10):
    """Compute IoU and BF-score restricted to a narrow band around GT boundary.

    Full-image metrics are dominated by DINOv2 localization error. This function
    isolates boundary quality by computing metrics only within `band_width` pixels
    of the GT boundary, where SAM2's edge refinement effect is concentrated.

    Args:
        pred: uint8 or bool ndarray, binary mask
        gt: uint8 or bool ndarray, binary mask
        band_width: dilation radius in pixels (5/10/20)

    Returns:
        dict: {'local_iou': float, 'local_bf': float, 'local_r_b': float,
               'local_p_b': float, 'band_pixel_count': int, 'band_fg_count': int,
               'band_bg_count': int}
        Returns None if GT has no boundary or band is empty.
    """
    pred_bool = (pred > 127) if pred.dtype == np.uint8 else pred.astype(bool)
    gt_bool = (gt > 127) if gt.dtype == np.uint8 else gt.astype(bool)

    # Extract GT boundary via morphological gradient
    gt_u8 = gt_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_boundary = (cv2.dilate(gt_u8, kernel) - cv2.erode(gt_u8, kernel)) > 0

    # Dilate boundary to create narrow band
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
    band_mask = cv2.dilate(gt_boundary.astype(np.uint8), dilate_kernel) > 0

    band_count = band_mask.sum()
    if band_count == 0:
        return None

    # IoU within narrow band
    pred_in_band = np.logical_and(pred_bool, band_mask)
    gt_in_band = np.logical_and(gt_bool, band_mask)
    inter = np.logical_and(pred_in_band, gt_in_band).sum()
    union = np.logical_or(pred_in_band, gt_in_band).sum()
    local_iou = float(inter / union) if union > 0 else 0.0

    # BF-score within narrow band
    pred_boundary = _extract_boundary(pred_bool)
    gt_boundary_bool = gt_boundary  # already extracted above

    # Restrict boundary pixels to those within the band
    pred_b_in_band = np.logical_and(pred_boundary, band_mask)
    gt_b_in_band = np.logical_and(gt_boundary_bool, band_mask)

    pred_b_count = pred_b_in_band.sum()
    gt_b_count = gt_b_in_band.sum()

    if pred_b_count == 0 and gt_b_count == 0:
        return {'local_iou': local_iou, 'local_bf': 0.0, 'local_r_b': 0.0,
                'local_p_b': 0.0, 'band_pixel_count': int(band_count),
                'band_fg_count': int(gt_in_band.sum()),
                'band_bg_count': int(band_count - gt_in_band.sum())}

    # Precision: distance from pred boundary pixels to GT boundary pixels
    if pred_b_count > 0:
        dt_gt = cv2.distanceTransform(
            (1 - gt_boundary_bool.astype(np.uint8)), cv2.DIST_L2, maskSize=3)
        matched_pred = (dt_gt[pred_b_in_band] <= 3).sum()
        local_p_b = float(matched_pred / pred_b_count)
    else:
        local_p_b = 0.0

    # Recall: distance from GT boundary pixels to pred boundary pixels
    if gt_b_count > 0:
        dt_pred = cv2.distanceTransform(
            (1 - pred_boundary.astype(np.uint8)), cv2.DIST_L2, maskSize=3)
        matched_gt = (dt_pred[gt_b_in_band] <= 3).sum()
        local_r_b = float(matched_gt / gt_b_count)
    else:
        local_r_b = 0.0

    if local_p_b + local_r_b > 0:
        local_bf = 2 * local_p_b * local_r_b / (local_p_b + local_r_b)
    else:
        local_bf = 0.0

    return {
        'local_iou': local_iou,
        'local_bf': local_bf,
        'local_r_b': local_r_b,
        'local_p_b': local_p_b,
        'band_pixel_count': int(band_count),
        'band_fg_count': int(gt_in_band.sum()),
        'band_bg_count': int(band_count - gt_in_band.sum()),
    }


# ---- Soft Narrow-Band Metrics (no thresholding) ----

def compute_narrow_band_soft(pred_soft, gt_binary, band_width=2):
    """Compute soft MAE and soft IoU within a narrow band around GT boundary.

    Unlike compute_localized_metrics (binary-only), this operates on soft float
    predictions [0,1] vs binary GT {0,1}. Designed for 68x68 resolution where
    SAM2's boundary crispness (LANCZOS from original PNG) vs coarse's boundary
    blur (bilinear from 16x16) creates measurable differences in the narrow band.

    Args:
        pred_soft: float ndarray [0,1], soft prediction (no thresholding)
        gt_binary: bool or uint8 ndarray, binary GT {0,1} or {0,255}
        band_width: dilation radius in pixels (2-3 recommended for 68x68)

    Returns:
        dict: {'soft_mae': float, 'soft_mse': float, 'band_pixels': int,
               'band_fg_pixels': int, 'band_bg_pixels': int}
        Returns None if GT has no boundary.
    """
    gt_bool = (gt_binary > 127) if gt_binary.dtype == np.uint8 else gt_binary.astype(bool)

    # GT boundary via morphological gradient
    gt_u8 = gt_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_boundary = (cv2.dilate(gt_u8, kernel) - cv2.erode(gt_u8, kernel)) > 0

    if gt_boundary.sum() == 0:
        return None

    # Dilate boundary to create narrow band
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
    band_mask = cv2.dilate(gt_boundary.astype(np.uint8), dilate_kernel) > 0

    band_count = band_mask.sum()
    if band_count == 0:
        return None

    # Soft metrics within narrow band
    pred_in_band = pred_soft[band_mask]
    gt_in_band = gt_bool[band_mask].astype(np.float32)

    soft_mae = float(np.mean(np.abs(pred_in_band - gt_in_band)))

    # Soft MSE (penalizes large deviations more heavily)
    soft_mse = float(np.mean((pred_in_band - gt_in_band) ** 2))

    band_fg = gt_in_band.sum()
    band_bg = band_count - band_fg

    return {
        'soft_mae': soft_mae,
        'soft_mse': soft_mse,
        'band_pixels': int(band_count),
        'band_fg_pixels': int(band_fg),
        'band_bg_pixels': int(band_bg),
    }


if __name__ == '__main__':
    # Smoke test
    print("Testing BF-score...")
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[20:80, 20:80] = 255
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[22:78, 22:78] = 255
    r = compute_bfscore(pred, gt, width=3)
    print(f"  BF-score: {r}, expecting near 1.0")

    print("Testing narrow-band metrics...")
    # Test 1: Small boundary offset within narrow band
    n1 = compute_localized_metrics(pred, gt, band_width=10)
    print(f"  Band 10px: {n1}")

    # Test 2: Different prediction — offset by 5px
    pred2 = np.zeros((100, 100), dtype=np.uint8)
    pred2[25:85, 25:85] = 255
    n2 = compute_localized_metrics(pred2, gt, band_width=10)
    print(f"  Offset pred band 10px: {n2}")

    # Test 3: Non-overlapping prediction (simulates extreme DINOv2 error)
    pred3 = np.zeros((100, 100), dtype=np.uint8)
    pred3[0:20, 0:20] = 255
    n3 = compute_localized_metrics(pred3, gt, band_width=10)
    print(f"  Non-overlap pred band 10px: {n3}")
    print(f"  Expected: local_iou=0, local_bf=0 (pred nowhere near GT boundary)")

    print("Testing build_unified_index...")
    idx = build_unified_index()
    print(f"  Total: {len(idx)} images")
    print(f"  Sample: {idx[0]['img_name']}")
    print(f"  GT path: {idx[0]['gt_path']}")
    print(f"  Dataset: {idx[0]['dataset']}")
    print("utils_metrics.py OK")
