"""Quick test: does looser prompting make SAM2 actually produce different masks?"""
import sys, os, pickle, torch, json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.offline_sam2_refine import (
    load_coarse_label, compute_iou, compute_edge_align,
    sample_points_in_mask, expand_bbox, SAM2Wrapper,
    stage2_sam2_inference, CFG
)

def prompt_loose(coarse_mask, im_h, im_w, rng):
    """V_loose: 1 centroid point, no box, no negatives."""
    fg_y, fg_x = np.where(coarse_mask >= 128)
    if len(fg_y) == 0:
        return None, [], []
    cx, cy = int(fg_x.mean()), int(fg_y.mean())
    return None, [(cx, cy)], []

def prompt_centroid_box(coarse_mask, im_h, im_w, rng):
    """V_centroid_box: centroid point + bounding box with 0.5 expansion, no negatives."""
    fg_y, fg_x = np.where(coarse_mask >= 128)
    if len(fg_y) == 0:
        return None, [], []
    cx, cy = int(fg_x.mean()), int(fg_y.mean())
    x, y = int(fg_x.min()), int(fg_y.min())
    w, h = int(fg_x.max() - fg_x.min()), int(fg_y.max() - fg_y.min())
    bbox = expand_bbox((x, y, w, h), 0.5, im_h, im_w)
    return bbox, [(cx, cy)], []

def run_one(wrapper, image, coarse, bbox, pos, neg, rng):
    """Run SAM2, pick best mask by IoU with coarse, return stats."""
    if len(pos) == 0:
        return {'error': 'no points'}
    try:
        masks, scores = stage2_sam2_inference(wrapper, image, coarse, bbox, pos, neg)
    except Exception as e:
        return {'error': str(e)}
    
    coarse_bool = coarse >= 128
    best_iou, best_mask = -1, None
    for m in masks:
        if m.shape != coarse.shape:
            m = cv2.resize(m.astype(np.uint8)*255,
                          (coarse.shape[1], coarse.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
            m_bool = m >= 128
        else:
            m_bool = m.astype(bool) if m.dtype != bool else m
        iou = compute_iou(m_bool, coarse_bool)
        if iou > best_iou:
            best_iou, best_mask = iou, m_bool
    
    sam_area = best_mask.sum()
    coarse_area_bin = coarse_bool.sum()
    # Also count ALL non-zero coarse pixels (including blur zone)
    coarse_area_all = (coarse > 0).sum() if coarse.dtype == np.uint8 else coarse_bool.sum()
    return {
        'iou_coarse': best_iou,
        'sam_area': sam_area,
        'coarse_area_bin': coarse_area_bin,
        'coarse_area_all': coarse_area_all,
        'area_ratio_bin': sam_area / max(coarse_area_bin, 1),
        'diff_ratio': (np.logical_xor(best_mask, coarse_bool).sum()) / max(np.logical_or(best_mask, coarse_bool).sum(), 1),
        'n_masks': len(masks),
        'scores': [round(float(s), 3) for s in scores],
    }

def main():
    print("Loading SAM2...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapper = SAM2Wrapper(device)
    rng = np.random.RandomState(42)
    
    # Pick test images: 5 degraded + 5 improved + 5 neutral
    import csv
    csv_path = r'C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main\experiments\output\label_quality_per_image.csv'
    rows = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[row['img_name']] = row
    
    deltas = [(n, float(r['refined_bf'])-float(r['coarse_bin_bf'])) for n, r in rows.items()]
    deltas.sort(key=lambda x: x[1])
    
    test_names = (
        [n for n, d in deltas[:5]] +         # 5 most degraded
        [n for n, d in deltas[-5:]] +         # 5 most improved
        [n for n, d in deltas[len(deltas)//2-2:len(deltas)//2+3]]  # 5 neutral
    )
    
    coarse_dir = CFG['coarse_dir']
    image_dirs = CFG['image_dirs']
    paths = []
    for d in image_dirs:
        paths.extend(sorted(str(p) for p in Path(d).glob('*.jpg')))
        paths.extend(sorted(str(p) for p in Path(d).glob('*.png')))
    name_to_idx = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}
    
    with open(os.path.join(coarse_dir, 'index.json'), 'r') as f:
        index_map = json.load(f)
    
    strategies = ['V_loose', 'V_centroid_box', 'V_current']
    
    print(f"\nTesting {len(test_names)} images x {len(strategies)} strategies...")
    print(f"{'Image':<35} {'Strategy':<20} {'IoU_coarse':>10} {'diff%':>8} {'area_ratio':>10} {'SAM_area':>10} {'Coarse_bin':>10}")
    print("-" * 115)
    
    for name in test_names:
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        
        # Find image path
        img_path = None
        for d in image_dirs:
            for ext in ['.jpg', '.png']:
                p = os.path.join(d, name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
        if img_path is None:
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_h, im_w = image.shape[:2]
        coarse = load_coarse_label(os.path.join(coarse_dir, index_map[str(idx)]), (im_h, im_w))
        
        for strat_name in strategies:
            if strat_name == 'V_loose':
                bbox, pos, neg = prompt_loose(coarse, im_h, im_w, rng)
            elif strat_name == 'V_centroid_box':
                bbox, pos, neg = prompt_centroid_box(coarse, im_h, im_w, rng)
            elif strat_name == 'V_current':
                from scripts.offline_sam2_refine import stage1_adaptive_prompt
                bbox, pos, neg = stage1_adaptive_prompt(coarse, im_h, im_w, rng)
            
            result = run_one(wrapper, image, coarse, bbox, pos, neg, rng)
            if 'error' in result:
                print(f"{name:<35} {strat_name:<20} ERROR: {result['error']}")
            else:
                print(f"{name:<35} {strat_name:<20} {result['iou_coarse']:>10.4f} {result['diff_ratio']*100:>7.1f}% {result['area_ratio_bin']:>10.4f} {result['sam_area']:>10} {result['coarse_area_bin']:>10}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
