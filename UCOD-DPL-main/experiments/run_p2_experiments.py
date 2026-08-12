"""P2: Compare 5 label selection strategies on 100 images with GT."""
import sys, os, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
sys.path.insert(0, '.')
from scripts.offline_sam2_refine import load_coarse_label, compute_iou, compute_edge_align, CFG

RAW_DIR = './datasets/cache/raw_sam2_outputs'
GT_DIR = r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\gt'
IM_DIR = r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\im'

def get_mask(m, im_h, im_w):
    mb = m.astype(bool) if m.dtype != bool else m
    if mb.shape[:2] != (im_h, im_w):
        mb = cv2.resize(mb.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128
    return mb

def strategy_best_iou(masks, coarse_bool):
    best_iou, best_mask = -1, None
    for m in masks:
        iou = compute_iou(m, coarse_bool)
        if iou > best_iou: best_iou, best_mask = iou, m
    return best_mask if best_mask is not None else coarse_bool

def strategy_edge_align(masks, scores, coarse_bool, image):
    best_s, best_mask = -1, None
    for i, m in enumerate(masks):
        iou = compute_iou(m, coarse_bool)
        edge = compute_edge_align(m, image) / 255.0
        s = 0.3*scores[i] + 0.4*iou + 0.3*edge
        if s > best_s: best_s, best_mask = s, m
    return best_mask if best_mask is not None else coarse_bool

def strategy_reverse(masks, coarse_bool):
    best_iou, best_mask = 999, None
    for m in masks:
        iou = compute_iou(m, coarse_bool)
        if iou < best_iou: best_iou, best_mask = iou, m
    return best_mask if best_mask is not None else coarse_bool

def strategy_cpi_union(masks, coarse_bool, im_h, im_w):
    union = np.zeros((im_h, im_w), dtype=bool)
    for m in masks:
        iou = compute_iou(m, coarse_bool)
        if iou >= 0.25:
            part = np.logical_and(m, coarse_bool)
            union = np.logical_or(union, part)
    uncovered = np.logical_and(coarse_bool, ~union)
    return np.logical_or(union, uncovered)

def strategy_area_match(masks, coarse_bool):
    ca = coarse_bool.sum()
    if ca == 0: return coarse_bool
    best_diff, best_mask = 999, None
    for m in masks:
        diff = abs(m.sum() - ca)
        if diff < best_diff: best_diff, best_mask = diff, m
    return best_mask if best_mask is not None else coarse_bool

def main():
    with open(os.path.join(CFG['coarse_dir'], 'index.json'), 'r') as f:
        index_map = json.load(f)
    paths = sorted([str(p) for p in Path(IM_DIR).glob('*.jpg')] + [str(p) for p in Path(IM_DIR).glob('*.png')])
    name_to_idx = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}

    gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(('.jpg','.png'))][:100]

    results = {k: [] for k in ['best_iou','edge_align','reverse','cpi_union','area_match','coarse']}

    for gt_file in gt_files:
        gt_name = os.path.splitext(gt_file)[0]
        npz_path = os.path.join(RAW_DIR, gt_name + '.npz')
        if not os.path.exists(npz_path): continue

        gt = np.array(Image.open(os.path.join(GT_DIR, gt_file)).convert('L')) >= 128
        img_path = None
        for ext in ['.jpg', '.png']:
            p = os.path.join(IM_DIR, gt_name + ext)
            if os.path.exists(p): img_path = p; break
        if img_path is None: continue
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        im_h, im_w = image.shape[:2]

        idx = name_to_idx.get(gt_name)
        if idx is None: continue
        coarse = load_coarse_label(os.path.join(CFG['coarse_dir'], index_map[str(idx)]), (im_h, im_w))
        coarse_bool = coarse >= 128

        if gt.shape[:2] != (im_h, im_w):
            gt = cv2.resize(gt.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128

        data = np.load(npz_path, allow_pickle=True)
        masks_raw = data['masks']
        scores = list(data['scores'])
        masks = [get_mask(m, im_h, im_w) for m in masks_raw]

        results['coarse'].append(compute_iou(coarse_bool, gt))
        results['best_iou'].append(compute_iou(strategy_best_iou(masks, coarse_bool), gt))
        results['edge_align'].append(compute_iou(strategy_edge_align(masks, scores, coarse_bool, image), gt))
        results['reverse'].append(compute_iou(strategy_reverse(masks, coarse_bool), gt))
        results['cpi_union'].append(compute_iou(strategy_cpi_union(masks, coarse_bool, im_h, im_w), gt))
        results['area_match'].append(compute_iou(strategy_area_match(masks, coarse_bool), gt))

    cc = np.array(results['coarse'])
    with open('experiments/output/p2_results.txt', 'w', encoding='utf-8') as f:
        f.write('N=' + str(len(cc)) + '\nCoarse IoU: ' + str(round(np.mean(cc),4)) + '\n\n')
        for strat in ['best_iou','edge_align','reverse','cpi_union','area_match']:
            arr = np.array(results[strat])
            delta = np.mean(arr - cc)
            win = np.mean(arr > cc) * 100
            f.write(strat + ': IoU=' + str(round(np.mean(arr),4)) + '  delta=' + str(round(delta,4)) + '  win=' + str(round(win,1)) + '%\n')
        best = max(['best_iou','edge_align','reverse','cpi_union','area_match'], key=lambda s: np.mean(np.array(results[s]) - cc))
        f.write('\nBEST: ' + best + '\n')
    print('Done! See experiments/output/p2_results.txt')

if __name__ == '__main__':
    main()
