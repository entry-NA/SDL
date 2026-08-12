"""Regenerate labels from raw SAM2 V_loose outputs with corrected edge_align.
S = 0.3*scores[i] + 0.4*IoU(mask_i, coarse) + 0.3*EdgeAlign(mask_i, image)/255
Missing 14 images -> fallback to coarse labels."""
import sys, os, json, pickle, torch, numpy as np, cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Import from scripts/ directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import offline_sam2_refine as osm
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align
CFG = osm.CFG

RAW_DIR = './datasets/cache/raw_sam2_outputs'
OUT_DIR = './datasets/cache/refined_pseudo_labels'
os.makedirs(OUT_DIR, exist_ok=True)

# Build mappings
image_dirs = CFG['image_dirs']
paths = []
for d in image_dirs:
    paths.extend(sorted(str(p) for p in Path(d).glob('*.jpg')))
    paths.extend(sorted(str(p) for p in Path(d).glob('*.png')))
name_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
name_to_idx = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}

coarse_dir = CFG['coarse_dir']
with open(os.path.join(coarse_dir, 'index.json'), 'r') as f:
    index_map = json.load(f)

npz_names = set(f.replace('.npz', '') for f in os.listdir(RAW_DIR))
all_names = sorted(name_to_path.keys())

stats = {'edge_align': 0, 'coarse_fallback': 0, 'errors': 0}

for name in tqdm(all_names, desc='Regen'):
    try:
        idx = name_to_idx[name]
        img_path = name_to_path[name]

        # Load image first to get dimensions
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if image is None:
            stats['errors'] += 1; continue
        im_h, im_w = image.shape[:2]

        # Load coarse label at image resolution
        pkl_path = os.path.join(coarse_dir, index_map[str(idx)])
        with open(pkl_path, 'rb') as f:
            coarse = pickle.load(f)
        if isinstance(coarse, torch.Tensor):
            coarse = coarse.numpy()
        coarse = coarse.squeeze()
        coarse = (coarse * 255).astype(np.uint8)
        coarse = cv2.resize(coarse, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        coarse_bool = coarse >= 128

        if name in npz_names:
            data = np.load(os.path.join(RAW_DIR, name + '.npz'), allow_pickle=True)
            masks_raw = data['masks']
            scores = list(data['scores'])

            best_s, best_mask = -1.0, None
            for i, m in enumerate(masks_raw):
                mb = m.astype(bool) if m.dtype != bool else m
                if mb.shape[:2] != (im_h, im_w):
                    mb = cv2.resize(mb.astype(np.uint8)*255, (im_w, im_h),
                                    cv2.INTER_NEAREST) >= 128
                iou_val = compute_iou(mb, coarse_bool)
                edge_val = compute_edge_align(mb, image)
                s = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * edge_val
                if s > best_s:
                    best_s, best_mask = s, mb

            if best_mask is None:
                best_mask = coarse_bool
                stats['coarse_fallback'] += 1
            else:
                stats['edge_align'] += 1
        else:
            best_mask = coarse_bool
            stats['coarse_fallback'] += 1

        out = (best_mask.astype(np.uint8)) * 255
        Image.fromarray(out, mode='L').save(os.path.join(OUT_DIR, name + '.png'))

    except Exception as e:
        stats['errors'] += 1
        if stats['errors'] <= 3:
            print('\n[ERR] ' + name + ': ' + str(e)[:100])

print('\nDone: ' + str(stats['edge_align']) + ' edge_align, ' +
      str(stats['coarse_fallback']) + ' coarse, ' +
      str(stats['errors']) + ' errors')
