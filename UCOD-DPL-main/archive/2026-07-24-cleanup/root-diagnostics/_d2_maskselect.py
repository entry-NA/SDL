# === 诊断脚本 2: mask选择公式对比 + 标签覆盖率 ===
# 运行: python _d2_maskselect.py > _d2_result.txt 2>&1
import sys, os
sys.path.insert(0, 'scripts')
import offline_sam2_refine as osm
import numpy as np, cv2, json, pickle
from PIL import Image

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align

RAW_DIR = 'datasets/cache/raw_sam2_outputs'
BACKUP_DIR = 'datasets/cache/refined_pseudo_labels'
VLOOSE_DIR = 'datasets/cache/refined_pseudo_labels_vloose'

coarse_dir = CFG['coarse_dir']
with open(os.path.join(coarse_dir, 'index.json')) as f:
    index_map = json.load(f)

# Find 10 images with both raw npz and labels
test_names = sorted([f.replace('.npz', '') for f in os.listdir(RAW_DIR) if f.endswith('.npz')])[:10]
# Filter to CAMO only for consistency
test_names = [n for n in test_names if n.startswith('camourflage_')][:8]

print('='*60)
print('TEST A: stage3(IoU*score) vs regenerate(3因子) mask选择')
print('='*60)

for name in test_names:
    npz_path = os.path.join(RAW_DIR, name + '.npz')
    data = np.load(npz_path, allow_pickle=True)
    masks = list(data['masks'])
    scores = list(data['scores'])
    
    # Find image
    img_path = None
    for d in CFG['image_dirs']:
        for ext in ['.jpg', '.png']:
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                img_path = p
                break
    if img_path is None:
        continue
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Load coarse
    for k, v in index_map.items():
        if name in v:
            with open(os.path.join(coarse_dir, v), 'rb') as f:
                coarse_pkl = pickle.load(f)
            break
    else:
        continue
    if hasattr(coarse_pkl, 'numpy'):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    # Method A: stage3 = IoU * score
    best_a_idx, best_a_score = -1, -float('inf')
    # Method B: regenerate = 0.3*score + 0.4*IoU + 0.3*EdgeAlign/255
    best_b_idx, best_b_score = -1, -float('inf')
    
    print(f'\n{name}:')
    for i, m in enumerate(masks):
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != (h, w):
            mb = cv2.resize(mb.astype(np.uint8)*255, (w, h), cv2.INTER_NEAREST) >= 128
        iou_val = compute_iou(mb, coarse_bool)
        edge_raw = compute_edge_align(mb, image)
        
        score_a = iou_val * scores[i]
        score_b = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * (edge_raw / 255.0)
        
        tag_a = '*' if score_a > best_a_score else ' '
        tag_b = '+' if score_b > best_b_score else ' '
        if score_a > best_a_score:
            best_a_score, best_a_idx = score_a, i
        if score_b > best_b_score:
            best_b_score, best_b_idx = score_b, i
        
        print(f'  mask[{i}]{tag_a}{tag_b}: IoU={iou_val:.3f}, SAM={scores[i]:.3f}, Edge={edge_raw:.1f}')
        print(f'    A(IoU*score)={score_a:.4f}  B(3factor)={score_b:.4f}')
    
    agree = 'AGREE' if best_a_idx == best_b_idx else f'DIFFER: A->mask[{best_a_idx}], B->mask[{best_b_idx}]'
    print(f'  >> {agree}')

print()
print('='*60)
print('TEST B: 标签前景覆盖率对比 (backup vs vloose vs broken)')
print('='*60)

for lbl_dir, lbl_name in [(BACKUP_DIR, 'backup_old'), (VLOOSE_DIR, 'vloose')]:
    if not os.path.isdir(lbl_dir):
        print(f'{lbl_name}: NOT FOUND')
        continue
    files = sorted([f for f in os.listdir(lbl_dir) if f.startswith('camourflage_') and f.endswith('.png')])[:20]
    fg_ratios = []
    for f in files:
        img = np.array(Image.open(os.path.join(lbl_dir, f)))
        ratio = (img >= 128).sum() / img.size
        fg_ratios.append(ratio)
    
    arr = np.array(fg_ratios)
    print(f'\n{lbl_name} ({len(files)} files):')
    print(f'  fg_ratio: mean={arr.mean():.4f}, median={np.median(arr):.4f}, min={arr.min():.4f}, max={arr.max():.4f}')
    print(f'  zero_count: {(arr == 0).sum()}')
    print(f'  <0.001 count: {(arr < 0.001).sum()}')
    print(f'  <0.01 count: {(arr < 0.01).sum()}')

print()
print('='*60)
print('TEST C: 同一图片 backup vs vloose 标签差异度')
print('='*60)

for lbl_name, lbl_dir in [('vloose', VLOOSE_DIR)]:
    common = []
    for f in sorted(os.listdir(BACKUP_DIR)):
        if f.startswith('camourflage_') and f.endswith('.png'):
            vf = os.path.join(lbl_dir, f)
            if os.path.exists(vf):
                common.append(f)
    common = common[:10]
    
    diffs = []
    for f in common:
        bak = np.array(Image.open(os.path.join(BACKUP_DIR, f)))
        vls = np.array(Image.open(os.path.join(lbl_dir, f)))
        diff_pct = (bak != vls).sum() / bak.size
        diffs.append(diff_pct)
    
    arr = np.array(diffs)
    print(f'backup vs {lbl_name} ({len(common)} files):')
    print(f'  diff%: mean={arr.mean():.4f}, max={arr.max():.4f}')
    for i, f in enumerate(common[:5]):
        print(f'  {f}: diff={diffs[i]:.4f}')
