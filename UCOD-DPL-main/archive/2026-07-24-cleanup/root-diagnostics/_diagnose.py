#!/usr/bin/env python3
\"\"\"
=== SAM2 精修管线诊断脚本 ===
测试你的四个创新点的实际表现：
  1. compute_edge_align 返回值域（确认 [0,255] vs [0,1]）
  2. stage3 mask 选择 vs regenerate_labels 的 3 因子公式
  3. stage4 门控阈值是否被 edge_align 短路
  4. _backup_old 标签 vs vloose 标签的前景覆盖率对比
  5. 测试集上的对比：CAMO 5张图的标签质量
\"\"\"
import sys, os, json, pickle, numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
import offline_sam2_refine as osm

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align

WORK_DIR = r'.'
TEST_IMAGES = [
    r'datasets/RefCOD/TR-CAMO/im/camourflage_00001.jpg',
    r'datasets/RefCOD/TR-CAMO/im/camourflage_00002.jpg',
    r'datasets/RefCOD/TR-CAMO/im/camourflage_00010.jpg',
    r'datasets/RefCOD/TR-CAMO/im/camourflage_00050.jpg',
    r'datasets/RefCOD/TR-CAMO/im/camourflage_00100.jpg',
]

def load_coarse_pkl(img_path, index_map, coarse_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    # Find index by scanning
    idx = None
    for k, v in index_map.items():
        if v.endswith(name + '.pkl') or name in v:
            idx = int(k)
            break
    if idx is None:
        # Fallback: load by file stem match
        for f in os.listdir(coarse_dir):
            if name in f:
                with open(os.path.join(coarse_dir, f), 'rb') as fh:
                    return pickle.load(fh)
        return None
    pkl_path = os.path.join(coarse_dir, index_map[str(idx)])
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def pkl_to_mask(coarse_pkl, im_h, im_w):
    if hasattr(coarse_pkl, 'numpy'):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    coarse = cv2.resize(coarse, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    return coarse

def mask_stats(mask, name=""):
    mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
    mask_bool = mask_u8 >= 128
    total = mask_bool.size
    fg = mask_bool.sum()
    ratio = fg / total if total > 0 else 0
    return {'name': name, 'fg_ratio': ratio, 'fg_pixels': int(fg), 'total': int(total)}

# =============================================
# TEST 1: edge_align 返回值域检查
# =============================================
print('='*60)
print('TEST 1: compute_edge_align 返回值域')
print('='*60)

coarse_dir = CFG['coarse_dir']
with open(os.path.join(coarse_dir, 'index.json'), 'r') as f:
    index_map = json.load(f)

for img_path in TEST_IMAGES[:3]:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]
    coarse_pkl = load_coarse_pkl(img_path, index_map, coarse_dir)
    if coarse_pkl is None:
        print(f'  SKIP {os.path.basename(img_path)}: no pkl')
        continue
    coarse = pkl_to_mask(coarse_pkl, im_h, im_w)
    coarse_bool = coarse >= 128
    
    # Create a test mask: the coarse mask itself
    ea = compute_edge_align(coarse_bool, image)
    print(f'  {os.path.basename(img_path)}: edge_align(coarse, image) = {ea:.4f}')
    
    # Create a random mask to test range
    rng = np.random.RandomState(42)
    random_mask = (rng.random((im_h, im_w)) > 0.98).astype(bool)
    ea_random = compute_edge_align(random_mask, image)
    print(f'    edge_align(random_2pct, image) = {ea_random:.4f}')

print(f'  CFG thresholds: s_lower={CFG[\"s_lower\"]}, s_upper={CFG[\"s_upper\"]}')
print(f'  If edge_align is [0,255], then gamma*edge = [0, {0.3*255:.1f}] >> thresholds')
print(f'  -> gating is ALWAYS full adoption (S >> 0.8)')
print()

# =============================================
# TEST 2: mask 选择公式对比
# =============================================
print('='*60)
print('TEST 2: stage3 (IoU*score) vs regenerate (3因子)')
print('='*60)

RAW_DIR = r'datasets/cache/raw_sam2_outputs'
for img_path in TEST_IMAGES[:3]:
    name = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(RAW_DIR, name + '.npz')
    if not os.path.exists(npz_path):
        print(f'  SKIP {name}: no npz')
        continue
    
    data = np.load(npz_path, allow_pickle=True)
    masks = list(data['masks'])
    scores = list(data['scores'])
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]
    coarse_pkl = load_coarse_pkl(img_path, index_map, coarse_dir)
    if coarse_pkl is None:
        continue
    coarse = pkl_to_mask(coarse_pkl, im_h, im_w)
    coarse_bool = coarse >= 128
    
    print(f'\n  {name}: {len(masks)} candidates')
    
    # Method A: stage3 (IoU * score)
    best_a_idx, best_a_score = -1, -float('inf')
    # Method B: regenerate (0.3*score + 0.4*IoU + 0.3*EdgeAlign/255)
    best_b_idx, best_b_score = -1, -float('inf')
    
    for i, m in enumerate(masks):
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != (im_h, im_w):
            mb = cv2.resize(mb.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128
        
        iou_val = compute_iou(mb, coarse_bool)
        edge_raw = compute_edge_align(mb, image)
        edge_val = edge_raw / 255.0  # CORRECT normalization
        
        # Method A
        score_a = iou_val * scores[i]
        # Method B
        score_b = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * edge_val
        
        best_str = ''
        if score_a > best_a_score:
            best_a_score, best_a_idx = score_a, i
        if score_b > best_b_score:
            best_b_score, best_b_idx = score_b, i
        
        print(f'    mask[{i}]: IoU={iou_val:.3f}, SAM_score={scores[i]:.3f}, edge_raw={edge_raw:.1f}, edge_norm={edge_val:.3f}')
        print(f'      method_A(IoU*score)={score_a:.4f}, method_B(3factor)={score_b:.4f}')
    
    agree = 'SAME' if best_a_idx == best_b_idx else f'DIFFER (A->{best_a_idx}, B->{best_b_idx})'
    print(f'  >>> Selection: {agree}')

print()

# =============================================
# TEST 3: 门控阈值分析
# =============================================
print('='*60)
print('TEST 3: stage4 门控 S 值实际分布')
print('='*60)
print(f'  CFG: alpha={CFG[\"alpha\"]}, beta={CFG[\"beta\"]}, gamma={CFG[\"gamma\"]}')
print(f'  Thresholds: s_lower={CFG[\"s_lower\"]}, s_upper={CFG[\"s_upper\"]}')

for img_path in TEST_IMAGES[:3]:
    name = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(RAW_DIR, name + '.npz')
    if not os.path.exists(npz_path):
        continue
    
    data = np.load(npz_path, allow_pickle=True)
    masks = list(data['masks'])
    scores = list(data['scores'])
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im_h, im_w = image.shape[:2]
    coarse_pkl = load_coarse_pkl(img_path, index_map, coarse_dir)
    if coarse_pkl is None:
        continue
    coarse = pkl_to_mask(coarse_pkl, im_h, im_w)
    coarse_bool = coarse >= 128
    
    # Find best mask (using regenerate method)
    best_idx, best_score = -1, -float('inf')
    for i, m in enumerate(masks):
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != (im_h, im_w):
            mb = cv2.resize(mb.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128
        iou_val = compute_iou(mb, coarse_bool)
        edge_val = compute_edge_align(mb, image) / 255.0
        s = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * edge_val
        if s > best_score:
            best_score, best_idx = s, i
    
    # Now compute S with and without /255 fix
    best_m = masks[best_idx]
    mb = best_m.astype(bool) if best_m.dtype != bool else best_m
    if mb.shape[:2] != (im_h, im_w):
        mb = cv2.resize(mb.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128
    
    iou_ori = compute_iou(mb, coarse_bool)
    edge_raw = compute_edge_align(mb, image)
    
    # Current buggy S
    S_buggy = CFG['alpha'] * scores[best_idx] + CFG['beta'] * iou_ori + CFG['gamma'] * edge_raw
    # Fixed S
    S_fixed = CFG['alpha'] * scores[best_idx] + CFG['beta'] * iou_ori + CFG['gamma'] * (edge_raw / 255.0)
    
    buggy_result = 'FULL' if S_buggy > CFG['s_upper'] else ('FALLBACK' if S_buggy < CFG['s_lower'] else 'FUSION')
    fixed_result = 'FULL' if S_fixed > CFG['s_upper'] else ('FALLBACK' if S_fixed < CFG['s_lower'] else 'FUSION')
    
    print(f'  {name}: S_buggy={S_buggy:.4f} -> {buggy_result}, S_fixed={S_fixed:.4f} -> {fixed_result}')
    print(f'    iou_pred={scores[best_idx]:.3f}, iou_ori={iou_ori:.3f}, edge_raw={edge_raw:.1f}, edge_norm={edge_raw/255:.3f}')

print()

# =============================================
# TEST 4: _backup_old vs vloose 标签前景对比
# =============================================
print('='*60)
print('TEST 4: 标签前景覆盖率对比 (10张CAMO)')
print('='*60)

BACKUP_DIR = r'datasets/cache/refined_pseudo_labels'
VLOOSE_DIR = r'datasets/cache/refined_pseudo_labels_vloose'
BROKEN_DIR = r'datasets/cache/refined_pseudo_labels_broken'

for lbl_dir, lbl_name in [(BACKUP_DIR, 'backup_old'), (VLOOSE_DIR, 'vloose'), (BROKEN_DIR, 'broken_current')]:
    if not os.path.isdir(lbl_dir):
        print(f'  {lbl_name}: dir not found')
        continue
    files = sorted([f for f in os.listdir(lbl_dir) if f.startswith('camourflage_') and f.endswith('.png')])[:10]
    stats_all = []
    for f in files:
        img = np.array(Image.open(os.path.join(lbl_dir, f)))
        ms = mask_stats(img, f)
        stats_all.append(ms)
    avg_fg = np.mean([s['fg_ratio'] for s in stats_all])
    zero_count = sum(1 for s in stats_all if s['fg_ratio'] == 0)
    print(f'  {lbl_name} ({len(files)} files): avg_fg={avg_fg:.4f}, zeros={zero_count}')
    for s in stats_all[:3]:
        print(f'    {s[\"name\"]}: fg={s[\"fg_pixels\"]}/{s[\"total\"]} ({s[\"fg_ratio\"]:.4f})')

print()

# =============================================
# TEST 5: 统计 edge_align 在整个数据集上的分布
# =============================================
print('='*60)
print('TEST 5: edge_align 值域分布（采样100张）')
print('='*60)

ea_values = []
import random
random.seed(42)
sampled = random.sample(sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.png')]), min(100, len(os.listdir(BACKUP_DIR))))
for f in sampled[:20]:
    img_name = os.path.splitext(f)[0]
    # Find matching image
    for d in CFG['image_dirs']:
        cand = os.path.join(d, img_name + '.jpg')
        if os.path.exists(cand):
            image = cv2.cvtColor(cv2.imread(cand), cv2.COLOR_BGR2RGB)
            break
        cand = os.path.join(d, img_name + '.png')
        if os.path.exists(cand):
            image = cv2.cvtColor(cv2.imread(cand), cv2.COLOR_BGR2RGB)
            break
    else:
        continue
    
    mask = np.array(Image.open(os.path.join(BACKUP_DIR, f)))
    mask_bool = mask >= 128
    ea = compute_edge_align(mask_bool, image)
    ea_values.append(ea)

if ea_values:
    print(f'  Sampled {len(ea_values)} images')
    print(f'  edge_align range: [{np.min(ea_values):.2f}, {np.max(ea_values):.2f}]')
    print(f'  edge_align mean: {np.mean(ea_values):.2f}, median: {np.median(ea_values):.2f}')
    print(f'  If max > 1.0, confirms [0,255] range (docstring says [0,1])')
    print(f'  gamma*edge_align range in S: [0.3*{np.min(ea_values):.1f}, 0.3*{np.max(ea_values):.1f}] = [{0.3*np.min(ea_values):.1f}, {0.3*np.max(ea_values):.1f}]')

print()
print('='*60)
print('DIAGNOSIS COMPLETE')
print('='*60)
