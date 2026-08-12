import sys, os, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
sys.path.insert(0, '.')
from scripts.offline_sam2_refine import load_coarse_label, compute_iou, CFG

vloose_dir = './datasets/cache/refined_pseudo_labels_vloose'
gt_dir = r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\gt'
im_dir = r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\im'

gt_files = sorted(os.listdir(gt_dir))[:50]
coarse_dir = CFG['coarse_dir']
with open(os.path.join(coarse_dir, 'index.json'), 'r') as f:
    index_map = json.load(f)
paths = sorted([str(p) for p in Path(im_dir).glob('*.jpg')] + [str(p) for p in Path(im_dir).glob('*.png')])
name_to_idx = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(paths)}

vl_ious, cc_ious = [], []
for gt_file in gt_files:
    gt_name = os.path.splitext(gt_file)[0]
    gt = np.array(Image.open(os.path.join(gt_dir, gt_file)).convert('L')) >= 128
    img_path = None
    for ext in ['.jpg', '.png']:
        p = os.path.join(im_dir, gt_name + ext)
        if os.path.exists(p): img_path = p; break
    if img_path is None: continue
    image = cv2.imread(img_path)
    if image is None: continue
    im_h, im_w = image.shape[:2]
    idx = name_to_idx.get(gt_name)
    if idx is None: continue
    vloose_png = os.path.join(vloose_dir, gt_name + '.png')
    if not os.path.exists(vloose_png): continue
    vl = np.array(Image.open(vloose_png)) >= 128
    coarse = load_coarse_label(os.path.join(coarse_dir, index_map[str(idx)]), (im_h, im_w))
    cb = coarse >= 128
    if gt.shape[:2] != (im_h, im_w):
        gt = cv2.resize(gt.astype(np.uint8)*255, (im_w, im_h), cv2.INTER_NEAREST) >= 128
    vl_ious.append(compute_iou(vl, gt))
    cc_ious.append(compute_iou(cb, gt))

vl = np.array(vl_ious); cc = np.array(cc_ious)
with open('_p1_quality.txt', 'w', encoding='utf-8') as f:
    f.write('N=' + str(len(vl)) + '\n')
    f.write('Coarse IoU  mean=' + str(round(np.mean(cc),4)) + '\n')
    f.write('V_loose IoU mean=' + str(round(np.mean(vl),4)) + '\n')
    f.write('Delta       =' + str(round(np.mean(vl-cc),4)) + '\n')
    f.write('Win rate    =' + str(round(100*np.mean(vl>cc),1)) + '%\n')
