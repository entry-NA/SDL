# === 诊断脚本 1 FIXED: edge_align + gating ===
import sys, os
sys.path.insert(0, "scripts")
import offline_sam2_refine as osm
import numpy as np, cv2, json, pickle
from PIL import Image

CFG = osm.CFG
compute_edge_align = osm.compute_edge_align
compute_iou = osm.compute_iou
s_low = CFG["s_lower"]
s_up = CFG["s_upper"]

coarse_dir = CFG["coarse_dir"]
with open(os.path.join(coarse_dir, "index.json")) as f:
    index_map = json.load(f)

image_dirs = CFG["image_dirs"]
test_images = []
for d in image_dirs:
    for f in sorted(os.listdir(d))[:2]:
        if f.endswith((".jpg", ".png")):
            test_images.append(os.path.join(d, f))

print("=" * 60)
print("TEST A: compute_edge_align actual return range")
print("=" * 60)

for img_path in test_images[:3]:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    name = os.path.splitext(os.path.basename(img_path))[0]
    
    found = False
    for k, v in index_map.items():
        if name in v:
            with open(os.path.join(coarse_dir, v), "rb") as fh:
                coarse_pkl = pickle.load(fh)
            found = True
            break
    if not found:
        continue
    
    if hasattr(coarse_pkl, "numpy"):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    ea = compute_edge_align(coarse_bool, image)
    zero_mask = np.zeros((h, w), dtype=bool)
    ea_zero = compute_edge_align(zero_mask, image)
    ones_mask = np.ones((h, w), dtype=bool)
    ea_ones = compute_edge_align(ones_mask, image)
    
    print("%s:" % os.path.basename(img_path))
    print("  edge_align(coarse)  = %.2f  (docstring says [0,1])" % ea)
    print("  edge_align(all_zero) = %.2f" % ea_zero)
    print("  edge_align(all_ones) = %.2f" % ea_ones)
    
    iou_pred = 0.95
    iou_ori = compute_iou(coarse_bool, coarse_bool)
    S_buggy = CFG["alpha"] * iou_pred + CFG["beta"] * iou_ori + CFG["gamma"] * ea
    S_fixed = CFG["alpha"] * iou_pred + CFG["beta"] * iou_ori + CFG["gamma"] * (ea / 255.0)
    
    buggy_label = "FULL" if S_buggy > s_up else ("FALLBACK" if S_buggy < s_low else "FUSION")
    fixed_label = "FULL" if S_fixed > s_up else ("FALLBACK" if S_fixed < s_low else "FUSION")
    
    print("  S_buggy = %.2f -> %s" % (S_buggy, buggy_label))
    print("  S_fixed = %.4f -> %s" % (S_fixed, fixed_label))
    print()

print("=" * 60)
print("TEST B: edge_align value distribution (50 samples)")
print("=" * 60)

ea_list = []
for d in image_dirs:
    for f in sorted(os.listdir(d))[:25]:
        if not f.endswith((".jpg", ".png")):
            continue
        name = os.path.splitext(f)[0]
        png_path = "datasets/cache/refined_pseudo_labels/%s.png" % name
        if not os.path.exists(png_path):
            continue
        image = cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(png_path))
        mask_bool = mask >= 128
        ea = compute_edge_align(mask_bool, image)
        ea_list.append(ea)

if ea_list:
    arr = np.array(ea_list)
    print("Samples: %d" % len(ea_list))
    print("Range: [%.2f, %.2f]" % (arr.min(), arr.max()))
    print("Mean: %.2f, Median: %.2f, Std: %.2f" % (arr.mean(), np.median(arr), arr.std()))
    print(">1.0: %d/%d" % ((arr > 1.0).sum(), len(ea_list)))
    print(">10: %d/%d" % ((arr > 10).sum(), len(ea_list)))
    print(">100: %d/%d" % ((arr > 100).sum(), len(ea_list)))
    print()
    if arr.max() > 10:
        print("CONFIRMED: edge_align returns [0,255] range, NOT [0,1]")
        print("gamma*edge = [0, %.1f] >> thresholds [%.1f, %.1f] -> gating short-circuited" % (0.3*arr.max(), s_low, s_up))
    else:
        print("UNEXPECTED: edge_align max = %.2f < 10. Check compute_edge_align" % arr.max())
