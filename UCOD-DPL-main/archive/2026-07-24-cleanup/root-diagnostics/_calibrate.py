# === 门控阈值校准脚本 ===
# 用 raw SAM2 outputs + fixed edge_align，统计 S 值分布，输出建议阈值
import sys, os
sys.path.insert(0, "scripts")
import offline_sam2_refine as osm
import numpy as np, cv2, json, pickle

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align

RAW_DIR = "datasets/cache/raw_sam2_outputs"
coarse_dir = CFG["coarse_dir"]
with open(os.path.join(coarse_dir, "index.json")) as f:
    index_map = json.load(f)

# Build sorted image list
all_images = []
for d in CFG["image_dirs"]:
    for f in sorted(os.listdir(d)):
        if f.endswith((".jpg", ".png")):
            all_images.append((os.path.join(d, f), os.path.splitext(f)[0]))
all_images.sort(key=lambda x: x[1])

# Build: image_name -> (image_path, pkl_path, index)
name_to_img = dict((b, a) for a, b in all_images)
name_to_info = {}
for idx, (img_path, name) in enumerate(all_images):
    k = str(idx)
    if k in index_map:
        name_to_info[name] = (img_path, os.path.join(coarse_dir, index_map[k]), idx)

# Find images with raw npz
npz_names = set(f.replace(".npz", "") for f in os.listdir(RAW_DIR) if f.endswith(".npz"))

S_values_full = []  # S for best mask (buggy)
S_values_fixed = []  # S for best mask (fixed edge)
best_edge_raw = []
best_edge_norm = []
best_iou_list = []
best_score_list = []

count = 0
for name in npz_names:
    if name not in name_to_info:
        continue
    img_path, pkl_path, idx = name_to_info[name]
    
    # Load coarse
    with open(pkl_path, "rb") as fh:
        coarse_pkl = pickle.load(fh)
    if hasattr(coarse_pkl, "numpy"):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    # Load npz
    try:
        data = np.load(os.path.join(RAW_DIR, name + ".npz"), allow_pickle=True)
    except:
        continue
    masks = list(data["masks"])
    scores = list(data["scores"])
    
    # Find best mask using BOTH methods
    best_s_full = -float("inf")
    best_s_fixed = -float("inf")
    best_info_full = None
    best_info_fixed = None
    
    for i, m in enumerate(masks):
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != (h, w):
            mb = cv2.resize(mb.astype(np.uint8)*255, (w, h), cv2.INTER_NEAREST) >= 128
        
        iou_val = compute_iou(mb, coarse_bool)
        edge_raw = compute_edge_align(mb, image)
        edge_norm = edge_raw / 255.0
        
        # Buggy scoring (stage3 current): IoU * score
        s_full = iou_val * scores[i]
        
        # Fixed scoring (stage3 with 3-factor)
        s_fixed = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * edge_norm
        
        if s_full > best_s_full:
            best_s_full = s_full
            best_info_full = (i, scores[i], iou_val, edge_raw, edge_norm)
        if s_fixed > best_s_fixed:
            best_s_fixed = s_fixed
            best_info_fixed = (i, scores[i], iou_val, edge_raw, edge_norm)
    
    if best_info_fixed is None:
        continue
    
    # Stage4 S value for the best mask (FIXED edge)
    _, iou_pred, iou_ori, edge_raw, edge_norm = best_info_fixed
    S_buggy = CFG["alpha"] * iou_pred + CFG["beta"] * iou_ori + CFG["gamma"] * edge_raw
    S_fixed = CFG["alpha"] * iou_pred + CFG["beta"] * iou_ori + CFG["gamma"] * edge_norm
    
    S_values_full.append(S_buggy)
    S_values_fixed.append(S_fixed)
    best_edge_raw.append(edge_raw)
    best_edge_norm.append(edge_norm)
    best_iou_list.append(iou_ori)
    best_score_list.append(iou_pred)
    
    count += 1
    if count >= 200:
        break

print("=" * 60)
print("Gating Calibration Report (n=%d)" % len(S_values_fixed))
print("=" * 60)

arr_full = np.array(S_values_full)
arr_fixed = np.array(S_values_fixed)
edge_raw_arr = np.array(best_edge_raw)
edge_norm_arr = np.array(best_edge_norm)
iou_arr = np.array(best_iou_list)
score_arr = np.array(best_score_list)

print("\n--- Stage4 S distribution (buggy vs fixed) ---")
fmt = "%-12s %-15s %-15s"
print(fmt % ("Percentile", "S_buggy", "S_fixed"))
print("-" * 42)
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(fmt % ("%d%%" % pct, "%.4f" % np.percentile(arr_full, pct), "%.4f" % np.percentile(arr_fixed, pct)))

print("\n--- S_fixed component breakdown ---")
print("alpha*iou_pred:  mean=%.4f, range [%.4f, %.4f]" % (
    CFG["alpha"] * score_arr.mean(), CFG["alpha"] * score_arr.min(), CFG["alpha"] * score_arr.max()))
print("beta*iou_ori:    mean=%.4f, range [%.4f, %.4f]" % (
    CFG["beta"] * iou_arr.mean(), CFG["beta"] * iou_arr.min(), CFG["beta"] * iou_arr.max()))
print("gamma*edge_norm: mean=%.4f, range [%.4f, %.4f]" % (
    CFG["gamma"] * edge_norm_arr.mean(), CFG["gamma"] * edge_norm_arr.min(), CFG["gamma"] * edge_norm_arr.max()))

print("\n--- Suggested thresholds ---")
# Design thresholds based on percentiles
# Fallback: bottom 10-15% (worst SAM2 outputs)
# Full: top 60-70% (confident SAM2 outputs)
# Fusion: middle

# Criteria for FALLBACK: IoU_ori < 0.3 OR edge_norm < 0.01
# Criteria for FULL: IoU_ori > 0.6 AND edge_norm > 0.03

# But let's use S_fixed percentiles
p10 = np.percentile(arr_fixed, 10)
p25 = np.percentile(arr_fixed, 25)
p75 = np.percentile(arr_fixed, 75)
p90 = np.percentile(arr_fixed, 90)

print("S_fixed percentiles: P10=%.4f P25=%.4f P50=%.4f P75=%.4f P90=%.4f" % (p10, p25, arr_fixed.mean(), p75, p90))

# Conservative: wide fusion band
s_low_conservative = p10
s_up_conservative = p75
# Aggressive: narrow fusion band
s_low_aggressive = p25
s_up_aggressive = p50

# Count what each threshold would do
for s_low, s_up, label in [
    (0.25, 0.50, "Conservative (s_low=0.25, s_up=0.50)"),
    (0.30, 0.45, "Narrow (s_low=0.30, s_up=0.45)"),
    (0.20, 0.55, "Wide (s_low=0.20, s_up=0.55)"),
]:
    fallback = (arr_fixed < s_low).sum()
    fusion = ((arr_fixed >= s_low) & (arr_fixed <= s_up)).sum()
    full = (arr_fixed > s_up).sum()
    pct_fallback = 100.0 * fallback / len(arr_fixed)
    pct_fusion = 100.0 * fusion / len(arr_fixed)
    pct_full = 100.0 * full / len(arr_fixed)
    print("\n%s:" % label)
    print("  FALLBACK: %d (%.1f%%)" % (fallback, pct_fallback))
    print("  FUSION:   %d (%.1f%%)" % (fusion, pct_fusion))
    print("  FULL:     %d (%.1f%%)" % (full, pct_full))

print("\n--- EdgeAlign raw vs norm ---")
print("edge_raw: mean=%.1f, range [%.1f, %.1f]" % (edge_raw_arr.mean(), edge_raw_arr.min(), edge_raw_arr.max()))
print("edge_norm: mean=%.4f, range [%.4f, %.4f]" % (edge_norm_arr.mean(), edge_norm_arr.min(), edge_norm_arr.max()))
