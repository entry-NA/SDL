"""P1: Run SAM2 refinement with V_loose prompt (1 centroid point, no box, no neg)."""
import sys, os, pickle, torch, json, gc
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import offline_sam2_refine as osm

load_coarse_label = osm.load_coarse_label
compute_iou = osm.compute_iou
SAM2Wrapper = osm.SAM2Wrapper
stage2_sam2_inference = osm.stage2_sam2_inference
CFG = osm.CFG

VLOOSE_OUTPUT = "./datasets/cache/refined_pseudo_labels_vloose"
RAW_OUTPUT = "./datasets/cache/raw_sam2_outputs"

def prompt_loose(coarse_mask, im_h, im_w, rng):
    fg_y, fg_x = np.where(coarse_mask >= 128)
    if len(fg_y) == 0:
        return None, [], []
    return None, [(int(fg_x.mean()), int(fg_y.mean()))], []

def pick_best_mask(masks, coarse):
    coarse_bool = coarse >= 128
    best_mask, best_iou = None, -1
    for m in masks:
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != coarse_bool.shape[:2]:
            mb = cv2.resize(mb.astype(np.uint8)*255,
                           (coarse_bool.shape[1], coarse_bool.shape[0]),
                           cv2.INTER_NEAREST) >= 128
        iou = compute_iou(mb, coarse_bool)
        if iou > best_iou:
            best_iou, best_mask = iou, mb
    return best_mask, best_iou

def get_image_paths(image_dirs):
    paths = []
    for d in image_dirs:
        paths.extend(str(p) for p in Path(d).glob("*.jpg"))
        paths.extend(str(p) for p in Path(d).glob("*.png"))
    return sorted(paths)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading SAM2 on", device)
    wrapper = SAM2Wrapper(device)
    rng = np.random.RandomState(42)

    image_paths = get_image_paths(CFG["image_dirs"])
    with open(os.path.join(CFG["coarse_dir"], "index.json"), "r") as f:
        index_map = json.load(f)

    os.makedirs(VLOOSE_OUTPUT, exist_ok=True)
    os.makedirs(RAW_OUTPUT, exist_ok=True)

    stats = {"total": 0, "no_points": 0, "errors": 0}
    per_image = {}

    for idx in tqdm(range(len(image_paths)), desc="V_loose refine"):
        try:
            img_path = image_paths[idx]
            image = cv2.imread(img_path)
            if image is None:
                stats["errors"] += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_h, im_w = image.shape[:2]

            coarse = load_coarse_label(
                os.path.join(CFG["coarse_dir"], index_map[str(idx)]), (im_h, im_w))

            bbox, pos, neg = prompt_loose(coarse, im_h, im_w, rng)
            if len(pos) == 0:
                stats["no_points"] += 1
                continue

            masks, scores = stage2_sam2_inference(wrapper, image, coarse, bbox, pos, neg)
            best_mask, best_iou = pick_best_mask(masks, coarse)

            img_name = os.path.splitext(os.path.basename(img_path))[0]
            out = (best_mask.astype(np.uint8)) * 255
            Image.fromarray(out, mode="L").save(
                os.path.join(VLOOSE_OUTPUT, img_name + ".png"))

            raw_save = {"masks": [m.astype(np.uint8) if m.dtype==bool else m for m in masks],
                        "scores": list(scores), "image_shape": [im_h, im_w],
                        "best_iou": float(best_iou)}
            np.savez_compressed(os.path.join(RAW_OUTPUT, img_name + ".npz"), **raw_save)

            per_image[img_name] = {
                "best_iou_vloose": float(best_iou),
                "n_masks": len(masks),
                "scores": [float(s) for s in scores],
                "image_shape": [im_h, im_w],
            }
            stats["total"] += 1

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print("\n[ERROR idx=" + str(idx) + "]", e)

        if idx % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(os.path.join(VLOOSE_OUTPUT, "vloose_stats.json"), "w") as f:
        json.dump({"stats": stats, "per_image": per_image}, f, indent=2)

    print("\nDone!", stats["total"], "refined,", stats["no_points"], "skipped,", stats["errors"], "errors")

if __name__ == "__main__":
    main()
