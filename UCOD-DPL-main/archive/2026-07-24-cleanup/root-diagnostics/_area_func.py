# ==============================================================================
# Area Conservation Calibration (between Stage 3 and Stage 4)
# ==============================================================================
def area_consistency_calibration(sam_mask, coarse_mask, candidate_masks=None,
                                  tau_low=0.70, tau_high=1.50):
    """Constrain SAM2 mask area to not significantly deviate from coarse label.
    CPI assembly + controlled dilation for tight masks; coarse clipping for loose masks."""
    coarse_bool = coarse_mask >= 128
    area_sam = int(sam_mask.sum())
    area_coarse = int(coarse_bool.sum())
    if area_coarse == 0:
        return sam_mask
    r = area_sam / area_coarse
    if r < tau_low:
        # --- SAM2 too tight: CPI assembly ---
        if candidate_masks is not None and len(candidate_masks) > 0:
            assembled = np.zeros_like(sam_mask, dtype=bool)
            for cm in candidate_masks:
                cmb = cm.astype(bool) if cm.dtype != bool else cm
                if cmb.shape != sam_mask.shape:
                    cmb = cv2.resize(cmb.astype(np.uint8) * 255,
                                     (sam_mask.shape[1], sam_mask.shape[0]),
                                     cv2.INTER_NEAREST) >= 128
                overlap = np.logical_and(cmb, coarse_bool)
                if overlap.sum() / max(cmb.sum(), 1) >= 0.3:
                    if cmb.sum() >= area_sam:
                        assembled = np.logical_or(assembled, cmb)
            if assembled.sum() >= area_sam:
                sam_mask = assembled
        # --- If still insufficient: controlled dilation ---
        deficit = area_coarse - int(sam_mask.sum())
        if deficit > 0 and int(sam_mask.sum()) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            coarse_dilated = cv2.dilate(coarse_bool.astype(np.uint8), kernel, iterations=2).astype(bool)
            compensation = np.logical_and(dilated, coarse_dilated) & (~sam_mask)
            if compensation.sum() > 0:
                sam_mask = np.maximum(sam_mask, compensation)
    elif r > tau_high:
        # --- SAM2 too loose: clip with coarse ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        coarse_dilated = cv2.dilate(coarse_bool.astype(np.uint8), kernel, iterations=2).astype(bool)
        sam_mask = np.logical_and(sam_mask, coarse_dilated)
    return sam_mask

