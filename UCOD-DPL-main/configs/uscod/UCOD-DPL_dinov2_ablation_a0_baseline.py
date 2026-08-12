cfg = dict(
    _BASE_=["UCOD-DPL_dinov2.py"],
    exp_name="UCOD-DPL_dinov2_ablation_a0_baseline_20260725_v1",
    dataset_cfg=dict(
        # This path must stay absent. BaseCODDataset then falls back to the
        # original 16x16 pseudo-label cache used by UCOD-DPL.
        refined_pseudo_label_dir=(
            "./artifacts/core_ablation/"
            "a0_baseline_20260725_v1/NO_REFINED_LABELS"
        ),
    ),
)
