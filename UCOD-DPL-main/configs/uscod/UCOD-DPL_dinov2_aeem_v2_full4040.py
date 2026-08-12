import os


experiment_id = os.environ.get(
    "AEEM_EXPERIMENT_ID",
    "m2_full4040_structure_20260724_v1",
)

cfg = dict(
    _BASE_=["UCOD-DPL_dinov2.py"],
    exp_name=f"UCOD-DPL_dinov2_aeem_v2_{experiment_id}",
    dataset_cfg=dict(
        refined_pseudo_label_dir=(
            f"./artifacts/aeem_v2/{experiment_id}/refined_pseudo_labels"
        ),
    ),
)
