cfg = dict(
    _BASE_=['UCOD-DPL_dinov2.py'],
    exp_name='UCOD-DPL_dinov2_m0_soft',
    dataset_cfg=dict(
        refined_pseudo_label_dir=(
            './artifacts/aeem_v2/m0_controls_20260724_v1/'
            'controls/soft_coarse/refined_pseudo_labels'
        ),
    ),
)
