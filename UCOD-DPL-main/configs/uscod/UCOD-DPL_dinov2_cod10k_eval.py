cfg = dict(
    _BASE_ = ['UCOD-DPL_dinov2.py'],
    dataset_cfg = dict(
        cache_dir = './datasets/cache/cod10kv3',
        dataset_dir = 'C:/Users/23991/Desktop/archive野生动物伪装目标数据集',
        valset_cfg = dict(
            DATASET = 'COD10K-v3/Test',
            require_label = True,
        ),
        trainset_cfg = dict(
            DATASET = 'COD10K-v3/Train',
            require_label = False,
        ),
    )
)
