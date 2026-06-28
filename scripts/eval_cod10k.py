"""
Custom eval script for COD10K-v3 dataset using UCOD-DPL eval pipeline.
Usage: python -m scripts.eval_cod10k --config configs/uscod/UCOD-DPL_dinov2_cod10k_eval.py --load_from C:/eval_ckp/epoch25.pth --work_dir work_dir
"""
import os
from scripts.args import parse_train_args
from engine.runner.runner import Runner
from engine.config.config import CfgNode


def init_cfg(args) -> CfgNode:
    cfg = CfgNode.load_with_base(args.config)
    cfg = CfgNode(cfg)
    cfg.dataset_cfg.valset_cfg.keep_size = True
    cfg.train_cfg.checkpoint = args.load_from
    cfg.mode = 'eval'
    cfg.work_dir = os.path.join(
        args.work_dir,
        os.path.relpath(os.path.dirname(args.config), './configs'),
        os.path.splitext(os.path.basename(args.config))[0]
    )
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.launcher = args.launcher
    return cfg


def main():
    args = parse_train_args()
    cfg = init_cfg(args)
    dataset = cfg.dataset_cfg.valset_cfg.DATASET
    print(f"Evaluating on dataset: {dataset}")
    runner = Runner(cfg)
    result = runner.launch_val_look_twice()
    print("\n=== Evaluation Results ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
