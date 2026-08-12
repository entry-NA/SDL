# SAM2 精修 UCOD-DPL — 基于 SAM2 的无监督伪装目标检测伪标签精修

## 项目概述

本项目基于 CVPR 2025 Highlight 论文 **UCOD-DPL**（Unsupervised Camouflaged Object Detection via Dynamic Pseudo-label Learning），引入完全冻结的 **SAM2**（Segment Anything Model 2）作为离线零样本伪标签精修器，对 DINOv2 生成的低分辨率粗伪标签进行边界增强，提升 UCOD-DPL 第一阶段训练性能。

核心链路：

```
DINOv2 16×16 粗伪标签 → SAM2 原图分辨率边界精修 → 精修标签喂给 UCOD-DPL 训练 → 4 个 COD 基准评估
```

### 两项创新

1. **离线零样本边界精修范式**：冻结 SAM2 作为离线边界专家，置于 APM 之前，改变伪标签输入源。修改仅约 30 行，不侵入 UCOD-DPL 网络结构。
2. **自适应边缘感知增强机制（AEEM v2）**：从旧版全局门控升级为语义定位校正、自适应提示路由、多候选质量评估、边界不确定带像素级融合、结构安全回退和训练源剂量控制。

---

## 项目结构

```
SAM2精修UCOD-DPL/
├── README.md                          ← 本文件
├── UCOD-DPL-main/                     ← 主项目目录
│   ├── README.md                      ← UCOD-DPL 上游原始 README
│   ├── CONTEXT.md                     ← 术语定义与概念关系
│   ├── task_plan.md                   ← 任务计划（12 个 Phase）
│   ├── progress.md                    ← 进度日志（Phase 1-12 详细记录）
│   ├── findings.md                    ← 研究结论与技术决策
│   ├── EXPERIMENT_LOG.md              ← 旧版实验记录（历史参考）
│   ├── requirement.txt                ← Python 依赖
│   ├── .gitignore
│   │
│   ├── aeem_v2/                       ← **AEEM v2 核心源码**
│   │   ├── __init__.py
│   │   ├── semantic.py                ← DINOv2 语义原型、定位可靠性
│   │   ├── refinement.py              ← 提示路由、候选质量、边界融合
│   │   ├── sam2_adapter.py            ← SAM2 适配器（同图单次编码、XYXY 修正）
│   │   ├── structure.py               ← 碎片清理、连通骨架保护
│   │   ├── topology.py                ← 拓扑结构风险回退
│   │   ├── pipeline.py                ← GPU/CPU 重叠流水线
│   │   ├── composition.py             ← 多来源标签组合与 Top-fraction 选择
│   │   ├── artifacts.py               ← 不可覆盖实验工件管理
│   │   ├── evaluation.py              ← 标签质量评估（IoU/BF/Boundary IoU/MAE/质心）
│   │   ├── controls.py                ← 控制组生成
│   │   └── dataset.py                 ← 数据支持
│   │
│   ├── configs/                       ← 训练/评估配置
│   │   ├── uscod/
│   │   │   ├── UCOD-DPL_dinov2.py                    ← 标准训练配置
│   │   │   ├── UCOD-DPL_dinov2_ablation_a0_baseline.py  ← 基线消融
│   │   │   ├── UCOD-DPL_dinov2_ablation_a1_naive_sam2.py ← 朴素 SAM2 消融
│   │   │   ├── UCOD-DPL_dinov2_aeem_v2_full4040.py   ← AEEM v2 完整方案
│   │   │   ├── UCOD-DPL_dinov2_m0_hard.py            ← Hard-Coarse 控制组
│   │   │   ├── UCOD-DPL_dinov2_m0_soft.py            ← Soft-Coarse 控制组
│   │   │   └── ...
│   │   └── dataset/cod4040.py
│   │
│   ├── data/                          ← 数据加载
│   │   ├── datasets/
│   │   │   ├── base_dataset.py        ← 基类（独立精修标签路径）
│   │   │   ├── uscod_dataset.py       ← USCOD 数据集
│   │   │   ├── lr_dataset.py          ← Look-Twice 数据集
│   │   │   ├── cache_manager.py       ← 多级缓存管理
│   │   │   └── dataloader_utils.py    ← DataLoader 工厂
│   │   └── utils/
│   │       └── feature_extractor.py   ← DINOv2 特征提取
│   │
│   ├── engine/                        ← 训练引擎
│   │   ├── runner/
│   │   │   ├── loop_UCOD_DPL.py       ← UCOD-DPL 训练主循环
│   │   │   ├── loop_CORAL.py          ← CORAL 训练循环
│   │   │   └── runner.py              ← 训练 Runner
│   │   ├── config/                    ← 配置系统
│   │   ├── registry/                  ← 模型注册
│   │   └── utils/                     ← 工具（metrics/fileio/logger）
│   │
│   ├── models/                        ← 模型定义
│   │   ├── backbones/                 ← 骨干网络（DINO/DINOv2/PVT/Swin/ResNet）
│   │   ├── modules/
│   │   │   ├── full_model.py          ← UCOD-DPL 完整模型
│   │   │   ├── DBA.py                 ← 双分支对抗解码器
│   │   │   ├── ASR.py / CSF.py / HRE.py  ← 各模块
│   │   │   └── ocm.py                 ← CUDA 编译扩展（CPU 不可用）
│   │   └── uscod.py
│   │
│   ├── scripts/                       ← 运行脚本
│   │   ├── train.py                   ← 训练入口
│   │   ├── eval.py / eval_cod10k.py   ← 评估入口
│   │   ├── offline_sam2_refine.py     ← 旧版 SAM2 精修（历史对照）
│   │   ├── run_aeem_v2_mvp.py         ← AEEM v2 MVP 运行入口
│   │   ├── compose_aeem_v2_labels.py  ← AEEM v2 标签组合
│   │   ├── prepare_aeem_controls.py   ← 控制组准备
│   │   ├── run_core_ablation_train.ps1 ← 核心消融训练
│   │   ├── run_core_ablation_eval.ps1  ← 核心消融评估
│   │   ├── run_aeem_v2_train.ps1      ← AEEM v2 训练
│   │   ├── run_aeem_v2_eval.ps1       ← AEEM v2 评估
│   │   ├── launch_train_first_stage.sh ← 上游训练启动脚本
│   │   ├── launch_val_first_stage.sh   ← 上游评估启动脚本
│   │   └── ...
│   │
│   ├── experiments/                   ← 实验脚本与数据
│   │   ├── run_ablation.py            ← 消融实验编排
│   │   ├── evaluate_aeem_labels.py    ← 参数化标签质量评估
│   │   ├── analyze_label_quality.py   ← 标签质量分析
│   │   ├── prepare_reliable_replacement_control.py ← 可靠替换控制
│   │   ├── output/                    ← 实验输出
│   │   └── ablation_flags_*.json      ← 消融实验配置
│   │
│   ├── tests/                         ← 测试（38 tests）
│   │   ├── test_aeem_v2_milestone0.py
│   │   ├── test_aeem_v2_milestone1.py
│   │   ├── test_aeem_v2_milestone2.py
│   │   ├── test_aeem_v2_pipeline.py
│   │   ├── test_aeem_v2_composition.py
│   │   ├── test_reliable_replacement_control.py
│   │   └── test_multiseed_summary.py
│   │
│   ├── docs/                          ← 文档
│   │   ├── HANDOFF.md                 ← **会话交接文档（新会话必读）**
│   │   ├── AEEM_V2_QSEMANTIC25_FINAL_CONFIG.md ← AEEM v2 最终配置
│   │   ├── AEEM_V2_UNIFIED_EXPERIMENT_MANUAL.md ← 统一实验操作手册
│   │   ├── AEEM_CORE_ABLATION_PROTOCOL.md ← 核心消融协议
│   │   ├── INNOVATION1_NAIVE_SAM2_BOUNDARY_EVIDENCE.md ← 创新1证据
│   │   ├── INNOVATION2_ATTRIBUTION_PROTOCOL.md ← 创新2归因协议
│   │   └── superpowers/               ← 设计文档与规划
│   │
│   ├── weights/                       ← 模型权重
│   │   ├── UCOD_DPL_dinov2.safetensors    ← UCOD-DPL 解码器权重
│   │   ├── UCOD_DPL_dinov1.safetensors
│   │   ├── CORAL_dinov2.safetensors
│   │   └── CORAL_dinov1.safetensors
│   │
│   ├── datasets/                      ← 数据集（需自行下载）
│   │   └── cache/
│   │       ├── pseudo_label_cache/    ← 原始 DINOv2 粗伪标签（4040 .pkl）
│   │       ├── naive_sam2_labels/     ← 朴素 SAM2 标签（4028 PNG）
│   │       ├── raw_sam2_outputs/      ← 历史 SAM2 raw 输出
│   │       └── features_cache/        ← DINOv2 特征缓存（~41 GB）
│   │
│   ├── artifacts/                     ← 实验工件（不可覆盖）
│   │   └── aeem_v2/
│   │       ├── m2_full4040_structure_20260724_v1/  ← 全量 AEEM v2
│   │       ├── m4_camo_all_cod10k_qsemantic25_20260724_v1/ ← 最终方案
│   │       ├── m0_controls_20260724_v1/             ← 控制组
│   │       └── innovation*/                         ← 创新证据工件
│   │
│   ├── work_dir/                      ← 训练输出
│   └── archive/                       ← 历史文件归档
│
├── AEEM_v2_GitHub首次复现_消融_完整模型_验证_多随机种子_统一操作手册.docx
├── AEEM_v2_消融_完整模型_验证实验_统一操作手册.docx
├── _doc_work/                         ← 文档生成工具
└── .agents/                           ← Agent 配置
```

---

## 环境要求

- **Python**: 3.9（推荐 conda 环境 `test01`）
- **GPU**: NVIDIA GPU（SAM2 推理 + UCOD-DPL 训练均需 CUDA）
- **CUDA**: 需支持 PyTorch 2.x
- **关键依赖**: PyTorch、SAM2、OpenCV、HuggingFace Transformers、PEFT、Accelerate

安装依赖：

```bash
conda create -n test01 python==3.9 -y && conda activate test01
pip install -r UCOD-DPL-main/requirement.txt
# SAM2 需额外安装：
pip install sam2
```

---

## 快速开始

### 1. 数据集准备

从 [GoogleDrive](https://drive.google.com/drive/folders/19MaIVAcqr8sIv0R1hIq7MZhPqO-9_s8v) 下载数据集放入 `UCOD-DPL-main/datasets/`，目录结构：

```
datasets/
├── RefCOD/
│   ├── TR-CAMO/im/  (1000 张训练图)
│   ├── TR-COD10K/im/ (3040 张训练图)
│   ├── TE-CAMO/im/gt/
│   ├── TE-COD10K/im/gt/
│   ├── CHAMELEON/im/gt/
│   └── NC4K/im/gt/
└── cache/
    ├── pseudo_label_cache/TR-CAMO+TR-COD10K/  (4040 .pkl)
    └── features_cache/                         (自动生成)
```

### 2. 伪标签生成（训练前置条件）

```bash
conda activate test01
cd UCOD-DPL-main
python generate_pseudo_label.py \
  --dataset 'TR-CAMO+TR-COD10K' \
  --image_path 'datasets/RefCOD/{}/im' \
  --cache_path 'datasets/cache'
```

### 3. 训练

```bash
# 基线训练（A0）
bash scripts/launch_train_first_stage.sh -c configs/uscod/UCOD-DPL_dinov2.py

# 或使用 PowerShell 脚本（Windows）
powershell.exe -File scripts/run_core_ablation_train.ps1 -Group a0
```

### 4. 评估

```bash
bash scripts/launch_val_first_stage.sh \
  -c configs/uscod/UCOD-DPL_dinov2.py \
  -m path/to/checkpoint.pth
```

### 5. 运行测试

```powershell
conda activate test01
Set-Location UCOD-DPL-main
python -m unittest discover -s tests -v
# 当前: 38 tests passed
```

---

## AEEM v2 技术架构

### 整体流程

```
DINOv2 语义定位可靠性
  → High / Medium / Low 路由
  → 自适应点提示、弱框、安全背景负点
  → SAM2 多提示多候选
  → q_semantic / q_stability / q_edge / q_safety 四维质量评估
  → 边界不确定带内像素级残差融合
  → 可靠前景核心、远端背景、连通骨架保护
  → 结构风险回退 Soft-Coarse
  → 训练源剂量控制
```

### 关键设计规则

| 规则 | 说明 |
|------|------|
| Low 路由不调用 SAM2 | 直接回退粗标签 |
| SAM2 仅修改边界不确定带 | 可靠前景核心与远背景受保护 |
| 候选质量 ≠ 融合权重 | 两个独立概念，不再使用全局 S |
| GT 仅用于冻结后诊断 | 不参与候选选择或阈值调参 |
| 最大连通域增长 ≤ 1 | 最大额外结构质量 ≤ 0.05 |
| 精修标签独立目录 | 不改变 DINO 特征和原始伪标签缓存 |

### 实验命名约定

- **基线（Baseline / A0）**：UCOD-DPL 原始训练
- **朴素 SAM2（Naive SAM2 / A1）**：4028 张 SAM2 PNG + 12 张 pkl 回退
- **完整方案（Full / m4）**：1760 AEEM v2 + 2280 Soft-Coarse，含 COD10K `q_semantic` Top-25% 剂量控制

---

## 当前实验结果

### 三组主要结果（5 指标 × 4 数据集）

| 方案 | CHAMELEON | TE-CAMO | TE-COD10K | NC4K |
|------|-----------|---------|-----------|------|
| 论文参考 SMeasure | 0.880 | 0.793 | 0.781 | 0.825 |
| 本地基线 SMeasure | 0.8630 | 0.7932 | 0.8344 | 0.8507 |
| 朴素 SAM2 SMeasure | 0.8661 | 0.7859 | 0.8367 | 0.8498 |
| 完整方案 SMeasure | 0.8648 | 0.7939 | 0.8344 | 0.8513 |

完整方案相对论文：18 项严格更优、2 项持平、0 项下降（单 seed 42）。

### 已知局限

- 当前仅单 seed（42），需多 seed 验证统计稳定性
- 朴素 SAM2 在四数据集上 F_MAX 全部提升（+0.0086），但完整方案在 CHAMELEON/TE-COD10K 不如朴素 SAM2
- COD10K 碎片问题是跨数据集泛化的主要瓶颈

---

## 重要保护规则

以下路径和数据**不可修改、删除或覆盖**：

- `datasets/cache/pseudo_label_cache/` — 原始粗伪标签
- `datasets/cache/naive_sam2_labels/` — 朴素 SAM2 标签
- `artifacts/aeem_v2/m2_*/` — 全量 AEEM v2 工件
- `artifacts/aeem_v2/m4_*/` — 最终完整方案
- `work_dir/uscod/*/` — 三组正式 checkpoint 和 eval0.log

### 新实验必须

1. 使用新 experiment ID + 新目录
2. `cache_dir` 始终为 `./datasets/cache`
3. 仅通过 `refined_pseudo_label_dir` 切换实验标签
4. 保存 manifest、Git diff、输入/输出 hash
5. 先通过单元测试 + 标签级诊断，再训练

---

## 文档导航

| 需求 | 文档 |
|------|------|
| 新会话接手 | `UCOD-DPL-main/docs/HANDOFF.md` |
| 训练/评估操作 | `UCOD-DPL-main/docs/AEEM_V2_UNIFIED_EXPERIMENT_MANUAL.md` |
| AEEM v2 最终配置 | `UCOD-DPL-main/docs/AEEM_V2_QSEMANTIC25_FINAL_CONFIG.md` |
| 消融实验协议 | `UCOD-DPL-main/docs/AEEM_CORE_ABLATION_PROTOCOL.md` |
| 创新点1证据 | `UCOD-DPL-main/docs/INNOVATION1_NAIVE_SAM2_BOUNDARY_EVIDENCE.md` |
| 创新点2归因 | `UCOD-DPL-main/docs/INNOVATION2_ATTRIBUTION_PROTOCOL.md` |
| 旧版实验记录 | `UCOD-DPL-main/EXPERIMENT_LOG.md` |
| 任务计划 | `UCOD-DPL-main/task_plan.md` |
| 进度日志 | `UCOD-DPL-main/progress.md` |
| 研究发现 | `UCOD-DPL-main/findings.md` |
| 术语定义 | `UCOD-DPL-main/CONTEXT.md` |

---

## 上游论文引用

```bibtex
@inproceedings{yan2025ucod,
  title={UCOD-DPL: Unsupervised Camouflaged Object Detection via Dynamic Pseudo-label Learning},
  author={Yan, Weiqi and Chen, Lvhai and Kou, Huaijia and Zhang, Shengchuan and Zhang, Yan and Cao, Liujuan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30365--30375},
  year={2025}
}
```

- 论文: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_UCOD-DPL_Unsupervised_Camouflaged_Object_Detection_via_Dynamic_Pseudo-label_Learning_CVPR_2025_paper.pdf)
- arXiv: [2506.07087](https://arxiv.org/abs/2506.07087)
- 项目主页: [Project Page](https://heartfirey.top/project_page/UCOD-DPL/)
