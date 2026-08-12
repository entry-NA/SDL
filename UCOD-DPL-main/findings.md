# Findings & Decisions: AEEM v2

## Requirements
- 保留“自适应边缘感知增强机制（Adaptive Edge-Aware Enhancement Mechanism, AEEM）”名称。
- 保留两项创新叙事：离线零样本边界精修范式 + AEEM。
- 将 AEEM 从 Canny 三因子全局门控升级为语义定位校正、宽松自适应提示、边界不确定带约束和像素级置信融合。
- 先写可执行设计，再开始实现与实验。

## Research Findings
- 当前 `s_upper=999` 几乎禁用 FULL，但二值掩码的所谓 FUSION 在阈值化后退化为整图 SAM/粗标签硬切换。
- 旧 EdgeAlign 曾返回近似 `[0,255]`，导致边缘项压倒其他评分；修正后继续扫 `gamma` 只能带来小幅波动。
- 强提示容易让 SAM2 复制粗标签；松提示能产生新边界，但可能发生空间漂移。
- 现有标签级报告没有证明旧精修标签稳定提升 IoU、BF-score 或窄带边界质量。
- 现有候选实验不支持把 Canny EdgeAlign 作为核心候选选择依据。
- Excel 内统一论文行显示当前完整方案具有小幅潜力，但多数差值处于千分位，需要统一协议与多随机种子验证。
- 后续 UCOD 工作普遍转向多线索伪标签演化、可靠区域锚定、跨图原型或语义一致性，而不是继续调单一全局门控。
- 当前数据集加载器会把精修 PNG 保留为 `[0,1]` 灰度张量，而不是立即二值化。
- 当前主分割损失 `BCEWithLogitsLoss` 可以使用软目标；APM 也会将固定伪标签与教师预测继续做凸组合。
- 当前判别器路径会把伪标签阈值化，因此首版像素置信度不会影响判别器，只影响固定标签本身和主分割目标。
- 这允许 AEEM v2 MVP 只生成软精修 PNG，不立即修改训练循环；像素置信图可先作为 sidecar 用于分析和后续集成。
- `experiments/utils_metrics.py` 已具备 BF-score、软/硬粗标签加载和 bootstrap CI，可复用为新标签级评估基础，但当前数据目录是硬编码的。
- `scripts/run_vloose_refine.py` 只生成单一质心点提示，并把 raw 候选写入固定目录；新方案不能复用其固定输出路径，否则存在覆盖风险。
- 现有 raw 输出只保存 3 个松提示候选，不足以支持多提示候选共识；AEEM v2 需要按提示变体保存候选来源、分数和提示元数据。
- `BaseCODDataset` 当前从 `<cache_dir>/refined_pseudo_labels` 查找精修 PNG，而同一个 `cache_dir` 还被 `MultiCacheManager` 用于特征和原始伪标签缓存。把 `cache_dir` 指向 AEEM artifact 会改变训练输入链路并可能重建缓存，因此必须把精修标签路径独立出来。
- `refined_pseudo_label_dir` 需要沿 `DataLoaderFactory → USCODDataset/LRDataset → BaseCODDataset` 显式传递；只在 `BaseCODDataset` 内读取顶层 `dataset_cfg` 不可行，因为它收到的是 `trainset_cfg` 或 `valset_cfg`。
- 当前仓库没有正式 pytest 配置，只有实验性质的 `experiments/test_prompt_strategies.py`；Milestone 0 使用标准库 `unittest`，避免新增测试依赖。
- 历史 `experiments/utils_metrics.py` 与 `analyze_label_quality.py` 含硬编码目录且属于用户现有实验资产；新增独立参数化评估入口比原地重构更安全。
- 当前粗伪标签缓存样本为 `torch.float32`、形状 `(1,16,16)`、数值范围 `[0,1]`；控制组生成器必须兼容 torch Tensor 并显式裁剪范围。
- `CfgNode.__getattr__` 对缺失键抛出 `AttributeError`，因此 `getattr(config, 'refined_pseudo_label_dir', None)` 可安全保持旧配置兼容。
- 本地 `datasets/RefCOD/TR-CAMO` 与 `TR-COD10K` 已通过目录连接提供标准 `im/`、`gt/` 结构，控制组和评估 CLI 可使用项目内路径，不必硬编码桌面绝对路径。
- `requirement.txt` 未声明 pytest；标准库 `unittest` 是不增加依赖的最小验证方式。
- 实验 artifact 不是源码，Milestone 0 应将根级 `/artifacts/` 加入 `.gitignore`，避免 4040 张标签和 hash 清单进入版本控制。
- `m0_controls_20260724_v1` 已生成 4040 张 Hard-Coarse 与 4040 张 Soft-Coarse PNG；manifest、4040 条 audit、输入/输出 hash 数量一致，`cache_dir` 保持 `./datasets/cache`。
- 冻结配置后的 GT 诊断显示 Hard 相对 Soft：全体 `ΔIoU=+0.0001426`、`ΔBF=+0.0001297`、`ΔS=+0.0002312`，区域/边界差异只有万分位，不能解释主要实验波动。
- Hard 相对 Soft 的全体 MAE 从 `0.2981830` 降至 `0.2948042`，改善约 `0.0033788`；软插值带会明显影响像素误差，因此下游训练仍必须同时保留 Hard/Soft 表示控制组。
- Hard 相对 Soft 的 5px 窄带 IoU 反而约 `-0.0001347`，质心偏移约 `+0.0000904`，进一步说明“硬化”不是稳定的边界精修机制。
- 训练 DINOv2 特征缓存可直接复用，单样本为 `torch.float32 (768,37,37)`；Milestone 1 不需要重新运行 DINOv2。
- 系统 Python 可用 RTX 5070 Laptop GPU，但未安装 `sam2`；`test01` 环境已安装 SAM2，且 `sam2.1_hiera_tiny.pt` 已存在本地 Hugging Face 缓存。
- SAM2 `SAM2ImagePredictor.predict()` 明确要求框为 `XYXY`，旧 `offline_sam2_refine.py` 却将 `(x,y,w,h)` 直接传入。AEEM v2 必须使用正确 XYXY；旧脚本保持只读作为历史对照。
- 旧 wrapper 每个提示都会重新 `set_image()`；新多提示候选库应对同一原图只编码一次，再依次调用 predictor，避免无谓重复推理。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| AEEM v2 采用“定位可靠性路由” | 高、中、低质量粗标签不能使用同一种 SAM2 提示和采纳规则 |
| 使用多提示候选库 | 在“复制粗标签”和“无约束漂移”之间提供可选择的中间候选 |
| SAM2 修改范围限制在边界不确定带 | 保留可靠前景核心与远端背景，降低拓扑破坏和远端噪声 |
| MVP 输出 soft mask + confidence sidecar | 先用软目标保留不确定性并保持训练接口兼容；ignore map 留到 v2.1 |
| Canny 仅作为弱线索 | 伪装背景纹理边缘不能等价于目标语义边界 |
| Local-SAM 保留为辅助路由 | 当前触发比例过低，不单独承担主要创新贡献 |
| 首版不引入 ignore loss | 先利用软标签表达不确定性，避免离线算法和训练损失同时变化造成归因困难 |
| 后续 APM v2 再显式读取 confidence map | 只有离线标签级与下游结果通过后，才扩大训练侧改动范围 |
| 新评估脚本全部使用 CLI 参数传入目录 | 取消 `utils_metrics.py` 中对历史标签目录的隐式依赖 |
| 候选缓存按实验 ID 隔离 | 每个提示变体保存独立 NPZ，禁止写入历史 `raw_sam2_outputs` 固定目录 |
| 增加可选 `dataset_cfg.refined_pseudo_label_dir` | 只切换精修 PNG；`cache_dir` 保持 `./datasets/cache`，未配置时保持历史回退行为 |
| 为 Hard/Soft 控制组使用独立配置文件 | 两份配置继承同一 UCOD-DPL 基线，只改变 `refined_pseudo_label_dir`，避免训练变量混杂 |
| AEEM v2 框提示统一使用 XYXY | 遵循 SAM2 predictor 官方接口，修复旧代码把 XYWH 误当 XYXY 的提示偏差 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 历史标签目录、代码参数和 checkpoint 缺少不可变绑定 | 新设计要求每次实验保存 config、Git diff、输入清单、hash、输出目录和 checkpoint manifest |
| 旧日志记录的候选索引不一定是真正选中索引 | v2 日志记录每个候选的全部分量、真实 selected index 和最终像素决策统计 |
| 使用粗标签 IoU 选择候选会奖励复制空间偏移 | 引入教师/多层 DINO/跨增强稳定性等独立信号，粗标签一致性只作保护项 |
| 原始 pkl 与精修 PNG 经过不同加载和 resize 路径 | 增加 Hard-Coarse PNG 与 Soft-Coarse PNG 控制组，先量化表示方式本身的影响 |
| `cache_dir` 同时承担实验标签与基础缓存定位 | 不再把它指向 artifact；通过独立精修标签参数进行可审计注入 |

## Resources
- `CONTEXT.md`
- `EXPERIMENT_LOG.md`
- `docs/HANDOFF.md`
- `docs/superpowers/specs/2026-07-16-sam2-refinement-design.md`
- `docs/superpowers/specs/2026-07-18-sam2-analysis-design.md`
- `scripts/offline_sam2_refine.py`
- `experiments/output/label_quality_report.md`
- `experiments/output/p2_results.txt`
- EReCu: https://arxiv.org/abs/2603.11521
- DSS: https://arxiv.org/abs/2602.19944
- Selfment: https://arxiv.org/abs/2602.23759
- RISE: https://arxiv.org/abs/2510.18437

## Visual/Browser Findings
- SAPNet 框架使用可靠区域级标签、语义一致性增强与渐进上下文推理，而非直接依赖噪声像素标签。
- EASE 从环境原型出发反推目标，说明独立背景/语义定位信号可以缓解单图粗标签偏移。

## 2026-07-25 Verified Current State
- AEEM v2 已实现，不再停留在 Phase 7 设计阶段。核心源码位于 `aeem_v2/`，包括语义原型、提示/候选管线、边界带融合、结构校准、artifact 与评估模块。
- 当前 AEEM v2 的准确技术结构是：语义定位可靠性路由、自适应提示、多候选质量评估、边界不确定带像素级融合、结构安全回退和训练源剂量控制。
- 旧版“面积框扩张 + 截断式多掩码 + 三因子全局门控 + Local-SAM”只能作为历史方案；Local-SAM 当前不是核心贡献。
- 全量 AEEM v2 工件为 `m2_full4040_structure_20260724_v1`；最终单 seed 候选为 `m4_camo_all_cod10k_qsemantic25_20260724_v1`。
- m4 标签不是“朴素 SAM2 为底座的增量 AEEM”：它由 1760 张 AEEM（CAMO 1000 + COD10K q_semantic Top-25% 760）和 2280 张 Soft-Coarse 组成。此事实必须在解释渐进消融时保留。
- m4 四数据集结果已完成；按 `实验数据 (2).xlsx` 的论文行，20 项为 18 项严格更优、2 项显示精度持平、0 项更差，但差值多为千分位且仅单 seed。
- 2026-07-25 已完成隔离后的基线与朴素 SAM2 正式消融；结果和训练/评估日志必须纳入新 HANDOFF，旧 `docs/HANDOFF.md` 已过时。
- 全量 AEEM/AEEM 的标签级 GT 诊断相对 Soft-Coarse 在 TR-CAMO 与 TR-COD10K 均提升，但下游出现来源交互：TE-CAMO 明显受益，CHAMELEON 与 TE-COD10K 小幅退化。
- m3 来源隔离证明 TR-CAMO 与 TR-COD10K 标签贡献不是简单相加，因此 m4 使用无 GT 的 `q_semantic` 对 COD10K 做 Top-25% 剂量控制。
- 当前“完整方案”含训练源剂量控制，这既是最终配置的一部分，也是与“全量朴素 SAM2”比较时必须说明的变量；不能把完整方案错误描述为旧四模块，也不能把它无说明地视为逐模块相加。
- 全量测试重新验证为 32/32 通过；其中包括 GPU 推理前后的 CPU staged pipeline、输入顺序保持、边界带外不变、结构回退、独立标签路径和部分 PNG/pkl 混合 batch。

## 2026-07-25 Innovation 1 Label-Level Evaluation Audit
- `experiments/evaluate_aeem_labels.py` 已是显式路径、不可覆盖输出的参数化入口，可直接复用现有标签，不需要重新运行 SAM2。
- `aeem_v2/evaluation.py` 当前已计算全图 IoU、BF-score、边界 precision/recall、S-measure、MAE、最大连通域质心偏移、GT 面积比、连通域数，以及 5/10/20px 窄带 IoU/BF，并为均值提供 bootstrap CI。
- 当前比较摘要已有 `Delta IoU`、`Delta BF`、`Delta MAE`、质心偏移增量和 `Delta IoU < -0.2` 灾难性失败率。
- 当前缺口是标准 Boundary IoU 未实现；窄带 IoU 不是 Boundary IoU 的同义指标。面积误差与连通域变化也只存在于单方案行级数据，尚未进入成对比较摘要和报告。
- 正式朴素 SAM2 输入由 4028 张 PNG 和 12 张原始 pkl 回退组成；当前评估器要求每个目录独立覆盖全部 4040 个 stem，因而不能直接忠实复现这套输入。
- 冻结的 `m0_controls_20260724_v1` 已提供 4040 张 Soft-Coarse 原图尺寸 PNG。评估时以它代表原始软粗标签，并将其同时作为朴素 SAM2 缺失 12 张的显式 fallback，可保持 4028/12 的正式训练语义且不生成新标签。
- Hard-Coarse 与 Soft-Coarse 的二值指标在既有诊断中仅有万分位差异；MAE 对软插值敏感，因此创新点1主比较应使用 Soft-Coarse 作为原始标签基线，同时在报告中明确阈值化只用于二值/边界指标。
- 全量评估 `innovation1_naive_vs_softcoarse_20260725_v1` 已完成：4040 GT、Soft-Coarse 4040、Naive-SAM2 4028 + Soft-Coarse fallback 12，共 8080 行；manifest 和输入/摘要 hash 完整。
- Naive-SAM2 相对 Soft-Coarse 的全体 `Delta Boundary IoU=+0.044999`，95% bootstrap CI `[+0.041834,+0.047961]`；`Delta BF=+0.071180`，CI `[+0.067363,+0.074932]`。TR-CAMO 与 TR-COD10K 两个来源的区间也都严格大于 0，直接支持“边界精修”主张。
- 全体 `Delta IoU=-0.000954`，CI `[-0.003084,+0.001120]`，不能宣称全图区域 IoU 改善，也不能断言有稳定退化。
- 全体 `Delta MAE=+0.004199`，CI `[+0.001761,+0.006957]`，说明软像素误差平均变差；TR-CAMO 的变差明确，TR-COD10K 的区间跨 0。
- 质心偏移全体平均增加 `+0.003666`，CI `[+0.002243,+0.005181]`；`Delta IoU < -0.2` 的灾难性失败率为 `1.4604%`（TR-CAMO 2.5%，TR-COD10K 1.1184%）。
- 原始连通域误差平均大幅增加约 `+71.12`，表明 naive PNG 存在大量碎片；该指标对微小连通域高度敏感，论文解释时应明确是 raw component count，不能替代有效连通域/额外结构质量分析。
- 逐图分布显示全体 Boundary IoU 提升率 78.02%、BF 提升率 61.46%、IoU 提升率 59.65%、MAE 改善率 65.74%；IoU 和 MAE 的中位数略有改善，但少量严重失败使均值 IoU 无显著变化、均值 MAE 变差。
- 创新点1可辩护表述是“冻结 SAM2 显著增强标签边界质量，并在下游四数据集带来一致 F_MAX 提升”；不可扩大为“朴素 SAM2 全面提升标签区域/结构质量”。

## 2026-07-25 Innovation 2 Attribution Audit
- 当前最低成本且尚未完成的直接诊断是：在相同 4040 个训练样本上比较全量 AEEM v2 与正式朴素 SAM2（4028 PNG + 12 Soft-Coarse fallback）。
- 该比较能验证 AEEM v2 是否相对 naive 修复区域误差、质心漂移、碎片与灾难失败，同时是否保留边界收益；它不需要重新运行 SAM2 或训练。
- 即使标签级 AEEM v2 优于 naive，也不能把它当作下游 `完整方案 vs 朴素 SAM2` 的严格增量归因，因为 m4 使用 1760 AEEM + 2280 Soft-Coarse，而朴素方案使用 4028 naive + 12 fallback。
- 只有标签级比较完成后，才有依据决定是否值得增加一个“同底座、只替换可靠部分”的额外训练控制组。
- 全量同样本评估 `innovation2_m2_vs_naive_20260725_v1` 已完成：AEEM v2 相对 naive 的全体 `Delta IoU=+0.028078`，95% CI `[+0.025808,+0.030247]`；`Delta MAE=-0.013335`，CI `[-0.016113,-0.010863]`。
- AEEM v2 同时改善质心偏移 `-0.004527`，并把 raw 连通域数量误差平均降低 `65.60`；两项 CI 均严格小于 0，证明结构安全机制确实缓解 naive 的漂移与碎片。
- 代价是相对 naive 的 `Delta Boundary IoU=-0.017168`、`Delta BF=-0.046053`，CI 均严格小于 0。AEEM v2 不是在所有边界指标上超过 naive，而是牺牲部分极锐边界收益换取区域与结构可靠性。
- 结合绝对值，AEEM v2 的 Boundary IoU `0.111681` 和 BF `0.086987` 仍高于 Soft-Coarse 的 `0.083851/0.061860`；因此 AEEM v2 保留了正边界增益，而非简单回退粗标签。
- 保存的 m2 与正式朴素 SAM2 `config.yaml` 已逐字段核对：除 `exp_name`、`work_dir/log_path`、`checkpoint` 和 `refined_pseudo_label_dir` 这些实验身份/输入输出路径外，数据集、DINOv2-base、518 输入、batch 16、25 epoch、优化参数、EMA、APM、Look-Twice 与评估设置一致。因此这两次既有训练可作为“同训练底座、只更换伪标签输入”的下游控制，不需要重复训练 m2。
- m4 的 1760 个可靠替换样本与正式朴素 SAM2 的 12 个缺失 PNG 已做集合交叉：缺失样本中 3 个来自 TR-CAMO，均落在 m4 的 AEEM 侧；另外 9 个来自 TR-COD10K，均落在 m4 的 Soft-Coarse 侧。若生成“以朴素 SAM2 为底座、仅替换可靠样本”的额外控制，规则应固定为：1760 个选中样本优先使用现有 AEEM 标签（包括那 3 个无 naive PNG 的 TR-CAMO 样本），其余 2280 个样本优先使用 naive PNG，只有其中缺失的 9 个 TR-COD10K 样本回退 Soft-Coarse。最终构成应为 1760 AEEM + 2271 naive + 9 Soft-Coarse，而不是 1760 AEEM + 2268 naive + 12 Soft-Coarse。

## 2026-07-25 Reliable Replacement Control Implementation
- 现有 `compose_label_artifact()` 只支持 AEEM/Soft 两种来源；把 naive 目录冒充 `soft_dir` 会在 9 个缺失 PNG 上失败，即使先补齐也会把 2271 个 naive 错记为 soft，破坏来源审计。
- 验证控制必须使用隔离的新实验准备入口，直接读取冻结 m4 `audit.jsonl` 的 `source_type=aeem` 集合作为 1760 个可靠样本；非可靠样本按 naive PNG 优先、Soft-Coarse 仅缺失回退。
- 新入口只复用现有 artifact/hash 工具，不修改现有组合器、训练主干或标签生成算法；公开验收行为是三来源计数准确、已有目标拒绝覆盖、复制前后逐文件 hash 相同。
- 新工件 `innovation2_reliable_on_naive_20260725_v1` 已完成：4040 输入、4040 PNG、4040 条 audit；来源为 1760 AEEM、2271 朴素 SAM2、9 Soft-Coarse fallback，复制 hash 不一致为 0。
- 新工件 `output_hashes.json` 的文件 SHA256 为 `a2e1d827fd75ffc3c69c4207d403b7e4970ed0957f453d06ee701cb55730d116`。
- 独立训练目录 `work_dir_validation_20260725` 已完成 25 epoch，epoch5/10/15/20/25 均存在；训练没有覆盖正式消融或 m4 输出。
- 可靠替换控制四数据集宏平均为 E_MEAN 0.910000、F_MAX 0.811825、SMeasure 0.837200、MAE 0.044400、WFM 0.790775。
- 相对朴素 SAM2，宏平均 SMeasure `+0.002575`、MAE `-0.000825`、E_MEAN `+0.004050`、F_MAX `-0.005600`、WFM `+0.000050`；SMeasure 和 MAE 均仅 2/4 数据集方向一致。
- 按预注册规则结论必须是“混合结果”：主要终点宏平均方向正确，但跨数据集一致性不足且 F_MAX 下降。单 seed 只能称方向性证据，不能称全面提升、稳健或统计显著。

## 2026-07-25 Safe Disk Cleanup Audit
- 项目内最大目录是 `datasets/cache/features_cache/dinov2`，约 41.18 GiB、10518 个特征文件。正式训练直接复用这些 DINOv2 特征；删除会迫使老师重算并增加环境/下载失败风险，不能作为无关缓存清理。
- `artifacts/aeem_v2` 约 1.94 GiB；必须保留 m2、m4、m0、可靠替换工件及两个创新点诊断。m1/validation/smoke/structure replay 和 m3 来源隔离是可重建中间产物，不是正式训练输入。
- `work_dir` 与验证 work_dir 合计约 203 MiB，其中每个评估的 `preds` 约 17 MiB，可由保存的 epoch25 重新评估生成；保留 checkpoint、config 和 `eval0.log` 即不改变结果。
- `weights` 约 1.35 GiB，存在四个约 330 MiB 的 DINOv2 权重文件。删除前必须核对 hash、链接关系和实际加载路径。
- 明确可删的小项包括 Python `__pycache__`、`.pytest_cache` 和 `datasets/cache/refined_pseudo_labels_broken`；不会触碰原始 4040 pkl、4028 朴素 SAM2、raw SAM2、features cache、正式标签或 checkpoint。
