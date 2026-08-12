# Progress Log: AEEM v2

## Session: 2026-07-24

### Phase 1: 需求与事实冻结
- **Status:** complete
- **Started:** 2026-07-24
- Actions taken:
  - 对齐用户提供的两份 Word 文档与 Excel 实验数据。
  - 审计当前 SAM2 精修代码、实验日志、HANDOFF 和标签级分析。
  - 确认保留 AEEM 名称，重构其技术内涵。
  - 确认停止继续盲扫 `s_lower`、`s_upper` 和 `gamma`。
- Files created/modified:
  - `task_plan.md`（创建）
  - `findings.md`（创建）
  - `progress.md`（创建）

### Phase 2: 旧设计与实现约束核对
- **Status:** complete
- Actions taken:
  - 核对旧四阶段 AEEM 设计、HANDOFF 和当前门控实现。
  - 核对 `base_dataset.py` 的灰度 PNG 加载路径。
  - 核对 `loop_UCOD_DPL.py` 的主损失、判别器二值化和 APM 融合逻辑。
  - 确认 MVP 可先保持训练接口不变，用软 PNG 表达像素级不确定性。
- Files created/modified:
  - `task_plan.md`（更新）
  - `findings.md`（更新）
  - `progress.md`（更新）

### Phase 3: AEEM v2 设计文档
- **Status:** complete
- Actions taken:
  - 已确定目标架构与 MVP/后续训练集成的分层边界。
  - 核对现有 BF-score、bootstrap CI 和软/硬粗标签加载工具。
  - 核对 V_loose 提示与 raw 候选缓存格式，确认新方案必须参数化输出并隔离实验目录。
  - 新增统一领域术语表，区分定位可靠性、候选质量和像素置信度。
  - 完成四个论文模块、六个工程阶段、数据契约、回退策略、消融矩阵与验收门槛。
  - 修正设计初稿中的缓存耦合：增加独立 `refined_pseudo_label_dir`，不再替换 `cache_dir`。
- Files created/modified:
  - `docs/superpowers/specs/2026-07-24-aeem-v2-design.md`（创建并完成审查）
  - `CONTEXT.md`（创建）

### Phase 4: 设计审查
- **Status:** complete
- Actions taken:
  - 确认候选质量只用于候选选择，不直接触发整图 SAM2 采纳。
  - 确认像素置信度只在边界不确定带内生效，可靠前景核心与远端背景受保护。
  - 确认生成与无标签阈值冻结过程不读取 GT；GT 仅用于冻结后的诊断。
  - 确认 SLC、APR、BUB、PCF 与 Local-SAM 均有独立消融位置。
  - 确认实验标签路径与 DINO 特征、原始伪标签缓存解耦。

### Phase 5: 启动交付
- **Status:** complete
- Actions taken:
  - 用户已确认开始实施。
  - 冻结本轮范围：只做 Milestone 0，不运行全量 SAM2 或完整训练。

### Phase 6: Milestone 0 实现
- **Status:** complete
- Actions taken:
  - 审计 `BaseCODDataset`、DataLoaderFactory、USCOD/LR Dataset 与 `MultiCacheManager` 的参数链路。
  - 确认 `base_dataset.py` 已有用户修改，后续只扩展路径注入，不回退现有 PNG 优先加载逻辑。
  - 确认控制组需保存为原图尺寸 PNG，才能复用与 SAM2 标签相同的训练加载路径。
  - 确认新增参数化评估入口，不原地改写含历史硬编码的实验脚本。
  - 确认粗标签 PKL 为 `(1,16,16)` 的 float32 torch Tensor，范围 `[0,1]`。
  - 确认系统 Python 3.9.13 与 torch 2.8.0 可用于 Milestone 0 测试。
  - 实现独立 `refined_pseudo_label_dir` 参数链路，保持 `cache_dir` 不变。
  - 实现不可覆盖 artifact、Hard/Soft-Coarse 控制组、Git 状态与输入/输出 hash 审计。
  - 实现显式 GT/prediction 目录的标签质量评估器与 bootstrap 对比报告。
  - 生成 `m0_controls_20260724_v1`：Hard/Soft 各 4040 张，TR-CAMO 1000、TR-COD10K 3040。
  - 完成 4040 GT、8080 行的冻结配置诊断，manifest 状态为 complete。
  - 新增 Hard/Soft 两份训练配置，只切换 `refined_pseudo_label_dir`，暂未启动训练。
  - 验证两份配置均保留 `cache_dir='./datasets/cache'` 且精修标签目录存在。
  - 最终通过 `py_compile`、两个 CLI `--help`、5 项 unittest 与 `git diff --check`。

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 融合代数最小复现 | 二值 SAM/粗标签与不同 S | 验证是否为软融合 | `S<0.5` 等于粗标签，`S>0.5` 等于 SAM | 通过 |
| Excel 指标交叉核对 | 基线、朴素 SAM2、完整、论文行 | 统一比较口径 | 发现旧报告混用论文基线；当前增益多数较小 | 通过 |
| 标签级质量核对 | 粗标签与旧精修标签 | 验证边界是否改善 | IoU、BF-score、窄带指标几乎无改善 | 通过 |
| Milestone 0 单元测试 | 临时 2 图控制组与 1 图评估集 | 路径隔离、不可覆盖、软硬差异、改进识别 | 5 项全部通过 | 通过 |
| Milestone 0 完整控制组 | 4040 粗标签与原图 | Hard/Soft 各 4040、audit/hash 完整 | Hard 4040、Soft 4040、audit 4040 | 通过 |
| Hard vs Soft GT 诊断 | TR-CAMO 1000 + TR-COD10K 3040 | 量化表示方式混杂 | ΔIoU +0.000143、ΔBF +0.000130、ΔMAE -0.003379 | 通过 |
| 最终静态检查 | Milestone 0 Python 与配置文件 | 无语法或补丁空白错误 | `py_compile`、`git diff --check` 均通过 | 通过 |

### Phase 7: Milestone 1 边界安全 MVP
- **Status:** in_progress
- Actions taken:
  - 审计旧 SAM2 wrapper、V_loose、DINO 特征缓存和本地运行环境。
  - 确认复用 `(768,37,37)` DINO 特征，不重新提取特征。
  - 确认新实现必须修复旧框提示 XYWH/XYXY 契约错误，并复用单次图像编码生成多提示候选。

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-24 | 内置浏览器受本机 AppData 权限阻止 | 1 | 改用 Crossref、DBLP、Semantic Scholar 与作者仓库公开接口 |
| 2026-07-24 | CVF/IEEE PDF 下载不稳定 | 1 | 使用正式元数据、arXiv 摘要与作者 README 交叉核对 |
| 2026-07-24 | 并行组合读取未返回工具结果 | 1 | 拆分为独立读取命令，设计文件未受影响 |
| 2026-07-24 | 多文件同步补丁上下文不匹配 | 1 | 改为逐文件应用精确补丁，避免误改已有内容 |
| 2026-07-24 | 组合环境审计被非零子命令中断 | 1 | 拆分为 CfgNode、解释器和 PKL 独立检查 |
| 2026-07-24 | `test01` Python 在沙箱内无法直接启动 | 1 | 改用工作区系统 Python 3.9.13 |
| 2026-07-24 | torch 首次导入超过 10 秒 | 1 | 放宽超时到 30 秒后成功 |
| 2026-07-24 | unittest 误导入环境中的 Ultralytics `tests` | 1 | 新增本地 `tests/__init__.py` 后重跑 |
| 2026-07-24 | `git diff --check` 发现一行尾随空格 | 1 | 删除空格后重新验证 |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 7：Milestone 0 已完成，准备实现边界安全 MVP |
| Where am I going? | 实现语义定位、提示路由、候选质量、边界带和软残差融合 |
| What's the goal? | 在保留 AEEM 名称的前提下形成可实现、可验证、可复现的新方案 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已完成 Milestone 0 工具、4040 控制标签、GT 诊断和训练配置 |

## Session: 2026-07-25 Handoff Audit

### Phase 8: 状态审计与交接
- **Status:** complete
- Actions taken:
  - 确认旧 `docs/HANDOFF.md` 已过时，仍混用 AEEM v1 叙事且缺少全量实现、m4和正式核心消融。
  - 核对 `aeem_v2/` 当前源码模块与 `docs/AEEM_V2_QSEMANTIC25_FINAL_CONFIG.md`。
  - 核对精修标签独立路径注入和 A1 的 12 张 pkl 回退缩放修复。
  - 核对 A0、A1、m4 三个 epoch25 权重文件均存在。
  - 核对全量 AEEM 与 m4 各有 4040 张输出，naive SAM2 有 4028 张 PNG。
  - 重新运行完整测试集：32 tests passed。
  - 重写 `docs/HANDOFF.md`，完整覆盖已做工作、当前进度、注意点、挫折、保护边界和下一目标。
- Verified hashes:
  - m2 full4040: `2e3a081f55d806b3530d00626c231c0e221dd33ac0a0515e308e8fc6e2473850`
  - m4 final composition: `073abc4dcd13eaa24eb050a7dc063a88dda1a5a644750eefd8b7270cb92895e`
- Files modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/HANDOFF.md`

## Session: 2026-07-25 Innovation 1 Label-Level Evidence

### Phase 9: 原始粗标签 vs 朴素 SAM2
- **Status:** complete
- Scope frozen:
  - 仅复用现有原始粗标签、朴素 SAM2 PNG/pkl 回退和 GT。
  - 不重新运行 SAM2，不训练，不覆盖任何现有标签、artifact 或 checkpoint。
  - 优先复用现有参数化评估器；只有指标缺失时才做最小扩展。
- Actions taken:
  - 审计确认现有评估器缺少标准 Boundary IoU、面积/连通域成对误差统计和 partial-directory fallback。
  - 使用 Soft-Coarse 作为原始标签基线，并指定其为朴素 SAM2 缺失 PNG 的显式 fallback。
  - 为评估器增加标准内部边界区域 IoU（默认边界宽度为图像对角线的 2%）。
  - 增加绝对面积误差、连通域数量误差及其相对基线的成对 bootstrap CI。
  - 增加显式 `--prediction-fallback NAME=PATH`，主目录优先且 manifest 记录实际 fallback 数。
- Tests:
  - Phase 9 红灯：缺少 `_boundary_iou`，按预期失败。
  - EvaluationTests：3/3 passed。
  - 完整项目测试：34/34 passed。
  - `py_compile` 与评估 CLI `--help`：通过。
- Full evaluation:
  - Output: `artifacts/aeem_v2/innovation1_naive_vs_softcoarse_20260725_v1`
  - Manifest: complete；GT 4040；rows 8080；Naive-SAM2 fallback 12。
  - 全体 `Delta Boundary IoU=+0.044999`，95% CI `[+0.041834,+0.047961]`。
  - 全体 `Delta BF=+0.071180`，95% CI `[+0.067363,+0.074932]`。
  - 全体 `Delta IoU=-0.000954`，95% CI 跨 0。
  - 全体 `Delta MAE=+0.004199`，95% CI 严格大于 0（变差）。
  - 灾难性失败率（`Delta IoU < -0.2`）：1.4604%。
  - 新增论文证据文档：`docs/INNOVATION1_NAIVE_SAM2_BOUNDARY_EVIDENCE.md`。
  - 最终完整性核对：manifest complete、所需 7 个输出文件齐全、`git diff --check` 通过。

## Session: 2026-07-25 Innovation 2 Attribution

### Phase 10: AEEM v2 vs Naive-SAM2
- **Status:** complete; the preregistered extra control was executed in Phase 11
- Scope frozen:
  - 先比较现有 `m2_full4040_structure_20260724_v1` 与正式朴素 SAM2 标签。
  - GT 仅用于冻结后的诊断；不重新生成标签、不训练、不修改现有工件。
  - 标签级结果只证明机制输出质量，不能冒充下游增量消融。
- Actions taken:
  - 完成 4040 张同样本标签级比较 `innovation2_m2_vs_naive_20260725_v1`；完整方案标签相对朴素 SAM2 改善区域 IoU、MAE、质心和 raw 连通域误差，但牺牲部分朴素 SAM2 的极锐边界收益。
  - 逐字段核对 m2 与正式朴素 SAM2 保存的 `config.yaml`；除实验身份/输入输出路径外，训练与评估参数一致，确认无需重复训练 m2。
  - 复核两份 `eval0.log`；m2 相对朴素 SAM2 在四数据集 20 项核心指标中 8 项提升、12 项下降，宏观呈现区域/结构指标改善与 F_MAX/WFM 下降的权衡。
  - 交叉核对 m4 的 1760 个可靠样本与 naive 缺失清单：12 个缺失 PNG 中 3 个属于 TR-CAMO AEEM 侧，9 个属于 TR-COD10K Soft-Coarse 侧。
  - 新增 `docs/INNOVATION2_ATTRIBUTION_PROTOCOL.md`，冻结三层证据边界、额外控制的 1760/2271/9 组合、唯一变量、实验 ID 和单 seed 判定标准。
  - 本阶段截至当前仅修改研究记录；未生成额外控制标签、未启动训练、未覆盖任何既有工件。

## Session: 2026-07-25 Reliable Replacement Control

### Phase 11: 可靠替换控制与统一操作手册
- **Status:** complete
- User authorization:
  - 用户已明确要求继续验证实验。
  - 最终必须交付一份同时覆盖消融实验、完整模型训练和本次验证实验的统一操作手册。
- Frozen boundaries:
  - 不修改训练主干、数据加载链路、AEEM/SAM2 生成算法和既有配置语义。
  - 不覆盖或删除任何现有标签、artifact、checkpoint、日志或缓存。
  - 新验证使用独立实验 ID、独立工件目录和独立训练目录。
- Actions completed:
  - 新增隔离的三来源组合入口 `experiments/prepare_reliable_replacement_control.py`，先完整预检再创建目录，已有实验目录拒绝覆盖。
  - 生成 `innovation2_reliable_on_naive_20260725_v1`：1760 AEEM、2271 朴素 SAM2、9 Soft-Coarse fallback，共 4040 张；复制前后 hash 不一致为 0。
  - 在独立 `work_dir_validation_20260725` 完成单 seed、25 epoch 训练，epoch5/10/15/20/25 checkpoint 均存在。
  - 使用 epoch25 完成 CHAMELEON、TE-CAMO、TE-COD10K、NC4K 评估，`eval0.log` 正常结束。
  - 按预注册主要终点判定为“混合结果”：宏平均 SMeasure +0.002575、MAE -0.000825，但两项均只有 2/4 数据集方向一致，且 F_MAX -0.005600。
  - 新增 4 项可靠替换安全测试；完整项目测试为 38/38 通过，`py_compile`、CLI `--help` 和 `git diff --check` 通过。
- Final verification:
  - 完整项目测试 38/38 通过。
  - 可靠替换组合脚本 `py_compile` 与 CLI `--help` 通过。
  - `git diff --check` 退出码 0；只有既有 LF/CRLF 转换提示。
  - 标签工件 4040 PNG、4040 audit，五个 checkpoint、epoch25 权重和 `eval0.log` 均存在。
  - 统一 Markdown 与新 Word 均存在；Word 含 175 个段落、56 个标题和 6 张表。
  - 旧 Word 保持 51562 字节、原修改时间不变；本次未覆盖。

## Session: 2026-07-25 Safe Disk Cleanup

### Phase 12: 不影响老师复现的安全清理
- **Status:** in_progress
- User authorization:
  - 删除与本体无关、不会降低老师复现结果的缓存和无用数据。
- Protected paths:
  - `datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K`
  - `datasets/cache/naive_sam2_labels`
  - `datasets/cache/raw_sam2_outputs`
  - `artifacts/aeem_v2/m2_full4040_structure_20260724_v1`
  - `artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1`
  - 基线、朴素 SAM2、完整方案的正式 epoch25 与 `eval0.log`
  - 可靠替换的 artifact、epoch25、`eval0.log`、统一手册和源代码
- Current free space before cleanup: approximately 97 GiB.
- Read-only size audit:
  - `datasets`: 41.3 GiB，其中 DINOv2 features cache 41.18 GiB，属于正式训练输入缓存，保留。
  - `artifacts`: 1.94 GiB，其中 m2 1.02 GiB、m4 178 MiB、m0 152 MiB、可靠替换 114 MiB，均保留。
  - `weights`: 1.35 GiB，正在核对四份约 330 MiB 权重是否为可安全删除的重复文件。
  - `work_dir*`: 约 203 MiB，评估 `preds` 可重建，checkpoint 与日志保留。
- Documentation completed:
  - 更新 `docs/INNOVATION2_ATTRIBUTION_PROTOCOL.md`，写入工件 hash、训练路径、四测试集结果和“混合结果”判定。
  - 更新 `docs/HANDOFF.md`，将旧执行闸门同步为实际完成状态。
  - 新增 `docs/AEEM_V2_UNIFIED_EXPERIMENT_MANUAL.md`，覆盖消融、完整方案、可靠替换验证、复用/重建、结果解释和排错。
  - 新增 `scripts/generate_unified_experiment_manual.py` 并生成独立 Word；未覆盖桌面旧 Word。
