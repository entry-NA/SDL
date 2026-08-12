# Task Plan: AEEM v2 设计与启动准备

## Goal
形成一份可直接指导实现和实验的 AEEM v2 设计方案，保留“自适应边缘感知增强机制”名称，同时修复旧方案在定位、提示、候选选择、融合语义和实验可复现性上的问题。

## Current Phase
Phase 12 in progress: safe disk cleanup without changing teacher reproduction inputs

## Phases

### Phase 1: 需求与事实冻结
- [x] 明确保留 AEEM 名称与两项创新叙事
- [x] 汇总当前代码、实验数据和标签级诊断结论
- [x] 记录不得覆盖现有标签、缓存和 checkpoint 的约束
- **Status:** complete

### Phase 2: 旧设计与实现约束核对
- [x] 对齐旧设计、HANDOFF、实验日志和当前代码
- [x] 划分可复用模块、必须修复模块和暂缓模块
- [x] 明确最小可验证版本的代码边界
- **Status:** complete

### Phase 3: AEEM v2 设计文档
- [x] 定义目标、非目标和术语
- [x] 定义语义定位、提示路由、边界带与像素置信模块
- [x] 定义数据格式、接口、回退策略和日志字段
- [x] 定义实验矩阵与阶段验收门槛
- **Status:** complete

### Phase 4: 设计审查
- [x] 检查是否仍存在全局硬切换或循环评分
- [x] 检查无监督设定是否被 GT 调参破坏
- [x] 检查每个创新声明是否有独立消融支撑
- [x] 检查精修标签路径是否会改变特征与原始伪标签缓存
- **Status:** complete

### Phase 5: 启动交付
- [x] 给出第一批实现任务顺序
- [x] 标明暂不运行的高成本训练
- [x] 与用户确认后进入实现阶段
- **Status:** complete

### Phase 6: Milestone 0 实现
- [x] 解耦精修标签目录与基础缓存目录
- [x] 实现实验目录不可覆盖与 manifest/hash 审计
- [x] 实现 Hard-Coarse 与 Soft-Coarse PNG 控制组
- [x] 实现参数化标签质量评估入口
- [x] 增加针对性测试并完成小样本验证
- [x] 生成完整控制组并完成冻结配置 GT 诊断
- **Status:** complete

### Phase 7: Milestone 1 边界安全 MVP
- [x] 实现 DINO 语义原型与定位可靠性
- [x] 实现 High/Medium/Low 提示路由
- [x] 实现多提示候选缓存与质量估计
- [x] 实现边界不确定带与软残差融合
- [x] 在固定样本、120样本和全量4040上完成验证
- **Status:** complete

### Phase 8: 2026-07-25 状态审计与会话交接
- [x] 核对当前实现、artifact、配置、训练日志和正式消融结果
- [x] 区分已实现代码、已完成实验和仍停留在论文设计层的内容
- [x] 明确不可修改文件、不可覆盖数据和实验复现入口
- [x] 重写 `docs/HANDOFF.md`，使下一会话可直接继续
- [x] 更新 `findings.md` 与 `progress.md`
- **Status:** complete

### Phase 9: 创新点1标签级边界证据
- [x] 审计现有参数化标签评估器与原始/朴素 SAM2 标签输入契约
- [x] 在不重新生成标签的前提下比较原始粗标签与朴素 SAM2
- [x] 汇总 Mask IoU、BF-score、Boundary IoU、MAE、质心偏移、面积误差与结构变化
- [x] 补充灾难性失败率和统计不确定性，形成可用于论文的结论边界
- [x] 记录命令、输出和验证结果，不修改冻结标签与 checkpoint
- **Status:** complete

### Phase 10: 创新点2归因收敛
- [x] 直接比较全量 AEEM v2 与朴素 SAM2 的同样本标签质量
- [x] 检查 AEEM 是否降低 naive 的区域、质心、碎片和灾难失败代价
- [x] 把标签级机制证据与现有 m3/m4 下游结果分开解释
- [x] 形成最小额外训练控制组及明确判定标准
- [x] 未经用户确认不生成控制标签、不启动训练
- **Status:** complete; extra reliable-replacement control also completed in Phase 11

### Phase 11: 可靠替换控制与统一操作手册
- [x] 只读预检现有组合入口、训练入口、评估入口和所有源工件
- [x] 生成不可覆盖的 1760 AEEM + 2271 朴素 SAM2 + 9 Soft-Coarse 控制工件
- [x] 核对 manifest、逐文件来源、输入/输出 hash 与训练配置唯一变量
- [x] 使用新实验 ID 完成单 seed、25 epoch 训练
- [x] 使用冻结 checkpoint 完成 CHAMELEON、TE-CAMO、TE-COD10K、NC4K 评估
- [x] 按预注册标准比较朴素 SAM2、可靠替换控制与完整模型
- [x] 生成覆盖消融实验、完整模型与验证实验的 Markdown + Word 统一操作手册
- [x] 完成最终完整性、路径、命令与不覆盖旧工件核验
- **Status:** complete

### Phase 12: 不影响复现的安全磁盘清理
- [ ] 只读统计项目目录、artifact、work_dir、缓存和临时文件体积
- [ ] 冻结老师复现必须保留的输入、正式结果、代码、手册与审计文件
- [ ] 形成仅包含可重建临时文件和重复中间产物的精确删除清单
- [ ] 删除前核对每个绝对路径均位于项目工作区且不属于保护清单
- [ ] 删除安全项并记录释放空间与不可恢复范围
- [ ] 重新核对正式标签、三组 checkpoint、验证 epoch25、日志和完整测试
- **Status:** in_progress

## Key Questions
1. 如何在不改变 AEEM 名称的前提下，把技术内涵从 Canny 全局门控升级为边界区域感知与受控残差精修？
2. 如何用独立语义信号修正粗标签空间偏移，避免候选选择循环依赖粗标签 IoU？
3. 如何先通过标签级实验验证收益，再投入完整训练？
4. 如何区分 SAM2 贡献与 PNG、二值化、插值和加载路径造成的混杂效应？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 保留 AEEM 名称 | 名称覆盖定位、提示、边界与融合；问题在旧实现，不在总叙事 |
| SAM2 只做边界候选与残差修正 | 避免 SAM2 同时承担定位和整图结构改写 |
| 首先增加 Hard-Coarse PNG 控制组 | 排除标签硬化、插值和加载路径造成的假增益 |
| 候选质量、门控决策、融合权重分离 | 避免一个未校准标量同时承担三个语义 |
| GT 仅用于离线诊断，不用于最终阈值调参 | 保持无监督实验设定 |
| 新实验输出到独立、带名称和时间戳的目录 | 防止再次覆盖已有标签与污染训练状态 |
| MVP 先输出单通道软标签并保持当前训练接口 | 当前 BCE 与 APM 已能消费 `[0,1]` 灰度目标，可先隔离验证离线算法 |
| 像素置信图首版作为 sidecar 保存 | 避免首轮同时改数据结构、判别器和损失；验证有效后再接入 APM |
| 精修标签使用独立 `refined_pseudo_label_dir` | `cache_dir` 同时控制特征与原始伪标签缓存，不能被实验 artifact 替换 |
| 控制组先上采样到原图尺寸再保存 PNG | 确保后续经过与 SAM2 精修标签相同的 LANCZOS→68 加载路径 |
| Milestone 0 不修改现有实验分析脚本 | 这些文件含用户历史工作；新增参数化入口，避免破坏旧结果复现 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 旧实验混用不同论文基线 | 1 | 设计中要求固定 baseline manifest 与统一评估协议 |
| 旧融合公式对二值掩码退化为硬切换 | 1 | MVP 将像素置信保留为软权重，不立即二值化；ignore map 留到 v2.1 |
| EdgeAlign 曾存在 0/255 量纲错误 | 1 | v2 统一所有置信量到 [0,1]，并增加范围断言 |
| 设计初稿拟把 `cache_dir` 指向 artifact | 1 | 改为独立 `refined_pseudo_label_dir`，保留官方缓存根目录 |
| 并行组合读取没有返回工具结果 | 1 | 拆分为独立读取命令，未修改任何项目文件 |
| 多文件同步补丁因上下文不匹配失败 | 1 | 改为逐文件应用精确补丁，避免误改已有内容 |
| 组合环境审计被子命令非零退出中断 | 1 | 拆分 CfgNode、解释器和 PKL 检查命令 |
| `test01` Python 在沙箱内无法直接启动 | 1 | 使用工作区可用的系统 Python 3.9.13 完成只读检查 |
| 首次导入 torch 超过 10 秒 | 1 | 放宽到 30 秒后成功，确认 torch 2.8.0 可用 |
| `python -m unittest tests...` 导入了环境中的 Ultralytics `tests` | 1 | 新增本地 `tests/__init__.py`，固定项目测试包解析 |
| `git diff --check` 发现 `base_dataset.py` 尾随空格 | 1 | 删除该空白行并重新执行静态检查 |
| 组合读取设计文档时自动审批服务返回 503 | 1 | 未重试同一调用；改以已生成的最终配置、来源隔离报告、manifest和实际代码交叉核对 |
| 最终组合校验因 `rg` 零匹配返回1而提前结束 | 1 | 改用 PowerShell `Select-String` 统计；确认尾随空格为0且 `git diff --check` 通过 |
| Phase 9 红灯测试无法导入 `_boundary_iou` | 1 | 预期失败；证明标准 Boundary IoU 尚未实现，随后按测试补齐评估器 |
| 并行预检被自动审批服务以 429 拒绝 | 1 | 用户明确批准只读检查后改为逐项执行；未绕过审批，未影响文件 |
| 4040 样本评估的前端等待在 10 分钟超时 | 1 | 未重跑或覆盖；后台任务继续完成，manifest 最终为 complete，4040 GT/8080 rows/fallback 12 均核实 |
| Phase 10 逐图统计的 PowerShell 管道解析失败 | 1 | 未写文件；改为先收集 `foreach` 结果再序列化，避免空管道元素语法 |
| Phase 11 首次追加 findings 因上下文空格不匹配失败 | 1 | 未修改文件；使用准确现有尾行重新应用补丁 |
| Windows 下 `rg *.md` 通配符导致组合只读核对退出码 1 | 1 | 未修改文件；改用明确目录和文件名逐项读取 |
| 最终状态扫描因预期零匹配返回退出码 1 | 1 | 未修改文件；改用 `Select-String` 并显式处理零匹配 |

## Notes
- 不覆盖 `datasets/cache/` 下任何现有标签、raw 输出或缓存。
- 不直接修改或回退用户当前工作区中的未提交文件。
- Milestone 0 完成前不启动完整训练或全量 SAM2 推理。
