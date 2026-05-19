# AGENTS.md

本文件为 Codex 在本仓库工作的项目级说明。内容参考 `CLAUDE.md` 和 `docs/progress.md`。

## 项目概览

这是一个基于多智能体建模（ABM）的人宠共居住宅空间分析工具。项目通过 Python 仿真人与猫的时空轨迹，提取强度、熵、共现密度等多维评价指标，自动识别空间冲突节点并辅助设计推导。

## 技术栈

- 语言：Python 3.10+
- 数据处理：NumPy, Pandas
- 核心算法：Scikit-learn DBSCAN 聚类、SciPy 信息熵计算
- 图像处理：Matplotlib、Pillow
- 仿真环境：离散栅格坐标系，默认 `200 x 200` grid

## 核心文件

- `simulation_v9.py`：基础仿真引擎，包含户型图解析、人/猫 agent、tick 驱动与基础可视化。
- `time_use_parameter_builder.py`、`config/human_profile_mapping.json`：人类时间利用画像参数生成与配置。
- `trajectory_analyzer.py`：轨迹 CSV 导入导出、格栅化、行为频次字典、tick 级共现计数。
- `metrics_calculator.py`：空间功能强度、行为熵、全状态/活跃共现密度、主导行为等指标。
- `node_detector.py`：百分位阈值过滤、DBSCAN 聚类、节点分类。
- `dashboard.py`：多通道可视化仪表盘输出。
- `docs/progress.md`：当前实现进度和模块说明。
- `openspec/`、`.codex/skills/openspec-*`：OpenSpec 变更工作流。

## 必须遵守的工作原则

1. 所有评价指标计算必须追溯到原始 tick 级轨迹数据。
2. 禁止对已聚合的热力图进行二次运算来推导共现、冲突或行为指标，避免“伪共现”误差。
3. 若需要计算冲突点，必须检查同一时间步 tick 下的人猫位置关系，而不是简单叠加两张静态热力图。
4. 轨迹生成逻辑与指标分析逻辑必须解耦。分析脚本应通过 tick records 或轨迹 CSV 读取数据。
5. 坐标映射必须统一使用同一套缩放/格栅化逻辑，确保像素坐标、格栅索引和物理距离语义一致。
6. 可视化脚本应支持无界面环境，必要时使用 `matplotlib.use("Agg")`。
7. 修改时优先沿用现有模块边界，不做无关重构。

## 当前实现状态

项目核心模块 A/B/C/D 已完成：

- 模块 A：数据持久化与格栅化引擎。
- 模块 B：五维评价指标算法。
- 模块 C：节点峰值检测与聚类。
- 模块 D：多通道可视化仪表盘。

后续可扩展方向包括参数敏感性分析、多户型对比实验、HTML 交互式报告、CI/CD 自动化分析流水线。

## Agent 当前规则

- 人类 agent 使用时间利用画像驱动，默认 `default_china`；职业/人群配置来自 `config/human_profile_mapping.json`，源数据来自 `data/`。
- 人类外出状态记为 `outside`，不产生室内坐标；分析流程必须跳过外出 tick 的室内共现计算。
- 猫 agent 使用档案 + 动态状态 + 空间语义的规则模型；默认成年猫休息目标按约 `14/24` 校准。
- 猫预设包括 `sensitive_hiding`、`curious_active`、`friendly_companion`、`senior_arthritis`，修改行为权重时需保持档案差异可解释。

## 开发与验证

测试和验证运行产生的图片、CSV、JSON 等结果文件统一写入仓库根目录下的 `result/`：

```text
/home/administrator/human-cat-happily-living/result/
```

除非用户明确要求覆盖根目录历史产物，避免把临时测试结果散落在项目根目录。

优先使用已有脚本的独立测试入口验证变更：

```bash
python trajectory_analyzer.py
python metrics_calculator.py
python node_detector.py
python dashboard.py
```

注意：

- 这些脚本可能依赖运行后生成的 `result/floor_plan.png`、`result/trajectory.csv` 或其他输出文件。
- 若测试入口生成图片或 CSV，确认输出语义正确，不要把临时产物误当作源码改动提交。
- 如果修改了共现、冲突、聚类或指标逻辑，应至少验证 `trajectory_analyzer.py` 和相关下游模块。

## 提交规范

每完成一个算法模块或明确功能单元后再提交，commit message 使用：

- `feat: 描述`
- `fix: 描述`
- `research: 描述`

不要在未被要求时自动提交。

## OpenSpec 工作流

当用户要求提案、实现变更、探索需求或归档变更时，优先使用仓库内 `.codex/skills/openspec-*` 技能：

- `openspec-explore`：需求探索与澄清。
- `openspec-propose`：生成变更提案、设计和任务。
- `openspec-apply-change`：按 OpenSpec change 执行实现任务。
- `openspec-archive-change`：完成后归档变更。

## 给后续 agent 的提醒

- 中文注释和输出是本项目现有风格，可以保留。
- 关键算法注释应说明“为什么这样算”，避免只解释语法。
- 对人猫共现、冲突和活跃状态的任何改动都属于高风险改动，必须从 tick 级数据路径向下检查。
