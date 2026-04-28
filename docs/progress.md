## 1. 现有代码架构总结 (Current Implementation - V9)文件路径：\\wsl.localhost\Ubuntu\home\administrator\miaowu\simulation_v9_standalone
*当前代码已完成环境搭建与基础动态模拟，可作为 AI 进一步开发的 Context。*
*这是基础仿真引擎，需要理解其数据结构。*

### 1.1 核心类与功能
- **`FloorPlanParser`**: 户型图解析器。通过 RGB 颜色识别区域（卧室、猫休息区等），生成 `passable_map`（布尔矩阵）。
- **`CatAgent` / `HumanAgent`**: 智能体类。包含能量系统、基于时间段的决策逻辑（State Machine）和随机行走/寻路算法。
- **`Simulation`**: 主控类。负责 Tick 步进驱动和基础热力图可视化。

### 1.2 待对接数据接口 (Data Ready for Export)
每个 Tick 产生的核心数据（AI 需利用这些数据进行后续开发）：
- `tick`: 当前时间步
- `cat_pos (x, y)`, `cat_behavior`: 猫的坐标与行为（如：休息、奔跑、进食）
- `human_pos (x, y)`, `human_behavior`: 人的坐标与行为（如：工作、睡眠、移动）

## 2. 待开发模块：多维度空间节点评价 (Backlog & Prompts)

不要一次写完所有功能。按照 **“模块 A -> 模块 B -> 模块 C -> 模块 D”** 的顺序分步请求。

### 模块 A：数据持久化与格栅化引擎 (Data Grid Engine)
**任务目标**：将连续的轨迹点转化为 $200 \times 200$ 的离散行为矩阵。
*   **输入**：模拟运行产生的轨迹列表。
*   **计算逻辑**：
    1. **坐标转换**：将连续坐标映射至 `(int(x), int(y))` 格栅索引。
    2. **行为档案建立**：为每个格栅单元 (Cell) 建立一个计数器，记录该位置发生各类行为的频次。
*   **AI 任务点**：编写 `TrajectoryAnalyzer` 类，实现轨迹导出 CSV 和栅格化映射。

### 模块 B：五维评价指标算法 (Analysis Metrics)
**任务目标**：对每个格栅单元进行定量评价。
*   **指标 1：空间功能强度 ($S$)**：$S = \sum (Count_{behavior} \times Weight_{behavior})$。*参考：奔跑=8，休息=2。*
*   **指标 2：行为熵 ($H$)**：应用香农熵公式 $H = -\sum p_i \log p_i$，衡量该点功能的杂乱程度。
*   **指标 3：活跃共现密度 ($D_{active}$)**：**关键技术点**——仅当同一 Tick 内，人猫处于同一格栅，且双方 `behavior` 均非“静止/睡眠”状态时，计数 +1。
*   **指标 4：全状态共现密度**：记录人猫物理空间重叠的总频率。
*   **指标 5：主导行为类型**：识别该点频次最高的 Top 3 行为。

### 模块 C：节点峰值检测与聚类 (Peak Detection & Clustering)
**任务目标**：将离散的高分格栅聚合成具有设计意义的“空间节点”。
*   **步骤 1：阈值过滤**：利用 `numpy.percentile` 提取前 20%（强度）或前 10%（共现）的高分格栅。
*   **步骤 2：空间聚类**：调用 `sklearn.cluster.DBSCAN` 算法。
    *   `eps`: 建议 5 栅格（约 0.35m）。
    *   `min_samples`: 建议 3 个点。
*   **步骤 3：质心计算**：计算每个聚类簇的质心（Centroid），定义为该“空间节点”的坐标。

### 模块 D：多通道可视化仪表盘 (Visualization Dashboard)
**任务目标**：生成类似论文品质的六通道对比图。
*   **技术要求**：使用 `matplotlib.subplot` 创建 $2 \times 3$ 布局。
*   **输出通道**：
    1. 猫空间强度图 (Heatmap)
    2. 人空间强度图 (Heatmap)
    3. 行为熵分布图 (Texture/Density map)
    4. 活跃共现冲突点图
    5. 节点分类符号图（利用不同形状标记冲突、共享、低利用节点）
    6. 自动化设计策略建议表格（节点画像卡片）