# PRD：人类 Agent 职业画像与住宅活动轨迹优化模块（学术数据驱动版）

## 1. 文档目的

本文档用于指导 AI Coding 对现有人宠共居模拟程序中的 `HumanAgent` 进行精细化开发。

本版本对上一版 PRD 进行重要修正：**人类 Agent 的职业画像参数不得再使用工程启发式设定作为最终值**。由于项目目标包含学术论文发表，所有与人的居家活动时长、外出时间、工作时间、睡眠时间、家务/照护/休闲时间相关的参数，必须来自权威时间利用调查数据，或由权威数据通过明确公式换算得到。

当前阶段重点是将 `HumanAgent` 从“固定时间段区域偏好”升级为“权威时间利用数据驱动的住宅活动模拟”。

---

## 2. 核心原则

### 2.1 数据优先原则

所有人类 Agent 参数分为两类：

```text
A. 数据直接来源参数
来自权威时间利用调查数据，例如 BLS ATUS、中国全国时间利用调查等。

B. 模型转换参数
由 A 类数据通过明确公式换算得到，例如 outside_ticks、home_work_ticks、sleep_zone_ticks、work_zone_weight 等。
```

不得使用无来源的主观设定作为最终参数。

### 2.2 参数可追溯原则

每个职业画像必须记录：

1. 使用的数据集。
2. 使用的表格编号。
3. 使用的职业分类或人群分类。
4. 原始数据字段。
5. 换算公式。
6. 最终生成的模拟参数。

输出结果中必须保存 `source_metadata.json`，用于论文复现和审稿追溯。

### 2.3 模型假设显式化原则

权威时间利用调查通常只能提供宏观活动时间预算，例如工作、睡眠、家务、照护、休闲等。它不能直接提供住宅内部“人在窗边停留多久”“在客厅移动几次”“与猫互动几次”。

因此：

```text
权威数据负责活动时间预算。
住宅区域映射属于模型假设。
二者必须在代码、输出和论文写作中明确区分。
```

---

## 3. 背景与问题

### 3.1 当前 HumanAgent 逻辑

当前代码中，`HumanAgent` 的活动逻辑主要由 `HUMAN_CONFIG["activity_patterns"]` 控制：

```python
"activity_patterns": {
    "morning":  {"zones": ["human_sleep", "shared", "cat_feeding"], "stay_probability": 0.3},
    "daytime":  {"zones": ["human_work", "shared", "window"],     "stay_probability": 0.4},
    "evening":  {"zones": ["shared", "human_sleep", "window"],    "stay_probability": 0.35},
}
```

当前流程为：

1. 按模拟进度划分早晨、白天、晚上。
2. 每个时间段从指定区域中选择目标。
3. 使用 A* 路径规划移动到目标区域。
4. 到达目标后在当前区域闲逛一段时间。
5. 根据所在区域输出行为，如睡眠、工作、闲逛、移动。

### 3.2 当前问题

1. 所有人类 Agent 行为模式基本相同。
2. 无法表达居家办公、通勤、轮班、退休、学生、自由职业等差异。
3. 没有“外出”状态，通勤型职业白天仍会在住宅内部活动。
4. 工作区、共享空间、卧室等区域的使用强度缺少数据依据。
5. 缺少可供用户选择的人群画像入口。
6. 启发式设置不满足学术论文对参数来源的要求。

---

## 4. 权威数据来源

### 4.1 美国数据源：BLS American Time Use Survey

数据源：U.S. Bureau of Labor Statistics, American Time Use Survey, 2024.

建议使用表格：

| 表格 | 用途 |
|---|---|
| ATUS Table 5 | 按职业、就业类型、收入、星期类型统计主业工作参与率与工作时长 |
| ATUS Table 7 | 按职业、就业类型、收入统计在家工作、在工作场所工作的人数比例与工作时长 |
| ATUS Table 8A / 8B / 8C | 按就业状态、性别、是否有儿童等统计 24 小时主要活动时间构成 |

推荐用途：

1. 使用 Table 5 获取不同职业的平均工作时长。
2. 使用 Table 7 获取不同职业的在家工作比例与在家工作时长。
3. 使用 Table 8 系列表获取就业/非就业人群的睡眠、家务、照护、休闲、教育等 24 小时活动构成。

### 4.2 中国数据源：中国全国时间利用调查

数据源：国家统计局《2018 年全国时间利用调查公报》。

可使用的居民 24 小时活动构成：

| 活动类别 | 时间 |
|---|---|
| 个人生理必需活动 | 11 小时 53 分钟 |
| 有酬劳动 | 4 小时 24 分钟 |
| 无酬劳动 | 2 小时 42 分钟 |
| 个人自由支配活动 | 3 小时 56 分钟 |
| 学习培训 | 27 分钟 |
| 交通活动 | 38 分钟 |

推荐用途：

1. 作为中国居民整体 24 小时活动构成基准。
2. 在缺少职业细分时，作为基础时间预算模板。
3. 与 ATUS 职业细分模型分开使用，不混合口径。

---

## 5. 产品目标

### 5.1 总目标

将人类 Agent 从“固定时间段区域偏好”升级为“权威时间利用数据驱动的住宅活动模拟”，让用户可以选择职业/生活方式画像，并生成可追溯、可复现、可用于论文分析的住宅轨迹、区域热力图和行为统计。

### 5.2 阶段目标

本阶段开发目标：

1. 保留住宅友好型职业画像名称，作为用户选择入口。
2. 为每个画像绑定权威数据来源和职业/人群分类。
3. 使用权威数据计算活动时间预算。
4. 将活动时间预算转换为模拟 tick。
5. 增加外出状态 `outside`。
6. 根据活动预算驱动住宅区域使用时间。
7. 输出人类行为统计与数据来源元信息。
8. 明确区分“数据来源参数”和“模型假设参数”。

---

## 6. 模块范围

### 6.1 本阶段包含

1. 新增数据驱动的人类画像配置。
2. 新增时间利用数据读取与参数生成模块。
3. 主程序支持用户选择职业画像。
4. `HumanAgent` 接收由数据生成的活动预算。
5. 支持外出状态。
6. 优化行为标签。
7. 输出人类行为统计。
8. 输出参数来源元数据。
9. 保持现有轨迹图、人类热力图、报告输出正常。

### 6.2 本阶段不包含

1. 多人家庭模拟。
2. 复杂人猫互动。
3. 宠物主问卷。
4. CAD / DXF 自动上色模块。
5. 通勤距离与城市交通模型。
6. 对个体职业行为的真实预测。
7. 宠物照护时间的直接建模，除非有独立数据来源。

---

## 7. 时间尺度设定

为了与时间利用调查数据对齐，建议将模拟时间尺度设为：

```text
1 tick = 1 minute
1 day = 1440 ticks
```

如果需要更长或更短模拟，可通过比例缩放：

```python
tick_minutes = 1
total_ticks = 1440
```

所有小时数据统一换算为分钟：

```python
minutes = hours * 60
ticks = minutes / tick_minutes
```

---

## 8. 职业画像设计

### 8.1 用户可见画像

保留住宅友好型画像名称，便于用户理解：

```python
human_profile_type = [
    "remote_worker",
    "commuter_office_worker",
    "freelancer",
    "shift_worker",
    "student",
    "retired_person",
    "home_caregiver",
    "service_worker",
    "manual_worker",
    "default"
]
```

### 8.2 画像与统计口径绑定

住宅友好型画像必须绑定底层统计口径。

| 画像 ID | 用户可见名称 | 建议底层数据口径 |
|---|---|---|
| remote_worker | 居家办公者 | ATUS Table 7 中在家工作比例较高的职业，如 Management, business, and financial operations / Professional and related |
| commuter_office_worker | 通勤办公者 | ATUS Table 5 + Table 7 中工作场所工作比例较高的职业，如 Office and administrative support |
| freelancer | 自由职业者 | 若无直接数据，不作为最终论文核心画像；需标注 model assumption 或另找数据 |
| shift_worker | 轮班工作者 | 若无班次数据，不作为最终论文核心画像；需标注 model assumption 或另找数据 |
| student | 学生 | ATUS Table 8 中 education activities，或另找学生时间利用数据 |
| retired_person | 退休者 | ATUS 非就业人群 / 中国居民时间利用数据中的非就业或老年群体数据，若有 |
| home_caregiver | 家庭照护者 | ATUS Table 8 中 household activities / caring and helping household members |
| service_worker | 服务业工作者 | ATUS occupation: Service occupations |
| manual_worker | 体力劳动者 | ATUS occupation: Construction and extraction / Production / Transportation and material moving |
| default | 默认居民 | 中国全国时间利用调查整体居民 24 小时构成，或 ATUS 全体人口构成 |

注意：若某画像缺少权威数据支撑，应在输出中标注为 `model_assumption`，不得作为论文核心结论依据。

---

## 9. 数据文件结构

建议新增：

```text
data/
├─ atus_2024_table5.csv
├─ atus_2024_table7.csv
├─ atus_2024_table8a.csv
├─ atus_2024_table8b.csv
├─ atus_2024_table8c.csv
├─ china_time_use_2018.csv

config/
├─ human_profile_mapping.json

simulation/
├─ time_use_parameter_builder.py
```

### 9.1 human_profile_mapping.json

该文件不存放最终活动参数，只存放画像与数据源的映射关系。

示例：

```json
{
  "remote_worker": {
    "display_name": "居家办公者",
    "source_dataset": "ATUS_2024",
    "source_tables": ["Table 5", "Table 7", "Table 8B"],
    "occupation_categories": [
      "Management, business, and financial operations",
      "Professional and related"
    ],
    "profile_status": "data_supported"
  },

  "service_worker": {
    "display_name": "服务业工作者",
    "source_dataset": "ATUS_2024",
    "source_tables": ["Table 5", "Table 7", "Table 8B"],
    "occupation_categories": [
      "Service occupations"
    ],
    "profile_status": "data_supported"
  },

  "default_china": {
    "display_name": "中国居民基准画像",
    "source_dataset": "China_Time_Use_2018",
    "source_tables": ["2018 全国时间利用调查公报"],
    "profile_status": "data_supported"
  }
}
```

---

## 10. 数据驱动参数生成逻辑

### 10.1 输入

```python
profile_id = "remote_worker"
country = "US"
total_ticks = 1440
tick_minutes = 1
day_type = "average_day"  # average_day / weekday / weekend
```

### 10.2 原始数据字段

从 ATUS 中读取：

```python
raw_time_use = {
    "work_hours": 7.67,
    "home_work_share": 0.481,
    "home_work_hours": 5.79,
    "sleep_hours": 8.5,
    "household_hours": 1.8,
    "care_hours": 0.6,
    "leisure_hours": 4.2,
    "education_hours": 0.2
}
```

注：上方数值仅为字段示例，实际必须由 CSV 数据读取，不得硬编码为最终参数。

### 10.3 换算公式

统一换算为分钟：

```python
work_minutes = work_hours * 60
home_work_minutes = work_minutes * home_work_share
outside_work_minutes = work_minutes * (1 - home_work_share)

sleep_minutes = sleep_hours * 60
household_minutes = household_hours * 60
care_minutes = care_hours * 60
leisure_minutes = leisure_hours * 60
education_minutes = education_hours * 60
```

交通活动如有数据：

```python
travel_minutes = travel_hours * 60
outside_minutes = outside_work_minutes + travel_minutes
```

若缺少交通数据：

```python
travel_minutes = 0
travel_source = "missing"
travel_assumption = "not modeled in current version"
```

### 10.4 归一化校准

所有活动分钟之和可能不等于 1440，需要归一化：

```python
total_activity_minutes = (
    sleep_minutes
    + home_work_minutes
    + outside_work_minutes
    + household_minutes
    + care_minutes
    + leisure_minutes
    + education_minutes
    + travel_minutes
)

scale = 1440 / total_activity_minutes

sleep_ticks = sleep_minutes * scale
home_work_ticks = home_work_minutes * scale
outside_ticks = (outside_work_minutes + travel_minutes) * scale
household_ticks = household_minutes * scale
care_ticks = care_minutes * scale
leisure_ticks = leisure_minutes * scale
education_ticks = education_minutes * scale
```

若数据源本身已为完整 24 小时构成，则不需要归一化，只需记录：

```python
normalization_applied = False
```

---

## 11. 活动到住宅区域的映射

### 11.1 映射原则

时间利用调查提供的是活动类别，不是房间位置。因此需要建立活动到住宅区域的映射规则。

该映射属于模型假设，必须单独保存。

### 11.2 推荐映射

```python
ACTIVITY_TO_ZONE_MAP = {
    "sleep": {
        "human_sleep": 1.0
    },
    "home_work": {
        "human_work": 0.85,
        "shared": 0.15
    },
    "outside": {
        "outside": 1.0
    },
    "household": {
        "shared": 0.70,
        "human_sleep": 0.15,
        "cat_feeding": 0.15
    },
    "care": {
        "shared": 0.50,
        "human_sleep": 0.25,
        "cat_feeding": 0.25
    },
    "leisure": {
        "shared": 0.65,
        "window": 0.20,
        "human_sleep": 0.15
    },
    "education": {
        "human_work": 0.75,
        "shared": 0.25
    }
}
```

注意：

1. `ACTIVITY_TO_ZONE_MAP` 是模型假设，不是调查数据。
2. 输出文件必须记录该映射。
3. 论文中必须说明：时间预算来自权威数据，住宅区域映射为模型设定。

---

## 12. HumanProfile 数据结构

更新后的 `human_profile` 不再直接写死 `home_presence`、`outside_probability` 等参数，而是包含来源与派生参数。

```python
human_profile = {
    "profile_id": "remote_worker",
    "display_name": "居家办公者",

    "source": {
        "dataset": "ATUS_2024",
        "tables": ["Table 5", "Table 7", "Table 8B"],
        "occupation_categories": [
            "Management, business, and financial operations",
            "Professional and related"
        ],
        "day_type": "average_day"
    },

    "derived_activity_budget": {
        "sleep_ticks": 510,
        "home_work_ticks": 260,
        "outside_ticks": 180,
        "household_ticks": 100,
        "care_ticks": 40,
        "leisure_ticks": 300,
        "education_ticks": 0
    },

    "zone_budget": {
        "human_sleep": 540,
        "human_work": 230,
        "shared": 420,
        "window": 60,
        "cat_feeding": 30,
        "outside": 160
    },

    "assumptions": {
        "activity_to_zone_map": "ACTIVITY_TO_ZONE_MAP_v1",
        "tick_minutes": 1,
        "normalization_applied": True
    }
}
```

注：上方数值是结构示例，不是最终数据。

---

## 13. HumanAgent 行为逻辑

### 13.1 从时间段驱动改为预算驱动

旧逻辑：

```text
morning / daytime / evening -> 选区域
```

新逻辑：

```text
活动预算 activity_budget -> 活动序列 activity_schedule -> 区域目标 zone target
```

### 13.2 活动序列生成

根据 `derived_activity_budget` 生成一天活动序列。

示例：

```python
activity_schedule = [
    {"activity": "sleep", "duration": 480},
    {"activity": "household", "duration": 45},
    {"activity": "outside", "duration": 180},
    {"activity": "home_work", "duration": 240},
    {"activity": "leisure", "duration": 300}
]
```

初期可以采用固定顺序模板：

```text
sleep -> household/care -> work/outside/education -> household/care -> leisure -> sleep/rest
```

但必须在代码中标注：

```text
activity order is a model assumption.
activity durations are data-driven.
```

后续可接入更细粒度的时间段分布数据。

### 13.3 区域选择

当当前活动为 `home_work`：

```python
zone_weights = ACTIVITY_TO_ZONE_MAP["home_work"]
target_zone = weighted_random_choice(zone_weights)
```

当当前活动为 `outside`：

```python
self.state = "outside"
```

当当前活动为 `sleep`：

```python
target_zone = "human_sleep"
```

---

## 14. 外出状态设计

### 14.1 outside 行为

新增人类状态：

```python
self.state = "outside"
```

当 Agent 进入 `outside` 状态时：

1. 人类不在住宅内部移动。
2. 人类热力图不增加室内访问。
3. `human_behavior` 输出为 `"外出"`。
4. `human_zone` 记录为 `"outside"`。
5. 外出持续时间由数据驱动的 `outside_ticks` 决定。

### 14.2 外出入口点

当前户型暂无门区，初期采用逻辑外出状态，不移动到具体门口。

后续 CAD 解析出门后，再映射到真实门位置。

---

## 15. HumanAgent 行为标签扩展

建议扩展为：

```python
human_behaviors = [
    "sleep",
    "home_work",
    "outside",
    "household",
    "care",
    "leisure",
    "education",
    "move",
    "wander"
]
```

中文显示：

| 行为 | 中文 |
|---|---|
| sleep | 睡眠 |
| home_work | 居家工作 |
| outside | 外出 |
| household | 家务 |
| care | 照护 |
| leisure | 休闲 |
| education | 学习 |
| move | 移动 |
| wander | 闲逛 |

---

## 16. 行为统计输出

### 16.1 区域停留时间

```json
"human_zone_stay_ticks": {
    "human_sleep": 520,
    "human_work": 260,
    "shared": 360,
    "window": 80,
    "cat_feeding": 40,
    "outside": 180
}
```

### 16.2 活动时间统计

```json
"human_activity_ticks": {
    "sleep": 510,
    "home_work": 260,
    "outside": 180,
    "household": 100,
    "care": 40,
    "leisure": 300,
    "education": 0
}
```

### 16.3 参数来源摘要

```json
"human_profile_summary": {
    "profile_id": "remote_worker",
    "display_name": "居家办公者",
    "source_dataset": "ATUS_2024",
    "source_tables": ["Table 5", "Table 7", "Table 8B"],
    "occupation_categories": [
        "Management, business, and financial operations",
        "Professional and related"
    ],
    "tick_minutes": 1,
    "normalization_applied": true
}
```

---

## 17. 文件输出

建议在原有输出基础上新增：

```text
outputs/
├─ simulation_result.png
├─ tick_records.csv
├─ cat_behavior_summary.json
├─ cat_profile_used.json
├─ human_behavior_summary.json
├─ human_profile_used.json
├─ source_metadata.json
└─ activity_to_zone_mapping_used.json
```

### 17.1 tick_records.csv 新增字段

```text
tick,
cat_x, cat_y, cat_zone, cat_behavior,
human_x, human_y, human_zone, human_activity, human_behavior, human_state, human_profile_id
```

如果人类处于外出状态：

```text
human_x = None
human_y = None
human_zone = outside
human_activity = outside
human_behavior = 外出
human_state = outside
```

### 17.2 source_metadata.json

必须记录：

```json
{
  "dataset": "ATUS_2024",
  "tables": ["Table 5", "Table 7", "Table 8B"],
  "raw_files": [
    "atus_2024_table5.csv",
    "atus_2024_table7.csv",
    "atus_2024_table8b.csv"
  ],
  "occupation_categories": [
    "Management, business, and financial operations",
    "Professional and related"
  ],
  "derived_formulas": {
    "home_work_minutes": "work_minutes * home_work_share",
    "outside_work_minutes": "work_minutes * (1 - home_work_share)"
  },
  "model_assumptions": [
    "Activity order is model-defined.",
    "Activity-to-zone mapping is model-defined.",
    "Pet care time is not directly modeled unless independent data are provided."
  ]
}
```

---

## 18. 主程序交互设计

### 18.1 命令行选择

主程序启动时支持职业画像选择：

```text
请选择人类 Agent 职业画像：
1. 默认居民
2. 居家办公者
3. 通勤办公者
4. 服务业工作者
5. 体力劳动者
6. 中国居民基准画像
```

注意：

若某画像尚未完成数据支撑，不应作为默认可选项，或必须标注：

```text
该画像当前为模型假设，不用于论文正式实验。
```

### 18.2 非交互模式

为了 AI Coding 和批量实验，建议支持代码参数：

```python
sim = Simulation(
    floor_plan,
    total_ticks=1440,
    human_profile_id="remote_worker",
    country="US",
    day_type="average_day"
)
```

---

## 19. 推荐代码改造方案

### 19.1 新增模块

```python
class TimeUseParameterBuilder:
    def __init__(self, data_dir, mapping_path, tick_minutes=1):
        pass

    def build_profile(self, profile_id, country="US", day_type="average_day"):
        pass

    def load_source_tables(self):
        pass

    def extract_raw_time_budget(self, profile_mapping):
        pass

    def derive_activity_budget(self, raw_time_budget):
        pass

    def map_activity_to_zone_budget(self, activity_budget):
        pass

    def export_source_metadata(self, output_path):
        pass
```

### 19.2 修改 HumanAgent 初始化

当前：

```python
def __init__(self, start_x, start_y, zone_map, passable_map, scale_x, scale_y, total_ticks=5000):
```

改为：

```python
def __init__(
    self,
    start_x,
    start_y,
    zone_map,
    passable_map,
    scale_x,
    scale_y,
    total_ticks=1440,
    human_profile=None
):
```

内部：

```python
self.profile = human_profile
self.activity_schedule = self.profile["activity_schedule"]
self.current_activity_index = 0
self.current_activity_remaining = self.activity_schedule[0]["duration"]
```

### 19.3 修改 Simulation 初始化

当前：

```python
def __init__(self, floor_plan_path, total_ticks=5000):
```

改为：

```python
def __init__(
    self,
    floor_plan_path,
    total_ticks=1440,
    human_profile_id="default",
    country="US",
    day_type="average_day",
    data_dir="data"
):
```

内部：

```python
builder = TimeUseParameterBuilder(data_dir=data_dir, mapping_path="config/human_profile_mapping.json")
human_profile = builder.build_profile(human_profile_id, country=country, day_type=day_type)
self.human = HumanAgent(..., human_profile=human_profile)
```

### 19.4 HumanAgent 推荐新增方法

```python
class HumanAgent:
    def _get_current_activity(self):
        pass

    def _advance_activity_if_needed(self):
        pass

    def _start_activity(self, activity):
        pass

    def _select_zone_for_activity(self, activity):
        pass

    def _start_outside(self, duration):
        pass

    def _outside_step(self):
        pass

    def _record_statistics(self):
        pass
```

---

## 20. 可视化报告优化

在 `visualize()` 的报告区增加：

```text
Human Profile: 居家办公者
Source Dataset: ATUS 2024
Source Tables: Table 5, Table 7, Table 8B
Tick Scale: 1 tick = 1 minute

Human Activity Budget:
- Sleep: xxx min
- Home Work: xxx min
- Outside: xxx min
- Household: xxx min
- Leisure: xxx min

Note:
Activity-to-zone mapping is model-defined.
```

---

## 21. 验收标准

### 21.1 功能验收

满足以下条件视为完成：

1. 用户可以选择数据支持的人类职业画像。
2. `HumanAgent` 可以接收由 `TimeUseParameterBuilder` 生成的 `human_profile`。
3. 所有活动时长均来自权威数据或明确换算公式。
4. 程序不再使用无来源的 `home_presence`、`outside_probability`、`stay_duration_range` 作为最终参数。
5. 程序支持 `outside` 状态。
6. 程序能输出 `human_behavior_summary.json`。
7. 程序能输出 `source_metadata.json`。
8. 程序能输出 `activity_to_zone_mapping_used.json`。
9. 程序仍能正常输出 `simulation_result.png`。
10. 不破坏猫 Agent 原有逻辑。

### 21.2 学术验收

满足以下条件才可用于论文实验：

1. 所有画像均能追溯到具体数据源。
2. 所有换算公式均在代码和文档中记录。
3. 所有模型假设均被显式标注。
4. 活动时间预算与住宅区域映射分开记录。
5. 任何缺乏数据支撑的画像不得作为论文核心实验结论。
6. 输出结果可复现。

### 21.3 对比测试

#### 测试 A：居家办公者

输入：

```python
human_profile_id = "remote_worker"
```

预期：

- 居家工作活动时长由 ATUS Table 5 / Table 7 换算得到。
- 人类热力图在 `human_work` 和 `shared` 区域有明显分布。
- 外出时间由非居家工作时间和交通时间推导。

#### 测试 B：服务业工作者

输入：

```python
human_profile_id = "service_worker"
```

预期：

- 工作活动时长由 ATUS Table 5 读取。
- 在家工作比例由 ATUS Table 7 读取。
- 外出时间高于居家办公者。

#### 测试 C：中国居民基准画像

输入：

```python
human_profile_id = "default_china"
country = "CN"
```

预期：

- 24 小时活动构成来自中国 2018 年全国时间利用调查公报。
- 有酬劳动、无酬劳动、个人自由支配、学习培训、交通活动等被换算为活动预算。
- 住宅区域映射按 `ACTIVITY_TO_ZONE_MAP` 执行，并记录为模型假设。

---

## 22. 开发优先级

### P0：必须完成

1. 删除或停用启发式职业画像最终参数。
2. 新增 `TimeUseParameterBuilder`。
3. 新增 `human_profile_mapping.json`。
4. 支持读取权威数据 CSV。
5. 活动时长由数据生成。
6. 活动时长换算为 tick。
7. 输出 `source_metadata.json`。
8. 输出 `human_behavior_summary.json`。

### P1：建议完成

1. 支持美国 ATUS 的 Table 5、Table 7、Table 8B。
2. 支持中国 2018 年时间利用调查整体居民基准画像。
3. 报告图中展示数据来源。
4. `tick_records.csv` 记录活动类型与数据源。
5. 支持 `day_type = average_day / weekday / weekend`。

### P2：后续增强

1. 周末/工作日差异。
2. 多人家庭画像组合。
3. 人猫互动频率根据人的活动类型调制。
4. 更细粒度住宅活动：做饭、清洁、娱乐、健身、学习、照护。
5. 夜间时间段与轮班工作者特殊作息。
6. 接入更多国家或地区的时间利用调查数据。

---

## 23. 风险与注意事项

1. 不得将模型假设伪装成调查数据。
2. 不得将启发式参数作为论文最终依据。
3. 职业画像是统计口径下的活动模式，不是个体行为预测。
4. 不同国家数据不可随意混合，除非明确说明口径差异。
5. ATUS 中没有住宅房间级数据，房间映射必须作为模型假设。
6. 若缺少宠物照护数据，不应凭空生成“喂猫时间”作为论文结论。
7. 所有参数生成过程必须可复现。
8. 所有数据文件版本必须固定，避免后续网页更新导致结果变化。

---

## 24. 推荐开发顺序

```text
Step 1：整理权威数据为 CSV
Step 2：建立 human_profile_mapping.json
Step 3：新增 TimeUseParameterBuilder
Step 4：从数据中读取原始活动时间
Step 5：将小时换算为分钟 / tick
Step 6：建立 ACTIVITY_TO_ZONE_MAP
Step 7：生成 zone_budget 和 activity_schedule
Step 8：改造 HumanAgent，使其按 activity_schedule 行动
Step 9：加入 outside 状态
Step 10：输出 source_metadata.json 和 human_behavior_summary.json
Step 11：用 remote_worker / service_worker / default_china 做对比测试
```

---

## 25. 本阶段最终产物

开发完成后，应得到：

1. 一个数据驱动的人类 Agent。
2. 至少 3 个数据支持画像：居家办公者、服务业工作者、中国居民基准画像。
3. 每个画像对应的活动预算、区域预算、轨迹图、热力图和统计 JSON。
4. 完整的 source metadata。
5. 住宅区域映射假设记录。
6. 可供学术论文方法章节描述的参数生成流程。

---

## 26. 简要总结

本阶段核心不是让人类 Agent “看起来更合理”，而是让其参数来源能够经受学术审查。

目标效果是：

```text
同一住宅布局中，不同人群画像的住宅活动轨迹，
由权威时间利用调查中的活动时间预算驱动，
并通过明确的、可追溯的模型映射规则转换为空间使用行为。
```

这样，人类 Agent 模块才能从演示系统升级为可用于论文分析的研究工具。
