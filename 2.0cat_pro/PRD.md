# PRD：猫咪 Agent 五维猫格行为模拟模块

## 1. 文档目的

本文档用于指导 AI Coding 对现有人宠共居模拟程序进行猫咪 Agent 精细化开发。

当前项目已经具备基础模拟闭环：户型图生成与解析、人类 Agent 移动、猫咪 Agent 移动、轨迹记录、热力图输出和模拟报告。下一阶段重点是将猫咪 Agent 从“能量驱动 + 随机目标选择”的简单模型，升级为“猫格参数 + 客观档案 + 动态状态 + 行为权重”共同驱动的差异化行为模型。

本 PRD 主要覆盖猫咪 Agent 部分，不包含宠物主问卷设计。问卷输入与猫格映射将在后续版本补充。

---

## 2. 背景与问题

### 2.1 当前代码现状

现有 `CatAgent` 的核心逻辑如下：

1. 猫随机出生在可通行区域。
2. 猫拥有 `energy` 和 `satisfaction` 两个状态。
3. 猫根据能量值选择目标区域。
4. 猫朝目标方向移动。
5. 每一步有固定概率进入奔跑状态。
6. 到达特定区域后恢复能量或增加满意度。
7. 输出行为标签：奔跑、休息、进食、观望、游走。

### 2.2 当前问题

当前猫咪 Agent 存在以下不足：

1. 不同猫之间缺乏行为差异。
2. 猫咪目标选择主要受能量影响，缺少性格、年龄、健康状态等因素。
3. 猫咪行为频率、移动速度、奔跑概率、空间偏好均为固定参数。
4. 猫咪与住宅空间之间的关系较弱，缺少躲藏、探索、亲人、地盘占有等行为倾向。
5. 目前输出结果难以用于比较不同猫格与不同住宅布局的适配关系。

---

## 3. 产品目标

### 3.1 总目标

将猫咪 Agent 升级为可参数化、可扩展、可对比实验的行为模拟模块，使不同猫咪在同一住宅布局中表现出可区分的轨迹、热力图、区域停留和行为频率。

### 3.2 阶段目标

本阶段开发目标为：

1. 在 `CatAgent` 中加入五维猫格参数。
2. 在 `CatAgent` 中加入客观档案层，包括年龄、性别、绝育状态、疾病史、身体条件等。
3. 增加猫咪动态状态，包括压力、无聊、饥饿、安全感、社交需求等。
4. 使用行为权重替代现有固定目标列表。
5. 让猫格与客观条件共同影响移动速度、奔跑概率、目标区域偏好、行为切换频率和区域停留时间。
6. 输出可用于分析的行为统计数据。

---

## 4. 模块范围

### 4.1 本阶段包含

1. 猫咪 Agent 数据结构改造。
2. 五维猫格参数接入。
3. 客观档案参数接入。
4. 动态状态更新逻辑。
5. 目标区域权重选择逻辑。
6. 移动速度与奔跑概率调制。
7. 行为标签扩展。
8. 区域停留时间与行为频率统计。
9. 单只猫在单户型中的模拟结果输出。

### 4.2 本阶段不包含

1. 宠物主问卷设计。
2. 多猫互动。
3. 人猫复杂互动。
4. CAD / DXF 自动上色模块。
5. 职业人群画像模块。
6. 真实医学诊断或健康建议。
7. 基于真实传感器数据的行为校准。

---

## 5. 核心设计：猫咪 Agent 分层模型

猫咪 Agent 应拆分为四层：

```text
客观档案层 Objective Profile
年龄、性别、绝育状态、疾病史、体型、行动能力

人格层 Personality
安全感 / 好奇活跃度 / 地盘意识 / 冲动性 / 亲人程度

动态状态层 Dynamic State
能量、饥饿、压力、无聊、安全感、社交需求、满意度

行为决策层 Behavior Decision
休息、进食、探索、观望、躲藏、奔跑、亲近、占位、游走
```

其中：

- 客观档案层用于提供长期约束。
- 人格层用于提供稳定倾向。
- 动态状态层用于描述当前时刻的需求。
- 行为决策层根据上述变量选择当前目标与行为。

---

## 6. 数据结构设计

### 6.1 CatProfile

新增猫咪档案结构 `CatProfile`。

```python
cat_profile = {
    "name": "cat_01",

    "objective": {
        "age_stage": "adult",
        "sex": "female",
        "neutered": True,
        "body_condition": "normal",
        "mobility_level": 1.0,
        "vision_level": 1.0,
        "hearing_level": 1.0,
        "disease_history": []
    },

    "personality": {
        "neuroticism": 0.50,
        "extraversion": 0.50,
        "dominance": 0.50,
        "impulsiveness": 0.50,
        "agreeableness": 0.50
    }
}
```

### 6.2 年龄阶段

```python
age_stage = ["kitten", "young", "adult", "senior"]
```

| 枚举值 | 含义 | 行为倾向 |
|---|---|---|
| kitten | 幼猫 | 高活动、高探索、高玩耍、休息频率高 |
| young | 青年猫 | 高活动、高探索、高奔跑 |
| adult | 成年猫 | 行为较稳定，作为默认基准 |
| senior | 老年猫 | 低移动速度、低跳跃、高休息、偏好安静区域 |

### 6.3 性别与绝育状态

```python
sex = ["male", "female", "unknown"]
neutered = True / False / None
```

要求：

- 性别本身不应产生过强行为差异。
- 性别应与绝育状态、年龄、地盘意识共同使用。
- 避免写死“公猫一定怎样、母猫一定怎样”的刻板规则。
- 未绝育猫可略微提高巡逻、标记、地盘相关行为权重。
- 已绝育猫以默认行为为主。

### 6.4 疾病史

疾病史用于模拟行为约束，不用于医学判断。

```python
disease_history = [
    "arthritis",
    "obesity",
    "vision_impairment",
    "hearing_impairment",
    "urinary_issue",
    "chronic_pain",
    "none"
]
```

| 疾病史 | 行为影响 |
|---|---|
| arthritis | 降低移动速度、跳跃倾向、上高处概率，提高休息概率 |
| obesity | 降低移动速度、奔跑概率、活动半径 |
| vision_impairment | 降低探索新区域概率，提高贴边移动与谨慎行为 |
| hearing_impairment | 降低对声音事件反应，提高被接近时的突发反应概率 |
| urinary_issue | 提高猫砂盆相关区域访问频率，降低满意度基线 |
| chronic_pain | 提高易怒、躲藏、低亲近行为概率，降低活动量 |

---

## 7. 五维猫格参数

本阶段内部仍使用论文对应的五维变量名，面向用户展示名称后续另行设计。

| 内部变量 | 含义 | 高分行为倾向 |
|---|---|---|
| neuroticism | 敏感 / 焦虑 / 安全需求 | 更容易躲藏、受惊、避开人和开放空间 |
| extraversion | 好奇 / 活跃 / 探索 | 更爱探索、玩耍、靠近窗边、巡视空间 |
| dominance | 地盘意识 / 支配性 | 更爱占据高处、资源点、固定位置 |
| impulsiveness | 冲动性 / 不稳定性 | 更容易突然奔跑、频繁切换目标、路径随机 |
| agreeableness | 亲人程度 / 温和性 | 更愿意靠近人、停留共享空间、接受互动 |

所有值范围为 `0.0 - 1.0`。

---

## 8. 动态状态设计

在 `CatAgent` 中新增动态状态。

```python
self.state = {
    "energy": 1.0,
    "satisfaction": 0.5,
    "hunger": 0.2,
    "stress": 0.2,
    "boredom": 0.3,
    "security": 0.6,
    "social_need": 0.3
}
```

| 状态 | 含义 | 变化逻辑 |
|---|---|---|
| energy | 体力 | 移动下降，休息和进食恢复 |
| satisfaction | 满意度 | 到达偏好区域、满足需求后上升 |
| hunger | 饥饿 | 随时间上升，进食下降 |
| stress | 压力 | 受噪音、人类靠近、陌生区域影响上升，休息和躲藏下降 |
| boredom | 无聊 | 长时间无探索、无玩耍时上升 |
| security | 安全感 | 在隐蔽区、熟悉区、高处上升，在开放区或受刺激下降 |
| social_need | 社交需求 | 受亲人程度影响，长时间无互动时上升 |

---

## 9. 行为库设计

```python
behaviors = [
    "rest",
    "feed",
    "explore",
    "watch_window",
    "hide",
    "run",
    "wander",
    "claim_spot",
    "approach_human"
]
```

| 行为 | 中文显示 | 说明 |
|---|---|---|
| rest | 休息 | 在猫休息区、安静区域停留并恢复能量 |
| feed | 进食 | 前往喂食区，降低饥饿并恢复能量 |
| explore | 探索 | 前往空白区、窗边、共享空间等 |
| watch_window | 观望 | 停留窗边观察 |
| hide | 躲藏 | 前往隐蔽区域或猫休息区 |
| run | 奔跑 | 快速移动，短时间高速度 |
| wander | 游走 | 无明确需求时低强度移动 |
| claim_spot | 占位 | 偏好高处、猫窝、食盆附近等资源点 |
| approach_human | 亲近人 | 前往人类附近或共享空间 |

---

## 10. 行为决策逻辑

### 10.1 目标

将现有固定目标列表替换为行为权重选择。

```python
weights = calculate_behavior_weights()
behavior = weighted_random_choice(weights)
goal_zone = behavior_to_zone(behavior)
```

### 10.2 行为权重示例

```python
weights = {
    "rest": 1.0 + 2.0 * (1 - energy) + 1.0 * senior_factor,
    "feed": 1.0 + 2.5 * hunger,
    "hide": 0.5 + 2.0 * neuroticism + 1.5 * stress - 0.8 * agreeableness,
    "explore": 0.8 + 2.0 * extraversion + 1.0 * boredom - 1.0 * neuroticism,
    "watch_window": 0.5 + 1.5 * extraversion + 0.5 * boredom,
    "run": 0.2 + 1.5 * impulsiveness + 0.8 * extraversion - 0.8 * senior_factor,
    "wander": 0.8 + 0.5 * extraversion,
    "claim_spot": 0.3 + 1.8 * dominance,
    "approach_human": 0.3 + 2.0 * agreeableness + 0.8 * extraversion - 1.2 * neuroticism
}
```

所有权重需经过下限保护：

```python
weight = max(0.01, weight)
```

---

## 11. 空间偏好映射

```python
behavior_zone_map = {
    "rest": ["cat_rest", "human_sleep"],
    "feed": ["cat_feeding"],
    "hide": ["cat_rest", "human_sleep", "empty"],
    "explore": ["empty", "shared", "window"],
    "watch_window": ["window"],
    "run": ["shared", "empty"],
    "wander": ["empty", "shared"],
    "claim_spot": ["cat_rest", "cat_feeding", "window"],
    "approach_human": ["shared", "human_work", "human_sleep"]
}
```

后续如果引入更细的空间语义，可扩展为：

- 高处点
- 床底
- 沙发后
- 门口
- 窗台
- 猫爬架
- 猫砂盆
- 食盆
- 水碗

---

## 12. 移动参数调制

### 12.1 基础移动速度

```python
base_speed = CAT_CONFIG["step_length_m"]
speed_multiplier = (
    1.0
    + 0.25 * extraversion
    + 0.20 * impulsiveness
    - 0.15 * neuroticism
    - 0.25 * senior_factor
    - 0.20 * mobility_penalty
)
step_length = base_speed * clamp(speed_multiplier, 0.4, 1.8)
```

### 12.2 奔跑概率

```python
run_probability = (
    0.05
    + 0.25 * impulsiveness
    + 0.15 * extraversion
    - 0.15 * senior_factor
    - 0.20 * mobility_penalty
)
run_probability = clamp(run_probability, 0.01, 0.60)
```

### 12.3 行为切换频率

高冲动猫更频繁切换目标。

```python
goal_change_limit = int(
    220
    - 100 * impulsiveness
    + 80 * neuroticism
    + 60 * senior_factor
)
goal_change_limit = clamp(goal_change_limit, 60, 300)
```

---

## 13. 客观条件调制规则

### 13.1 年龄调制

```python
age_modifiers = {
    "kitten": {
        "speed": 1.10,
        "run": 1.20,
        "rest": 1.20,
        "explore": 1.20
    },
    "young": {
        "speed": 1.15,
        "run": 1.25,
        "explore": 1.15
    },
    "adult": {
        "speed": 1.00,
        "run": 1.00,
        "explore": 1.00
    },
    "senior": {
        "speed": 0.70,
        "run": 0.50,
        "rest": 1.50,
        "explore": 0.70
    }
}
```

### 13.2 疾病史调制

```python
disease_modifiers = {
    "arthritis": {
        "speed": 0.70,
        "run": 0.40,
        "vertical": 0.30,
        "rest": 1.40
    },
    "obesity": {
        "speed": 0.75,
        "run": 0.50,
        "explore": 0.80
    },
    "vision_impairment": {
        "explore": 0.70,
        "hide": 1.20,
        "stress": 1.20
    },
    "hearing_impairment": {
        "sound_reactivity": 0.50,
        "surprise_reactivity": 1.30
    },
    "urinary_issue": {
        "litter_related_visit": 1.50,
        "satisfaction": 0.90
    },
    "chronic_pain": {
        "speed": 0.70,
        "rest": 1.40,
        "agreeable_behavior": 0.70,
        "irritability": 1.30
    }
}
```

注：当前代码尚无猫砂盆、高处、噪声事件等对象，可先保留字段与调制接口，实际逻辑逐步接入。

---

## 14. 行为统计输出

### 14.1 区域停留时间

```json
"zone_stay_ticks": {
    "cat_rest": 320,
    "cat_feeding": 80,
    "shared": 260,
    "window": 140,
    "empty": 200
}
```

### 14.2 行为频率

```json
"behavior_counts": {
    "rest": 30,
    "feed": 8,
    "explore": 42,
    "watch_window": 12,
    "hide": 18,
    "run": 20,
    "wander": 36,
    "claim_spot": 10,
    "approach_human": 6
}
```

### 14.3 猫格摘要

```json
"cat_profile_summary": {
    "age_stage": "senior",
    "disease_history": ["arthritis"],
    "personality": {
        "neuroticism": 0.70,
        "extraversion": 0.35,
        "dominance": 0.20,
        "impulsiveness": 0.30,
        "agreeableness": 0.60
    }
}
```

---

## 15. 文件输出

本阶段建议在原有 `simulation_result.png` 基础上新增：

```text
outputs/
├─ simulation_result.png
├─ tick_records.csv
├─ cat_behavior_summary.json
└─ cat_profile_used.json
```

### 15.1 tick_records.csv

每个 tick 一行，至少包含：

```text
tick, cat_x, cat_y, cat_zone, cat_behavior, cat_energy, cat_stress, cat_hunger, cat_boredom, human_x, human_y, human_behavior
```

### 15.2 cat_behavior_summary.json

保存区域停留、行为频率、行为持续时间、满意度等汇总结果。

### 15.3 cat_profile_used.json

保存本次模拟的猫咪输入参数，方便复现实验。

---

## 16. 推荐代码改造方案

### 16.1 最小改造文件

当前如果仍保持单文件结构，优先修改：

```text
CatAgent.__init__()
CatAgent.choose_new_goal()
CatAgent.move()
CatAgent.update_state_at_goal()
CatAgent.get_behavior()
CatAgent.step()
Simulation.run()
Simulation.visualize()
```

### 16.2 推荐新增函数

```python
def clamp(value, min_value, max_value):
    pass

def weighted_random_choice(weights):
    pass

def calculate_age_factor(age_stage):
    pass

def calculate_disease_modifiers(disease_history):
    pass
```

### 16.3 CatAgent 推荐新增方法

```python
class CatAgent:
    def update_dynamic_state(self):
        pass

    def calculate_behavior_weights(self):
        pass

    def choose_behavior(self):
        pass

    def behavior_to_goal_zone(self, behavior):
        pass

    def choose_new_goal_by_behavior(self):
        pass

    def calculate_step_length(self):
        pass

    def calculate_run_probability(self):
        pass

    def record_statistics(self):
        pass
```

---

## 17. 验收标准

### 17.1 功能验收

满足以下条件视为本阶段完成：

1. 可以通过输入不同 `cat_profile` 创建不同猫咪。
2. 五维人格参数可以影响猫的目标区域选择。
3. 年龄和疾病史可以影响猫的移动速度、奔跑概率和休息倾向。
4. 同一户型下，不同猫格输出的热力图和行为统计存在可观察差异。
5. 程序仍能正常输出轨迹图、猫热力图、人热力图和报告。
6. 程序新增输出 `cat_behavior_summary.json`。
7. 程序新增输出 `cat_profile_used.json`。
8. 不破坏现有人类 Agent 基础逻辑。

### 17.2 对比实验验收

#### 测试猫 A：敏感躲藏型

```python
personality = {
    "neuroticism": 0.85,
    "extraversion": 0.25,
    "dominance": 0.20,
    "impulsiveness": 0.35,
    "agreeableness": 0.30
}
```

预期：

- `hide` 和 `rest` 行为占比更高。
- 猫休息区、卧室、角落区域热力更高。
- 共享空间和窗户停留相对减少。

#### 测试猫 B：好奇跑酷型

```python
personality = {
    "neuroticism": 0.20,
    "extraversion": 0.90,
    "dominance": 0.35,
    "impulsiveness": 0.85,
    "agreeableness": 0.55
}
```

预期：

- `explore`、`run`、`watch_window` 行为占比更高。
- 轨迹范围更大。
- 窗户、共享空间、空白区域热力更高。

#### 测试猫 C：亲人陪伴型

```python
personality = {
    "neuroticism": 0.25,
    "extraversion": 0.55,
    "dominance": 0.20,
    "impulsiveness": 0.25,
    "agreeableness": 0.90
}
```

预期：

- `approach_human` 和 `shared` 相关行为增加。
- 人类活动区域附近热力更高。
- 行为切换较稳定。

#### 测试猫 D：老年关节炎猫

```python
objective = {
    "age_stage": "senior",
    "disease_history": ["arthritis"],
    "mobility_level": 0.6
}
```

预期：

- 移动速度下降。
- 奔跑频率明显下降。
- 休息行为增加。
- 活动半径缩小。

---

## 18. 开发优先级

### P0：必须完成

1. 支持 `cat_profile` 输入。
2. 五维人格影响目标区域权重。
3. 年龄影响速度、奔跑概率、休息权重。
4. 疾病史影响基础行为参数。
5. 输出行为统计 JSON。

### P1：建议完成

1. 扩展行为标签。
2. 输出 `tick_records.csv`。
3. 可视化报告中展示猫格摘要。
4. 支持多个预设猫格 profile。

### P2：后续增强

1. 人猫互动。
2. 多猫互动。
3. 猫砂盆、高处、床底、门口等细粒度空间对象。
4. 噪声、陌生人、喂食等事件系统。
5. 宠物主问卷映射。
6. 批量模拟与参数敏感性分析。

---

## 19. 风险与注意事项

1. 不要让年龄、性别、疾病史产生过度刻板化判断。
2. 疾病史只用于模拟行为约束，不用于医学诊断。
3. 性格参数应影响概率和权重，而不是硬编码行为。
4. 保持参数可解释，方便后续论文或项目说明。
5. 先实现最小可运行版本，再逐步增加复杂行为。
6. 所有新增逻辑应保留默认值，确保没有输入猫档案时程序仍可运行。
7. 输出结果应可复现，建议支持随机种子 `random_seed`。

---

## 20. 推荐开发顺序

```text
Step 1：新增 cat_profile 默认配置
Step 2：CatAgent 接收 personality 和 objective
Step 3：新增动态状态 stress / hunger / boredom / social_need
Step 4：用行为权重替换 choose_new_goal 中的固定目标列表
Step 5：根据人格和客观条件调制速度与奔跑概率
Step 6：记录区域停留和行为频率
Step 7：输出 JSON / CSV
Step 8：用四类测试猫进行对比实验
```

---

## 21. 本阶段最终产物

开发完成后，应得到：

1. 一个可输入猫咪档案的模拟程序。
2. 至少四个预设测试猫 profile。
3. 每只猫对应的轨迹图、热力图、行为统计 JSON。
4. 同一户型下不同猫格的对比结果。
5. 后续可接入宠物主问卷、人群画像和 CAD 自动上色模块的清晰接口。

---

## 22. 简要总结

本阶段的核心不是增加更多随机行为，而是建立猫咪行为的参数化决策系统。

目标效果是：

```text
同一住宅布局中，不同猫咪因为年龄、健康状态和五维猫格不同，
自然产生不同的移动速度、空间偏好、行为频率、轨迹范围和热力分布。
```

这将使项目从“人猫轨迹模拟”升级为“人宠共居空间适配研究工具”的核心模块。
