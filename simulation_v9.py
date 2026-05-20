"""
人宠共居模拟引擎 v9.0 - 完全独立运行版
Standalone Edition: 无需任何外部配置文件
功能：生成户型图 → 运行双智能体模拟 → 输出轨迹图+热力图+报告
作者：基于ABM的人宠共居空间优化研究
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/human-cat-mplconfig")

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面后端，服务器/笔记本通用
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from heapq import heappush, heappop
import random
import csv
import json
from copy import deepcopy

from project_paths import ensure_project_dir, project_relative_display, resolve_project_path
from time_use_parameter_builder import TimeUseParameterBuilder

# ===================== 中文字体支持 =====================
for font_path in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 内置区域规则（替代config文件） =====================
ZONE_COLOR_RULES = {
    "wall":         {"name": "墙体",   "passable_human": False, "passable_cat": False},
    "partition":    {"name": "隔断",   "passable_human": False, "passable_cat": False},
    "window":       {"name": "窗户",   "passable_human": True,  "passable_cat": True},
    "human_sleep":  {"name": "卧室",   "passable_human": True,  "passable_cat": True},
    "human_work":   {"name": "工作区", "passable_human": True,  "passable_cat": True},
    "cat_rest":     {"name": "猫休息区","passable_human": True,  "passable_cat": True},
    "cat_feeding":  {"name": "喂食区", "passable_human": True,  "passable_cat": True},
    "shared":       {"name": "共享空间","passable_human": True,  "passable_cat": True},
    "empty":        {"name": "空白",   "passable_human": True,  "passable_cat": True},
}

# ===================== 彩虹热力图配色 =====================
def create_rainbow_colormap():
    colors = [
        (0.0, 0.0, 0.3), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0),
    ]
    return LinearSegmentedColormap.from_list("rainbow", colors)

RAINBOW_CMAP = create_rainbow_colormap()


def _dominant_cat_behavior_for_cells(analyzer, member_cells):
    """从节点成员格栅的猫行为频次里取主导行为。"""
    counts = {}
    for cell in member_cells:
        for behavior, count in analyzer.cat_behavior_grid.get(cell, {}).items():
            counts[behavior] = counts.get(behavior, 0) + count
    if not counts:
        return "未知行为"
    return max(counts.items(), key=lambda item: item[1])[0]


def _cat_function_type(behavior):
    """把主导行为映射到策略卡模块使用的功能类型。"""
    return {
        "奔跑": "运动型",
        "玩耍": "玩耍型",
        "探索": "探索型",
        "躲藏": "庇护型",
        "观察": "观察型",
        "观望": "观察型",
        "休息": "休息型",
        "睡眠": "休息型",
        "进食": "进食型",
        "抓挠": "抓挠型",
        "占位": "抓挠型",
        "亲近": "玩耍型",
    }.get(behavior, "探索型")


def _entropy_comp_label(entropy_value):
    if entropy_value >= 1.0:
        return "功能高度复合型"
    if entropy_value >= 0.5:
        return "功能复合型"
    return "功能单一型"


def _zone_for_node(sim, analyzer, node):
    """用节点质心格栅反查原始户型 zone。"""
    gx = int(np.clip(round(node.centroid_x), 0, analyzer.grid_width - 1))
    gy = int(np.clip(round(node.centroid_y), 0, analyzer.grid_height - 1))
    px = int(np.clip((gx + 0.5) / analyzer.grid_width * analyzer.source_width_px, 0, analyzer.source_width_px - 1))
    py = int(np.clip((gy + 0.5) / analyzer.grid_height * analyzer.source_height_px, 0, analyzer.source_height_px - 1))
    return str(sim.zone_map[py, px])


def build_strategy_card_nodes(nodes, analyzer, sim):
    """
    将模块 C 的 SpaceNode 转成 strategy_cards_module 约定的节点画像。

    strategy_cards_module 保持独立不改；这里仅做字段适配。
    """
    card_nodes = []
    for node in nodes:
        zone = _zone_for_node(sim, analyzer, node)
        if node.avg_cat_intensity > 0.1 and node.node_type != "人专属":
            behavior = _dominant_cat_behavior_for_cells(analyzer, node.member_cells)
            card_nodes.append({
                "nid": f"H{node.node_id}",
                "type": "猫节点",
                "zone": zone,
                "cfs": float(node.avg_cat_intensity),
                "dom_bh": behavior,
                "func": _cat_function_type(behavior),
                "comp": _entropy_comp_label(node.avg_cat_entropy),
            })

        if node.node_type == "冲突节点" or node.avg_cooc_active > 0:
            risk = "高风险" if node.avg_cooc_active >= 1.0 else "中风险"
            card_nodes.append({
                "nid": f"S{node.node_id}",
                "type": "共现节点",
                "zone": zone,
                "dcd_max": float(node.avg_cooc_active),
                "risk": risk,
                "conflict_mechanism": "相向交汇",
            })
    return card_nodes


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def weighted_random_choice(weights):
    valid_items = [(key, max(0.0, float(weight))) for key, weight in weights.items()]
    total = sum(weight for _, weight in valid_items)
    if total <= 0:
        return random.choice(list(weights.keys()))
    pick = random.uniform(0, total)
    current = 0.0
    for key, weight in valid_items:
        current += weight
        if pick <= current:
            return key
    return valid_items[-1][0]


def normalize_cat_profile(cat_profile=None):
    profile = deepcopy(DEFAULT_CAT_PROFILE)
    if cat_profile:
        profile["name"] = cat_profile.get("name", profile["name"])
        for section in ("objective", "personality"):
            profile[section].update(cat_profile.get(section, {}))

    objective = profile["objective"]
    personality = profile["personality"]
    if objective.get("age_stage") not in {"kitten", "young", "adult", "senior"}:
        objective["age_stage"] = "adult"
    if objective.get("sex") not in {"male", "female", "unknown"}:
        objective["sex"] = "unknown"
    if objective.get("body_condition") not in {"thin", "normal", "overweight", "obese"}:
        objective["body_condition"] = "normal"
    for key in ("mobility_level", "vision_level", "hearing_level"):
        objective[key] = clamp(float(objective.get(key, 1.0)), 0.2, 1.2)
    if objective.get("neutered") not in {True, False, None}:
        objective["neutered"] = True
    diseases = objective.get("disease_history", [])
    if diseases is None:
        diseases = []
    if isinstance(diseases, str):
        diseases = [] if diseases == "none" else [diseases]
    objective["disease_history"] = [d for d in diseases if d and d != "none"]

    for key in DEFAULT_CAT_PROFILE["personality"]:
        personality[key] = clamp(float(personality.get(key, 0.5)))
    return profile


def get_cat_preset_profiles():
    return {name: normalize_cat_profile(profile) for name, profile in CAT_PRESET_PROFILES.items()}

# ===================== 智能体配置参数 =====================
CAT_CONFIG = {
    "initial_energy": 1.0,
    "initial_satisfaction": 0.5,
    "energy_consume_per_tick": 0.0001,
    "rest_energy_recover": 0.03,
    "feed_energy_recover": 0.05,
    "step_length_m": 0.5,
    "max_turn_angle": 45,
    "window_attraction": 0.8,
    "run_probability": 0.3,
    "jump_distance_multiplier": 2.5,
    "heatmap_weight": 5.0,
    "run_heatmap_weight": 8.0,
}

CAT_BEHAVIOR_LABELS = {
    "rest": "休息",
    "feed": "进食",
    "explore": "探索",
    "watch_window": "观察",
    "hide": "躲藏",
    "run": "奔跑",
    "wander": "玩耍",
    "claim_spot": "抓挠",
    "approach_human": "玩耍",
}

# 输出层 canonical 行为：
# - 观望 -> 观察
# - 占位 -> 抓挠
# - 游走 / 亲近 -> 玩耍
# 这样最终暴露给分析和可视化的是稳定的八类行为。
CAT_BEHAVIOR_GROUP_LABELS = {
    "玩耍": {"游走", "亲近"},
    "抓挠": {"占位"},
    "观察": {"观望"},
}
CAT_BEHAVIOR_GROUP_LOOKUP = {
    raw_label: group_label
    for group_label, raw_labels in CAT_BEHAVIOR_GROUP_LABELS.items()
    for raw_label in raw_labels
}


def summarize_cat_behavior(label):
    return CAT_BEHAVIOR_GROUP_LOOKUP.get(label, label)


DEFAULT_CAT_PROFILE = {
    "name": "default_adult",
    "objective": {
        "age_stage": "adult",
        "sex": "unknown",
        "neutered": True,
        "body_condition": "normal",
        "mobility_level": 1.0,
        "vision_level": 1.0,
        "hearing_level": 1.0,
        "disease_history": [],
    },
    "personality": {
        "neuroticism": 0.50,
        "extraversion": 0.50,
        "dominance": 0.50,
        "impulsiveness": 0.50,
        "agreeableness": 0.50,
    },
}

CAT_PRESET_PROFILES = {
    "sensitive_hiding": {
        "name": "sensitive_hiding",
        "objective": {"age_stage": "adult", "neutered": True, "mobility_level": 0.95},
        "personality": {
            "neuroticism": 0.88,
            "extraversion": 0.25,
            "dominance": 0.35,
            "impulsiveness": 0.35,
            "agreeableness": 0.25,
        },
    },
    "curious_active": {
        "name": "curious_active",
        "objective": {"age_stage": "young", "neutered": True, "mobility_level": 1.0},
        "personality": {
            "neuroticism": 0.25,
            "extraversion": 0.90,
            "dominance": 0.45,
            "impulsiveness": 0.82,
            "agreeableness": 0.55,
        },
    },
    "friendly_companion": {
        "name": "friendly_companion",
        "objective": {"age_stage": "adult", "neutered": True, "mobility_level": 1.0},
        "personality": {
            "neuroticism": 0.22,
            "extraversion": 0.60,
            "dominance": 0.30,
            "impulsiveness": 0.35,
            "agreeableness": 0.90,
        },
    },
    "senior_arthritis": {
        "name": "senior_arthritis",
        "objective": {
            "age_stage": "senior",
            "neutered": True,
            "body_condition": "overweight",
            "mobility_level": 0.55,
            "disease_history": ["arthritis"],
        },
        "personality": {
            "neuroticism": 0.55,
            "extraversion": 0.25,
            "dominance": 0.45,
            "impulsiveness": 0.20,
            "agreeableness": 0.55,
        },
    },
}

HUMAN_CONFIG = {
    "initial_satisfaction": 0.5,
    "step_length_m": 0.7,
    "max_turn_angle": 25,
    "heatmap_weight": 4.0,
    "random_walk_steps": (50, 150),
    "use_internal_path": True,
    "path_offset_range": 0.3,
    "activity_patterns": {
        "morning":  {"zones": ["human_sleep", "shared", "cat_feeding"], "stay_probability": 0.3},
        "daytime":  {"zones": ["human_work", "shared", "window"],     "stay_probability": 0.4},
        "evening":  {"zones": ["shared", "human_sleep", "window"],    "stay_probability": 0.35},
    },
}


# ===================== 户型图生成器 =====================
def generate_floor_plan(save_path="floor_plan.png", size=200):
    """
    自动生成标准户型图（与你提供的原始户型完全一致）
    包含：卧室(粉)、工作区(绿)、猫休息区(橙)、喂食区(深红)、共享空间(黄)、窗户(蓝)
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # 严格对应识别区间的RGB颜色
    WALL = [20, 20, 20]
    PARTITION = [120, 120, 120]
    WINDOW = [30, 30, 230]
    HUMAN_SLEEP = [250, 195, 210]
    HUMAN_WORK = [30, 230, 30]
    CAT_REST = [250, 165, 30]
    CAT_FEEDING = [160, 50, 50]
    SHARED = [250, 250, 30]

    # 外墙（黑色，留出5像素边）
    img[0:5, :, :] = WALL
    img[-5:, :, :] = WALL
    img[:, 0:5, :] = WALL
    img[:, -5:, :] = WALL

    # 内部隔断（灰色）
    img[80:85, 5:150, :] = PARTITION      # 水平：客厅/卧室分隔
    img[5:80, 100:105, :] = PARTITION     # 垂直：卧室/工作区分隔

    # 窗户（蓝色）- 右侧中间
    img[90:110, 195:200, :] = WINDOW

    # 功能区域（严格匹配代码识别区间）
    img[10:75, 10:95, :] = HUMAN_SLEEP        # 卧室（左上）
    img[10:75, 110:145, :] = HUMAN_WORK       # 工作区（右上）
    img[150:190, 10:60, :] = CAT_REST         # 猫休息区（左下）
    img[140:170, 70:110, :] = CAT_FEEDING     # 喂食区（中下）
    img[90:130, 60:140, :] = SHARED           # 共享空间/客厅（中央）

    Image.fromarray(img).save(save_path)
    print(f"[户型图生成] 已保存: {save_path}")
    return save_path


# ===================== 户型图解析器 =====================
class FloorPlanParser:
    def __init__(self, floor_plan_path, house_width_m=16.5, house_depth_m=16.5):
        self.floor_plan_path = floor_plan_path
        self.house_width_m = house_width_m
        self.house_depth_m = house_depth_m
        self.max_image_pixels = 600

    def parse(self):
        img = imread(self.floor_plan_path)
        print(f"[解析] 户型图尺寸: {img.shape}")

        if len(img.shape) == 4:    img = img[:, :, :3]
        elif len(img.shape) == 2:  img = np.stack([img, img, img], axis=-1)
        if img.dtype in [np.float32, np.float64]: img = (img * 255).astype(np.uint8)

        img = self._resize_image(img, self.max_image_pixels)
        self.img_height, self.img_width = img.shape[:2]
        self.scale_x = self.house_width_m / self.img_width
        self.scale_y = self.house_depth_m / self.img_height
        print(f"[解析] 1像素 = {self.scale_x:.4f}m(宽) / {self.scale_y:.4f}m(深)")

        zone_map = np.full((self.img_height, self.img_width), "empty", dtype=object)
        passable_human = np.ones((self.img_height, self.img_width), dtype=bool)
        passable_cat = np.ones((self.img_height, self.img_width), dtype=bool)

        # 颜色识别区间（与代码完全对应）
        zone_colors = {
            "wall":         ([0, 0, 0],         [30, 30, 30]),
            "partition":    ([100, 100, 100],   [140, 140, 140]),
            "window":       ([0, 0, 200],       [50, 50, 255]),
            "human_sleep":  ([240, 180, 200],   [255, 210, 220]),
            "human_work":   ([0, 200, 0],       [50, 255, 50]),
            "cat_rest":     ([240, 150, 0],     [255, 180, 50]),
            "cat_feeding":  ([140, 40, 40],     [180, 60, 60]),
            "shared":       ([240, 240, 0],     [255, 255, 50]),
        }

        zone_stats = {}
        for zone_name, (min_c, max_c) in zone_colors.items():
            min_c, max_c = np.array(min_c), np.array(max_c)
            match = np.all((img >= min_c) & (img <= max_c), axis=-1)
            zone_map[match] = zone_name
            rule = ZONE_COLOR_RULES[zone_name]
            passable_human[match] = rule["passable_human"]
            passable_cat[match] = rule["passable_cat"]

            pixels = np.count_nonzero(match)
            area = pixels * self.scale_x * self.scale_y
            zone_stats[zone_name] = {"pixels": pixels, "area_m2": area}
            if pixels > 0:
                print(f"       {rule['name']:8s}: {pixels:5d}像素 | {area:6.1f}㎡")

        empty_pixels = np.count_nonzero(zone_map == "empty")
        print(f"       {'空白':8s}: {empty_pixels:5d}像素 | {empty_pixels * self.scale_x * self.scale_y:6.1f}㎡")

        passable_maps = {"human": passable_human, "cat": passable_cat}
        print(f"[解析] 可通行: 人 {np.count_nonzero(passable_human)}像素 | 猫 {np.count_nonzero(passable_cat)}像素")
        return img, zone_map, passable_maps, zone_stats, self.scale_x, self.scale_y

    def _resize_image(self, img, max_size):
        h, w = img.shape[:2]
        if h <= max_size and w <= max_size: return img.astype(np.uint8)
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = Image.fromarray(np.uint8(img))
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(img_pil, dtype=np.uint8)


# ===================== A*路径规划器 =====================
class PathFinder:
    def __init__(self, passable_map):
        self.passable_map = passable_map
        self.h, self.w = passable_map.shape
        # 预先估计可通行像素到最近障碍的距离，用于给贴墙路径增加代价。
        self.wall_clearance = distance_transform_edt(passable_map.astype(np.uint8))
        self.preferred_clearance_px = 8.0
        self.wall_penalty_scale = 0.12

    def find_path(self, start_x, start_y, goal_x, goal_y, use_internal=False):
        start = (int(start_y), int(start_x))
        goal = (int(goal_y), int(goal_x))
        if not self.passable_map[start[0], start[1]]: return []
        if not self.passable_map[goal[0], goal[1]]:
            goal = self._find_nearest_passable(goal[1], goal[0])
            if goal is None: return []

        if use_internal:
            mid = self._get_internal_midpoint(start, goal)
            if mid:
                p1 = self._find_path_segment(start, mid)
                p2 = self._find_path_segment(mid, goal)
                if p1 and p2: return p1 + p2[1:]
        return self._find_path_segment(start, goal)

    def _get_internal_midpoint(self, start, goal):
        mid_y, mid_x = (start[0] + goal[0]) // 2, (start[1] + goal[1]) // 2
        for _ in range(10):
            oy = random.randint(-20, 20)
            ox = random.randint(-20, 20)
            ny, nx = mid_y + oy, mid_x + ox
            if 0 <= ny < self.h and 0 <= nx < self.w and self.passable_map[ny, nx]:
                return (ny, nx)
        if 0 <= mid_y < self.h and 0 <= mid_x < self.w and self.passable_map[mid_y, mid_x]:
            return (mid_y, mid_x)
        return None

    def _find_path_segment(self, start, goal):
        open_set = []
        heappush(open_set, (0, start))
        came_from, g_score = {}, {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        visited = set()
        while open_set:
            _, cur = heappop(open_set)
            if cur in visited: continue
            visited.add(cur)
            if cur == goal: return self._reconstruct_path(came_from, cur)
            for nb in self._get_neighbors(cur):
                if nb in visited: continue
                tg = g_score[cur] + self._step_cost(cur, nb)
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = cur
                    g_score[nb] = tg
                    f_score[nb] = tg + self._heuristic(nb, goal)
                    heappush(open_set, (f_score[nb], nb))
        return []

    def _find_nearest_passable(self, x, y):
        for r in range(1, 20):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < self.h and 0 <= nx < self.w and self.passable_map[ny, nx]:
                        return (ny, nx)
        return None

    def _heuristic(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _step_cost(self, cur, nb):
        dy = nb[0] - cur[0]
        dx = nb[1] - cur[1]
        base_cost = float(np.hypot(dy, dx))
        clearance = float(self.wall_clearance[nb[0], nb[1]])
        wall_penalty = max(0.0, self.preferred_clearance_px - clearance)
        wall_penalty = (wall_penalty ** 2) * self.wall_penalty_scale
        return base_cost + wall_penalty

    def _get_neighbors(self, pos):
        y, x = pos
        nbs = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = y+dy, x+dx
            if not (0 <= ny < self.h and 0 <= nx < self.w):
                continue
            if not self.passable_map[ny, nx]:
                continue
            # 禁止沿墙角斜穿，避免路径贴着障碍切角。
            if dx != 0 and dy != 0:
                if not (self.passable_map[y, nx] and self.passable_map[ny, x]):
                    continue
            nbs.append((ny, nx))
        return nbs

    def _reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return [(x, y) for y, x in path]


# ===================== 猫智能体 =====================
class CatAgent:
    def __init__(self, start_x, start_y, zone_map, passable_map, scale_x, scale_y, cat_profile=None):
        self.zone_map, self.passable_map = zone_map, passable_map
        self.scale_x, self.scale_y = scale_x, scale_y
        self.h, self.w = zone_map.shape
        self.x, self.y = float(start_x), float(start_y)
        self.cat_profile = normalize_cat_profile(cat_profile)
        self.objective = self.cat_profile["objective"]
        self.personality = self.cat_profile["personality"]
        self.state = {
            "energy": CAT_CONFIG["initial_energy"],
            "satisfaction": CAT_CONFIG["initial_satisfaction"],
            "hunger": 0.25,
            "stress": 0.25 + self.personality["neuroticism"] * 0.15,
            "boredom": 0.35,
            "security": 0.65 - self.personality["neuroticism"] * 0.15,
            "social_need": 0.35 + self.personality["agreeableness"] * 0.20,
        }
        self.angle = random.uniform(0, 2*np.pi)
        self.trajectory = [(self.x, self.y)]
        self.visit_count = np.zeros((self.h, self.w), dtype=float)
        self._add_heatmap(int(self.y), int(self.x), CAT_CONFIG["heatmap_weight"]*3)
        self.goal_x = self.goal_y = None
        self.steps_since_goal_change = 0
        self.is_running = False
        self.run_steps_remaining = 0
        self.hide_hold_remaining = 0
        self.current_behavior = "wander"
        # 休息驱动初值与年龄目标保持一致，避免默认成人基线把日内分布拉回旧值。
        self.rest_drive = self._rest_target_ratio()
        self.zone_stay_ticks = {}
        self.behavior_counts = {}
        self.behavior_group_counts = {}
        self.behavior_durations = {}
        self.behavior_group_durations = {}
        self.choose_new_goal()

    @property
    def energy(self):
        return self.state["energy"]

    @energy.setter
    def energy(self, value):
        self.state["energy"] = clamp(value, 0.0, 1.0)

    @property
    def satisfaction(self):
        return self.state["satisfaction"]

    @satisfaction.setter
    def satisfaction(self, value):
        self.state["satisfaction"] = clamp(value, 0.0, 1.0)

    def _add_heatmap(self, y, x, weight):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.visit_count[y, x] += weight
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    self.visit_count[ny, nx] += weight * (0.5 / (1 + np.sqrt(dy*dy+dx*dx)))

    def _age_modifiers(self):
        age = self.objective["age_stage"]
        modifiers = {
            "speed": 1.0,
            "run": 1.0,
            "rest": 1.0,
            "explore": 1.0,
            "hide": 1.0,
            "switch_limit": 1.0,
        }
        if age == "kitten":
            modifiers.update({"speed": 1.05, "run": 1.15, "rest": 1.20, "explore": 1.25, "switch_limit": 0.80})
        elif age == "young":
            modifiers.update({"speed": 1.10, "run": 1.25, "explore": 1.25, "switch_limit": 0.85})
        elif age == "senior":
            modifiers.update({"speed": 0.72, "run": 0.45, "rest": 1.55, "explore": 0.65, "hide": 1.15, "switch_limit": 1.30})
        return modifiers

    def _rest_target_ratio(self):
        age = self.objective["age_stage"]
        if age == "kitten":
            return 0.33
        if age == "young":
            return 0.35
        if age == "adult":
            return 0.37
        if age == "senior":
            return 0.42
        return 0.37

    def _disease_modifiers(self):
        diseases = set(self.objective.get("disease_history", []))
        modifiers = {
            "speed": self.objective["mobility_level"],
            "run": self.objective["mobility_level"],
            "rest": 1.0,
            "explore": 1.0,
            "hide": 1.0,
            "satisfaction": 1.0,
        }
        if "arthritis" in diseases:
            modifiers.update({
                "speed": modifiers["speed"] * 0.70,
                "run": modifiers["run"] * 0.45,
                "rest": modifiers["rest"] * 1.45,
                "explore": modifiers["explore"] * 0.70,
            })
        if "obesity" in diseases or self.objective["body_condition"] in {"overweight", "obese"}:
            modifiers["speed"] *= 0.85
            modifiers["run"] *= 0.65
            modifiers["rest"] *= 1.15
        if "vision_impairment" in diseases:
            modifiers["explore"] *= 0.70
            modifiers["hide"] *= 1.20
        if "chronic_pain" in diseases:
            modifiers["speed"] *= 0.80
            modifiers["run"] *= 0.60
            modifiers["hide"] *= 1.30
            modifiers["satisfaction"] *= 0.90
        return modifiers

    def calculate_behavior_weights(self):
        p, s = self.personality, self.state
        age_mod = self._age_modifiers()
        disease_mod = self._disease_modifiers()
        cur_zone = self.zone_map[int(self.y), int(self.x)]

        weights = {
            "rest": 6.00 + self.rest_drive * 12.00 + (1.0 - s["energy"]) * 2.0 + s["stress"] * 0.70,
            "feed": 0.18 + s["hunger"] * 3.05,
            "explore": 0.48 + p["extraversion"] * 1.45 + s["boredom"] * 1.45,
            "watch_window": 0.25 + p["extraversion"] * 0.65 + s["boredom"] * 0.60,
            "hide": 0.28 + p["neuroticism"] * 1.45 + s["stress"] * 1.55 + (1.0 - s["security"]) * 1.05,
            "run": 0.08 + p["impulsiveness"] * 0.70 + p["extraversion"] * 0.45 + s["boredom"] * 0.30,
            "wander": 0.90 + (1.0 - abs(s["hunger"] - 0.45)) * 0.45,
            "claim_spot": 0.08 + p["dominance"] * 0.75,
            "approach_human": 0.15 + p["agreeableness"] * 1.55 + s["social_need"] * 1.10 - p["neuroticism"] * 0.45,
        }

        if self.objective.get("neutered") is False:
            weights["claim_spot"] += 0.55 + p["dominance"] * 0.50
            weights["wander"] += 0.20
        if cur_zone == "cat_feeding":
            weights["feed"] += 0.55
        if cur_zone == "cat_rest":
            weights["rest"] += 0.55
            weights["hide"] += 0.52 + s["security"] * 0.45 + (1.0 - s["stress"]) * 0.20
        if cur_zone == "window":
            weights["watch_window"] += 0.65
        if cur_zone in {"human_sleep", "human_work", "shared"}:
            weights["approach_human"] += p["agreeableness"] * 0.45
            weights["hide"] += p["neuroticism"] * 0.25
        if self.current_behavior == "hide":
            weights["hide"] += 0.28 + s["security"] * 0.32 + (1.0 - s["stress"]) * 0.18

        weights["rest"] *= age_mod["rest"] * disease_mod["rest"]
        weights["explore"] *= age_mod["explore"] * disease_mod["explore"]
        weights["hide"] *= age_mod["hide"] * disease_mod["hide"]
        weights["run"] *= age_mod["run"] * disease_mod["run"]
        total_ticks = sum(self.behavior_counts.values())
        if total_ticks > 20:
            rest_ratio = self.behavior_counts.get(CAT_BEHAVIOR_LABELS["rest"], 0) / total_ticks
            rest_target = self._rest_target_ratio()
            if rest_ratio > rest_target + 0.05:
                weights["rest"] *= max(0.10, 1.0 - (rest_ratio - rest_target) * 8.0)
        return {key: clamp(value, 0.01, 8.0) for key, value in weights.items()}

    def behavior_to_zones(self, behavior):
        mapping = {
            "rest": ["cat_rest", "human_sleep", "empty"],
            "feed": ["cat_feeding"],
            "explore": ["empty", "shared", "window", "human_work"],
            "watch_window": ["window"],
            "hide": ["cat_rest"],
            "run": ["empty", "shared"],
            "wander": ["empty", "shared", "window", "cat_rest"],
            "claim_spot": ["cat_rest", "window", "shared", "cat_feeding"],
            "approach_human": ["shared", "human_work", "human_sleep"],
        }
        return mapping.get(behavior, ["empty", "shared"])

    def choose_new_goal(self):
        coords = np.where(self.passable_map)
        if len(coords[0]) == 0: return
        cur_zone = self.zone_map[int(self.y), int(self.x)]
        total_ticks = max(1, sum(self.behavior_counts.values()))
        rest_ratio = self.behavior_counts.get(CAT_BEHAVIOR_LABELS["rest"], 0) / total_ticks
        rest_target = self._rest_target_ratio()
        if total_ticks > 20 and rest_ratio < rest_target - 0.02 and random.random() < 0.95:
            self.current_behavior = "rest"
        else:
            self.current_behavior = weighted_random_choice(self.calculate_behavior_weights())
        targets = self.behavior_to_zones(self.current_behavior)
        allow_same_zone = self.current_behavior == "hide"
        for tz in targets:
            if tz == cur_zone and not allow_same_zone:
                continue
            zc = np.where(self.zone_map == tz)
            if len(zc[0]) > 5:
                idx = random.randint(0, len(zc[0])-1)
                self.goal_x, self.goal_y = float(zc[1][idx]), float(zc[0][idx])
                self.steps_since_goal_change = 0
                return
        idx = random.randint(0, len(coords[0])-1)
        self.goal_y, self.goal_x = float(coords[0][idx]), float(coords[1][idx])
        self.steps_since_goal_change = 0

    def _goal_change_limit(self):
        impulsive = self.personality["impulsiveness"]
        age_mod = self._age_modifiers()
        base = 210 * age_mod["switch_limit"] * (1.25 - impulsive * 0.55)
        return int(clamp(base, 60, 280))

    def _step_length_px(self):
        p = self.personality
        age_mod = self._age_modifiers()
        disease_mod = self._disease_modifiers()
        speed_mod = age_mod["speed"] * disease_mod["speed"]
        speed_mod *= 0.88 + p["extraversion"] * 0.22 + p["impulsiveness"] * 0.10
        if self.state["energy"] < 0.25:
            speed_mod *= 0.75
        step_m = CAT_CONFIG["step_length_m"] * clamp(speed_mod, 0.30, 1.45)
        if self.is_running:
            step_m *= CAT_CONFIG["jump_distance_multiplier"]
        return step_m / self.scale_x

    def _run_probability(self):
        if self.current_behavior in {"rest", "feed", "hide"}:
            return 0.0
        p = self.personality
        age_mod = self._age_modifiers()
        disease_mod = self._disease_modifiers()
        prob = CAT_CONFIG["run_probability"]
        prob *= 0.005 + p["impulsiveness"] * 0.020 + p["extraversion"] * 0.015
        prob *= age_mod["run"] * disease_mod["run"]
        if self.current_behavior == "run":
            prob *= 2.50
        if self.state["energy"] < 0.30 or self.state["stress"] > 0.80:
            prob *= 0.55
        return clamp(prob, 0.0, 0.18)

    def move(self):
        if self.hide_hold_remaining > 0 and self.current_behavior == "hide":
            self.hide_hold_remaining -= 1
            self.trajectory.append((self.x, self.y))
            self._add_heatmap(int(self.y), int(self.x), CAT_CONFIG["heatmap_weight"])
            return
        if self.goal_x is None: self.choose_new_goal(); return
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        self.steps_since_goal_change += 1
        if dist < 3.0 or self.steps_since_goal_change > self._goal_change_limit():
            self.update_state_at_goal()
            self.choose_new_goal()
            self.is_running = False
            return
        if not self.is_running and random.random() < self._run_probability():
            self.is_running = True
            self.run_steps_remaining = random.randint(10, 30)
            self.current_behavior = "run"
        if self.is_running:
            self.run_steps_remaining -= 1
            if self.run_steps_remaining <= 0: self.is_running = False

        ta = np.arctan2(dy, dx)
        ad = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
        mt = np.radians(CAT_CONFIG["max_turn_angle"])
        self.angle += np.sign(ad) * min(abs(ad), mt) if abs(ad) > mt else ad

        step_px = self._step_length_px()
        nx = np.clip(self.x + np.cos(self.angle)*step_px, 1, self.w-2)
        ny = np.clip(self.y + np.sin(self.angle)*step_px, 1, self.h-2)

        if self.passable_map[int(ny), int(nx)]:
            self.x, self.y = nx, ny
        else:
            self.choose_new_goal(); self.is_running = False
        self.trajectory.append((self.x, self.y))
        w = CAT_CONFIG["run_heatmap_weight"] if self.is_running else CAT_CONFIG["heatmap_weight"]
        self._add_heatmap(int(self.y), int(self.x), w)

    def update_state_at_goal(self):
        cz = self.zone_map[int(self.y), int(self.x)]
        if cz == "cat_rest":
            self.energy = self.energy + CAT_CONFIG["rest_energy_recover"]
            self.satisfaction = self.satisfaction + 0.02
            self.state["security"] = clamp(self.state["security"] + 0.05)
            self.state["stress"] = clamp(self.state["stress"] - 0.05)
        elif cz == "cat_feeding":
            self.energy = self.energy + CAT_CONFIG["feed_energy_recover"]
            self.satisfaction = self.satisfaction + 0.03
            self.state["hunger"] = clamp(self.state["hunger"] - 0.18)
        elif cz == "window":
            self.satisfaction = self.satisfaction + 0.02
            self.state["boredom"] = clamp(self.state["boredom"] - 0.06)
        elif cz == "shared":
            self.satisfaction = self.satisfaction + 0.01
            self.state["social_need"] = clamp(self.state["social_need"] - 0.04)

        if self.current_behavior == "hide":
            self.state["security"] = clamp(self.state["security"] + 0.06)
            self.state["stress"] = clamp(self.state["stress"] - 0.08)
            self.satisfaction = self.satisfaction + 0.02
            if cz == "cat_rest":
                self.hide_hold_remaining = max(self.hide_hold_remaining, random.randint(8, 20))
        elif self.current_behavior in {"explore", "run", "wander"}:
            self.state["boredom"] = clamp(self.state["boredom"] - 0.04)
        elif self.current_behavior == "approach_human":
            self.state["social_need"] = clamp(self.state["social_need"] - 0.08)

    def get_behavior(self):
        if self.is_running:
            return CAT_BEHAVIOR_LABELS["run"]
        return CAT_BEHAVIOR_LABELS.get(self.current_behavior, "游走")

    def get_zone(self):
        return str(self.zone_map[int(self.y), int(self.x)])

    def _update_dynamic_state(self):
        p = self.personality
        disease_mod = self._disease_modifiers()
        cz = self.get_zone()

        self.energy = self.energy - CAT_CONFIG["energy_consume_per_tick"] * (1.8 if self.is_running else 1.0)
        self.state["hunger"] = clamp(self.state["hunger"] + 0.00075)
        boredom_delta = 0.00035
        if self.current_behavior == "rest":
            boredom_delta += 0.00025
        elif self.current_behavior == "hide":
            boredom_delta += 0.00015
        self.state["boredom"] = clamp(self.state["boredom"] + boredom_delta)
        self.state["social_need"] = clamp(self.state["social_need"] + 0.00025 * p["agreeableness"])

        stress_delta = 0.00020 * p["neuroticism"]
        if cz in {"shared", "human_sleep", "human_work"}:
            stress_delta += 0.00022 * p["neuroticism"]
        if cz in {"cat_rest", "window"}:
            stress_delta -= 0.00030
        if self.current_behavior == "hide":
            stress_delta -= 0.00020
        self.state["stress"] = clamp(self.state["stress"] + stress_delta)

        security_delta = -0.00015 * p["neuroticism"]
        if cz == "cat_rest":
            security_delta += 0.00018
        elif cz in {"shared", "human_work"}:
            security_delta -= 0.00018
        if self.current_behavior == "hide":
            security_delta += 0.00060
        self.state["security"] = clamp(self.state["security"] + security_delta)

        sat_delta = (self.state["security"] - self.state["stress"]) * 0.00025 * disease_mod["satisfaction"]
        sat_delta -= self.state["hunger"] * 0.00010
        self.satisfaction = self.satisfaction + sat_delta

        if self.get_behavior() == CAT_BEHAVIOR_LABELS["rest"]:
            self.rest_drive = clamp(self.rest_drive - 0.0009, 0.15, 0.95)
        else:
            self.rest_drive = clamp(self.rest_drive + 0.0012, 0.15, 0.95)

    def record_tick_stats(self):
        zone = self.get_zone()
        behavior = self.get_behavior()
        behavior_group = summarize_cat_behavior(behavior)
        self.zone_stay_ticks[zone] = self.zone_stay_ticks.get(zone, 0) + 1
        self.behavior_counts[behavior] = self.behavior_counts.get(behavior, 0) + 1
        self.behavior_group_counts[behavior_group] = self.behavior_group_counts.get(behavior_group, 0) + 1
        self.behavior_durations[behavior] = self.behavior_durations.get(behavior, 0) + 1
        self.behavior_group_durations[behavior_group] = self.behavior_group_durations.get(behavior_group, 0) + 1

    def get_behavior_summary(self):
        return {
            "profile_name": self.cat_profile["name"],
            "total_ticks": int(sum(self.zone_stay_ticks.values())),
            "zone_stay_ticks": dict(sorted(self.zone_stay_ticks.items())),
            "behavior_counts": dict(sorted(self.behavior_counts.items())),
            "behavior_group_counts": dict(sorted(self.behavior_group_counts.items())),
            "behavior_durations": dict(sorted(self.behavior_durations.items())),
            "behavior_group_durations": dict(sorted(self.behavior_group_durations.items())),
            "behavior_group_mapping": {
                key: sorted(value) for key, value in sorted(CAT_BEHAVIOR_GROUP_LABELS.items())
            },
            "dynamic_behavior_set": sorted(CAT_BEHAVIOR_GROUP_LABELS["玩耍"]),
            "final_state": {key: round(float(value), 4) for key, value in self.state.items()},
        }

    def step(self):
        self.move()
        self._update_dynamic_state()
        self.record_tick_stats()


# ===================== 人类智能体 =====================
HUMAN_BEHAVIOR_LABELS = {
    "sleep": "睡眠",
    "home_work": "居家工作",
    "outside": "外出",
    "household": "家务",
    "care": "照护",
    "leisure": "休闲",
    "education": "学习",
    "move": "移动",
    "wander": "闲逛",
}


class HumanAgent:
    def __init__(self, start_x, start_y, zone_map, passable_map, scale_x, scale_y, total_ticks=1440, human_profile=None):
        self.zone_map, self.passable_map = zone_map, passable_map
        self.scale_x, self.scale_y = scale_x, scale_y
        self.h, self.w = zone_map.shape
        self.total_ticks, self.current_tick = total_ticks, 0
        self.x, self.y = float(start_x), float(start_y)
        self.satisfaction = HUMAN_CONFIG["initial_satisfaction"]
        self.angle = random.uniform(0, 2*np.pi)
        self.trajectory = [(self.x, self.y)]
        self.visit_count = np.zeros((self.h, self.w), dtype=float)
        self._add_heatmap(int(self.y), int(self.x), HUMAN_CONFIG["heatmap_weight"]*2)
        self.path_finder = PathFinder(passable_map)
        self.path, self.path_index = [], 0
        self.state = "moving"
        self.human_profile = human_profile or TimeUseParameterBuilder(total_ticks=total_ticks).build_profile("default_china", country="CN")
        self.activity_schedule = self.human_profile["activity_schedule"]
        self.current_activity_index = -1
        self.current_activity = None
        self.current_activity_remaining = 0
        self.current_zone = str(self.zone_map[int(self.y), int(self.x)])
        self.target_zone = self.current_zone
        self.wander_steps = 0
        self.wander_target_x = self.wander_target_y = None
        self.zone_stay_ticks = {}
        self.activity_ticks = {}
        self.behavior_counts = {}
        self.outside_ticks = 0
        self._advance_activity_if_needed(force=True)

    def _add_heatmap(self, y, x, weight):
        if 0 <= y < self.h and 0 <= x < self.w: self.visit_count[y, x] += weight
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    self.visit_count[ny, nx] += weight * (0.4 / (1 + np.sqrt(dy*dy+dx*dx)))

    def _get_current_activity(self):
        return self.current_activity or "leisure"

    def _advance_activity_if_needed(self, force=False):
        if not force and self.current_activity_remaining > 0:
            return
        if not self.activity_schedule:
            self.current_activity = "leisure"
            self.current_activity_remaining = self.total_ticks
        else:
            self.current_activity_index = (self.current_activity_index + 1) % len(self.activity_schedule)
            item = self.activity_schedule[self.current_activity_index]
            self.current_activity = item["activity"]
            self.current_activity_remaining = int(item["duration"])
        self._start_activity(self.current_activity)

    def _start_activity(self, activity):
        if activity == "outside":
            self._start_outside()
            return
        self.state = "moving"
        self.target_zone = self._select_zone_for_activity(activity)
        self.choose_new_goal(self.target_zone)

    def _select_zone_for_activity(self, activity):
        mapping = self.human_profile.get("assumptions", {}).get("activity_to_zone_weights")
        if mapping is None:
            # Profile JSON keeps only the mapping version; import-time builder metadata is the authoritative export.
            from time_use_parameter_builder import ACTIVITY_TO_ZONE_MAP
            mapping = ACTIVITY_TO_ZONE_MAP
        zone_weights = mapping.get(activity, {"shared": 1.0})
        return weighted_random_choice(zone_weights)

    def _start_outside(self):
        self.state = "outside"
        self.current_zone = "outside"
        self.target_zone = "outside"
        self.path = []
        self.path_index = 0

    def choose_new_goal(self, target_zone=None):
        tz = target_zone or self.target_zone
        if tz == "outside":
            self._start_outside()
            return
        zc = np.where(self.zone_map == tz)
        if len(zc[0]) > 5:
            cy, cx = np.mean(zc[0]), np.mean(zc[1])
            best_idx, best_dist = None, float('inf')
            for _ in range(min(50, len(zc[0]))):
                idx = random.randint(0, len(zc[0])-1)
                py, px = zc[0][idx], zc[1][idx]
                d = (py-cy)**2 + (px-cx)**2
                if d < best_dist: best_dist, best_idx = d, idx
            idx = best_idx if best_idx is not None else random.randint(0, len(zc[0])-1)
            gx, gy = float(zc[1][idx]), float(zc[0][idx])
            self.path = self.path_finder.find_path(self.x, self.y, gx, gy, use_internal=HUMAN_CONFIG["use_internal_path"])
            self.path_index = 0
            if len(self.path) > 0:
                self.state = "moving"
                return

        coords = np.where(self.passable_map)
        if len(coords[0]) > 0:
            idx = random.randint(0, len(coords[0])-1)
            self.path = self.path_finder.find_path(self.x, self.y, float(coords[1][idx]), float(coords[0][idx]))
            self.path_index = 0
            self.state = "moving"

    def _start_wandering(self):
        if self.state == "outside":
            return
        self.state = "wandering"
        mn, mx = HUMAN_CONFIG["random_walk_steps"]
        self.wander_steps = min(random.randint(mn, mx), max(5, self.current_activity_remaining))
        cz = self.zone_map[int(self.y), int(self.x)]
        zc = np.where(self.zone_map == cz)
        if len(zc[0]) > 0:
            idx = random.randint(0, len(zc[0])-1)
            self.wander_target_y, self.wander_target_x = float(zc[0][idx]), float(zc[1][idx])
        else:
            self.wander_target_x = self.x + random.randint(-30, 30)
            self.wander_target_y = self.y + random.randint(-30, 30)

    def _wander_move(self):
        if self.wander_steps <= 0:
            self.state = "moving"
            self.choose_new_goal(self._select_zone_for_activity(self._get_current_activity()))
            return
        self.wander_steps -= 1
        dx, dy = self.wander_target_x - self.x, self.wander_target_y - self.y
        if np.sqrt(dx*dx+dy*dy) < 5.0:
            cz = self.zone_map[int(self.y), int(self.x)]
            zc = np.where(self.zone_map == cz)
            if len(zc[0]) > 0:
                idx = random.randint(0, len(zc[0])-1)
                self.wander_target_y, self.wander_target_x = float(zc[0][idx]), float(zc[1][idx])
            return
        ta = np.arctan2(dy, dx)
        ad = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
        mt = np.radians(HUMAN_CONFIG["max_turn_angle"])
        self.angle += np.sign(ad) * min(abs(ad), mt) if abs(ad) > mt else ad
        sp = HUMAN_CONFIG["step_length_m"] / self.scale_x
        nx = np.clip(self.x + np.cos(self.angle)*sp, 1, self.w-2)
        ny = np.clip(self.y + np.sin(self.angle)*sp, 1, self.h-2)
        if self.passable_map[int(ny), int(nx)]: self.x, self.y = nx, ny
        else: self._start_wandering()
        self._record_indoor_position()

    def _move_indoor(self):
        if len(self.path) == 0 or self.path_index >= len(self.path):
            self.update_state_at_goal()
            self._start_wandering()
            return
        nx, ny = self.path[self.path_index]
        dx, dy = nx - self.x, ny - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 2.0:
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.update_state_at_goal()
                self._start_wandering()
            return
        ta = np.arctan2(dy, dx)
        ad = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
        mt = np.radians(HUMAN_CONFIG["max_turn_angle"])
        self.angle += np.sign(ad) * min(abs(ad), mt) if abs(ad) > mt else ad
        sp = HUMAN_CONFIG["step_length_m"] / self.scale_x
        if dist >= sp:
            nx = self.x + np.cos(self.angle)*sp
            ny = self.y + np.sin(self.angle)*sp
        nx = np.clip(nx, 1, self.w-2); ny = np.clip(ny, 1, self.h-2)
        if self.passable_map[int(ny), int(nx)]:
            self.x, self.y = nx, ny
        else:
            self.choose_new_goal(self.target_zone)
        self._record_indoor_position()

    def _record_indoor_position(self):
        self.current_zone = str(self.zone_map[int(self.y), int(self.x)])
        self.trajectory.append((self.x, self.y))
        self._add_heatmap(int(self.y), int(self.x), HUMAN_CONFIG["heatmap_weight"])

    def update_state_at_goal(self):
        cz = self.zone_map[int(self.y), int(self.x)]
        if cz == "human_sleep":   self.satisfaction = min(1.0, self.satisfaction + 0.03)
        elif cz == "human_work":  self.satisfaction = min(1.0, self.satisfaction + 0.02)
        elif cz == "shared":      self.satisfaction = min(1.0, self.satisfaction + 0.03)
        elif cz == "window":      self.satisfaction = min(1.0, self.satisfaction + 0.02)

    def _outside_step(self):
        self.current_zone = "outside"

    def move(self):
        self._advance_activity_if_needed()
        if self.state == "outside":
            self._outside_step()
        elif self.state == "wandering":
            self._wander_move()
        else:
            self._move_indoor()

    def get_behavior(self):
        if self.state == "outside":
            return HUMAN_BEHAVIOR_LABELS["outside"]
        return HUMAN_BEHAVIOR_LABELS.get(self._get_current_activity(), HUMAN_BEHAVIOR_LABELS["move"])

    def get_record_x(self):
        return None if self.state == "outside" else self.x

    def get_record_y(self):
        return None if self.state == "outside" else self.y

    def _record_statistics(self):
        zone = self.current_zone if self.state == "outside" else str(self.zone_map[int(self.y), int(self.x)])
        activity = self._get_current_activity()
        behavior = self.get_behavior()
        self.zone_stay_ticks[zone] = self.zone_stay_ticks.get(zone, 0) + 1
        self.activity_ticks[activity] = self.activity_ticks.get(activity, 0) + 1
        self.behavior_counts[behavior] = self.behavior_counts.get(behavior, 0) + 1
        if self.state == "outside":
            self.outside_ticks += 1

    def get_behavior_summary(self):
        return {
            "profile_id": self.human_profile["profile_id"],
            "display_name": self.human_profile["display_name"],
            "total_ticks": int(sum(self.activity_ticks.values())),
            "human_zone_stay_ticks": dict(sorted(self.zone_stay_ticks.items())),
            "human_activity_ticks": dict(sorted(self.activity_ticks.items())),
            "behavior_counts": dict(sorted(self.behavior_counts.items())),
            "outside_ticks": self.outside_ticks,
            "final_state": {
                "human_state": self.state,
                "human_zone": self.current_zone,
                "current_activity": self._get_current_activity(),
                "satisfaction": round(float(self.satisfaction), 4),
            },
            "human_profile_summary": {
                "profile_id": self.human_profile["profile_id"],
                "display_name": self.human_profile["display_name"],
                "source_dataset": self.human_profile["source"]["dataset"],
                "source_tables": self.human_profile["source"]["tables"],
                "occupation_categories": self.human_profile["source"]["occupation_categories"],
                "tick_minutes": self.human_profile["assumptions"]["tick_minutes"],
                "normalization_applied": self.human_profile["assumptions"]["normalization_applied"],
            },
        }

    def step(self):
        self.move()
        self._record_statistics()
        self.current_activity_remaining -= 1
        self.current_tick += 1


# ===================== 模拟主控 =====================
class Simulation:
    def __init__(self, floor_plan_path, total_ticks=5000, cat_profile=None, random_seed=None,
                 output_dir="result", auto_export=False, human_profile_id="default_china",
                 country="CN", day_type="average_day", tick_minutes=1.0,
                 data_dir="data", human_mapping_path="config/human_profile_mapping.json"):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.total_ticks = total_ticks
        self.random_seed = random_seed
        self.output_dir = str(resolve_project_path(output_dir))
        self.auto_export = auto_export
        self.tick_minutes = tick_minutes
        self.human_profile_id = human_profile_id
        self.country = country
        self.day_type = day_type
        self.human_profile_builder = TimeUseParameterBuilder(
            data_dir=data_dir,
            mapping_path=human_mapping_path,
            tick_minutes=tick_minutes,
            total_ticks=total_ticks,
        )
        self.human_profile = self.human_profile_builder.build_profile(
            human_profile_id,
            country=country,
            day_type=day_type,
        )
        floor_plan_path = str(resolve_project_path(floor_plan_path))
        parser = FloorPlanParser(floor_plan_path)
        self.parser = parser
        self.img, self.zone_map, self.passable_maps, self.zone_stats, self.scale_x, self.scale_y = parser.parse()

        cp = np.where(self.passable_maps["cat"])
        if len(cp[0]) == 0: raise ValueError("没有猫可通行的区域")
        idx = random.randint(0, len(cp[0])-1)
        self.cat = CatAgent(cp[1][idx], cp[0][idx], self.zone_map, self.passable_maps["cat"], self.scale_x, self.scale_y, cat_profile)

        hp = np.where(self.passable_maps["human"])
        if len(hp[0]) == 0: raise ValueError("没有人类可通行的区域")
        idx = random.randint(0, len(hp[0])-1)
        self.human = HumanAgent(
            hp[1][idx], hp[0][idx],
            self.zone_map, self.passable_maps["human"],
            self.scale_x, self.scale_y,
            total_ticks,
            human_profile=self.human_profile,
        )

    def run(self):
        print(f"\n[模拟] 开始运行 ({self.total_ticks} 步)...")
        self.tick_records = []
        for tick in range(self.total_ticks):
            if tick % 500 == 0: print(f"       进度: {tick}/{self.total_ticks}")
            self.cat.step()
            self.human.step()
            self.tick_records.append({
                "tick": tick,
                "cat_x": self.cat.x,
                "cat_y": self.cat.y,
                "cat_zone": self.cat.get_zone(),
                "cat_behavior": self.cat.get_behavior(),
                "cat_behavior_group": summarize_cat_behavior(self.cat.get_behavior()),
                "cat_energy": self.cat.state["energy"],
                "cat_stress": self.cat.state["stress"],
                "cat_hunger": self.cat.state["hunger"],
                "cat_boredom": self.cat.state["boredom"],
                "human_x": self.human.get_record_x(),
                "human_y": self.human.get_record_y(),
                "human_zone": self.human.current_zone,
                "human_activity": self.human._get_current_activity(),
                "human_behavior": self.human.get_behavior(),
                "human_state": self.human.state,
                "human_profile_id": self.human.human_profile["profile_id"],
            })
        print("[模拟] 运行完成!")
        if self.auto_export:
            self.export_outputs()

    def export_tick_records_csv(self, csv_path=None):
        if not hasattr(self, "tick_records"):
            raise RuntimeError("尚未运行模拟，请先调用 run()")
        if csv_path is None:
            csv_path = os.path.join(self.output_dir, "tick_records.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fields = [
            "tick", "cat_x", "cat_y", "cat_zone", "cat_behavior", "cat_behavior_group",
            "cat_energy", "cat_stress", "cat_hunger", "cat_boredom",
            "human_x", "human_y", "human_zone", "human_activity",
            "human_behavior", "human_state", "human_profile_id",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.tick_records)
        print(f"[输出] tick records 已保存: {csv_path}")
        return csv_path

    def export_cat_behavior_summary(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "cat_behavior_summary.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        summary = self.cat.get_behavior_summary()
        summary["random_seed"] = self.random_seed
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[输出] 猫行为摘要已保存: {json_path}")
        return json_path

    def export_cat_profile_used(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "cat_profile_used.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.cat.cat_profile, f, ensure_ascii=False, indent=2)
        print(f"[输出] 猫档案已保存: {json_path}")
        return json_path

    def export_human_behavior_summary(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "human_behavior_summary.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        summary = self.human.get_behavior_summary()
        summary["random_seed"] = self.random_seed
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[输出] 人类行为摘要已保存: {json_path}")
        return json_path

    def export_human_profile_used(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "human_profile_used.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.human_profile, f, ensure_ascii=False, indent=2)
        print(f"[输出] 人类画像已保存: {json_path}")
        return json_path

    def export_source_metadata(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "source_metadata.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        metadata = self.human_profile_builder.last_source_metadata or {}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[输出] 来源元数据已保存: {json_path}")
        return json_path

    def export_activity_to_zone_mapping_used(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(self.output_dir, "activity_to_zone_mapping_used.json")
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        metadata = self.human_profile_builder.get_activity_to_zone_mapping_metadata()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[输出] 活动-区域映射已保存: {json_path}")
        return json_path

    def export_outputs(self, output_dir=None):
        if output_dir is not None:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        return {
            "tick_records": self.export_tick_records_csv(),
            "cat_behavior_summary": self.export_cat_behavior_summary(),
            "cat_profile_used": self.export_cat_profile_used(),
            "human_behavior_summary": self.export_human_behavior_summary(),
            "human_profile_used": self.export_human_profile_used(),
            "source_metadata": self.export_source_metadata(),
            "activity_to_zone_mapping_used": self.export_activity_to_zone_mapping_used(),
        }

    def visualize(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # 1. 轨迹图
        ax = axes[0, 0]
        ax.imshow(self.img, alpha=0.4)
        ct = np.array(self.cat.trajectory)
        ht = np.array(self.human.trajectory)
        if len(ct) > 1: ax.plot(ct[:,0], ct[:,1], color="#FF6600", lw=1.5, alpha=0.7, label="Cat")
        if len(ht) > 1: ax.plot(ht[:,0], ht[:,1], color="#3366FF", lw=1.5, alpha=0.7, label="Human")
        if len(ct) > 0:
            ax.scatter(ct[0,0], ct[0,1], color="green", s=80, marker="o", zorder=5, ec='white', lw=2)
            ax.scatter(ct[-1,0], ct[-1,1], color="red", s=80, marker="o", zorder=5, ec='white', lw=2)
        if len(ht) > 0:
            ax.scatter(ht[0,0], ht[0,1], color="green", s=80, marker="s", zorder=5, ec='white', lw=2)
            ax.scatter(ht[-1,0], ht[-1,1], color="red", s=80, marker="s", zorder=5, ec='white', lw=2)
        ax.set_title("Trajectory Map", fontsize=16, fontweight='bold')
        ax.legend(loc="upper right", fontsize=12); ax.axis("off")

        # 2. 猫热力图
        ax = axes[0, 1]
        ch = gaussian_filter(self.cat.visit_count, sigma=1.2)
        if ch.max() > 0: ch = ch / ch.max()
        ch = np.power(ch, 0.4)
        im = ax.imshow(ch, cmap=RAINBOW_CMAP, alpha=1.0, vmin=0, vmax=1)
        ax.imshow(self.img, alpha=0.15)
        ax.set_title("Cat Heatmap", fontsize=16, fontweight='bold'); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3. 人热力图
        ax = axes[1, 0]
        hh = gaussian_filter(self.human.visit_count, sigma=1.2)
        if hh.max() > 0: hh = hh / hh.max()
        hh = np.power(hh, 0.4)
        im = ax.imshow(hh, cmap=RAINBOW_CMAP, alpha=1.0, vmin=0, vmax=1)
        ax.imshow(self.img, alpha=0.15)
        ax.set_title("Human Heatmap", fontsize=16, fontweight='bold'); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4. 报告
        ax = axes[1, 1]; ax.axis("off")
        score = self.cat.satisfaction*0.4 + self.human.satisfaction*0.4 + 0.2
        status = "Good layout" if self.cat.satisfaction > 0.5 and self.human.satisfaction > 0.5 else "Needs optimization"
        human_source = self.human_profile["source"]
        human_budget = self.human_profile["derived_activity_budget"]
        report = f"""Simulation Report ({self.total_ticks} Steps)
{'='*50}

Parameters:
- Total Steps: {self.total_ticks}
- Tick Scale: {self.tick_minutes:g} min/tick
- Cat Profile: {self.cat.cat_profile['name']}
- Age Stage: {self.cat.objective['age_stage']}
- Cat Energy: {self.cat.energy:.2f}
- Cat Satisfaction: {self.cat.satisfaction:.2f}
- Cat Stress: {self.cat.state['stress']:.2f}
- Cat Hunger: {self.cat.state['hunger']:.2f}
- Cat Boredom: {self.cat.state['boredom']:.2f}
- Human Satisfaction: {self.human.satisfaction:.2f}
- Human Profile: {self.human_profile['profile_id']}
- Source Dataset: {human_source['dataset']}
- Source Tables: {', '.join(human_source['tables'])}

Human Activity Budget (ticks):
- Sleep: {human_budget.get('sleep', 0)}
- Home Work: {human_budget.get('home_work', 0)}
- Outside: {human_budget.get('outside', 0)}
- Household: {human_budget.get('household', 0)}
- Leisure: {human_budget.get('leisure', 0)}

Note: Activity-to-zone mapping is model-defined.
Overall Score: {score:.2f}

Status:
{status}
"""
        ax.text(0.1, 0.5, report, fontsize=13, verticalalignment="center",
                fontfamily="sans-serif", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
            print(f"[输出] 结果图已保存: {save_path}")
        plt.close()


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("="*60)
    print(" Cat-Human Co-living Simulation Engine v9.0")
    print(" Standalone Edition - Zero Dependencies")
    print("="*60)

    result_dir = ensure_project_dir("result")

    # 步骤1：自动生成标准户型图（与你的原始户型完全一致）
    floor_plan = generate_floor_plan(str(result_dir / "floor_plan.png"))

    # ★ 如果你想替换成自己的户型图，注释掉上面一行，改为：
    # floor_plan = "你的户型图.png"

    # 步骤2：创建模拟（默认1000步，可改为5000步获得更精确结果）
    sim = Simulation(floor_plan, total_ticks=1440, output_dir=str(result_dir))

    # 步骤3：运行模拟
    sim.run()

    # 步骤4：输出结果（四宫格图：轨迹 + 猫热力图 + 人热力图 + 报告）
    sim.visualize(save_path=str(result_dir / "simulation_result.png"))
    sim.export_outputs()

    # 计算节点画像，供策略卡后处理使用。
    from trajectory_analyzer import TrajectoryAnalyzer
    from metrics_calculator import SpaceMetricsCalculator
    from node_detector import NodeDetector
    from strategy_cards_module import draw_strategy_cards
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']

    analyzer = TrajectoryAnalyzer(
        house_width_m=sim.parser.house_width_m,
        house_depth_m=sim.parser.house_depth_m,
        source_width_px=sim.parser.img_width,
        source_height_px=sim.parser.img_height,
    )
    analyzer.load_from_records(sim.tick_records)
    analyzer.export_csv(str(result_dir / "trajectory.csv"))
    metrics = SpaceMetricsCalculator(analyzer).compute_all()
    detector = NodeDetector(metrics, intensity_pct=80, cooc_pct=90, dbscan_eps=2, dbscan_min_samples=3)
    nodes = detector.detect()
    strategy_nodes = build_strategy_card_nodes(nodes, analyzer, sim)
    draw_strategy_cards(
        strategy_nodes,
        output_dir=result_dir / "strategy_cards",
        sim_steps=sim.total_ticks,
    )

    print("\n" + "="*60)
    print(f" ✅ 全部完成！请查看 {project_relative_display(result_dir)} 下的 simulation_result.png、strategy_cards/ 和数据文件")
    print("="*60)
