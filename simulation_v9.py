"""
人宠共居模拟引擎 v9.0 - 完全独立运行版
Standalone Edition: 无需任何外部配置文件
功能：生成户型图 → 运行双智能体模拟 → 输出轨迹图+热力图+报告
作者：基于ABM的人宠共居空间优化研究
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面后端，服务器/笔记本通用
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter
from heapq import heappush, heappop
import random

# ===================== 中文字体支持 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
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
    def __init__(self, floor_plan_path, house_width_m=14.0, house_depth_m=16.5):
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
                tg = g_score[cur] + 1
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

    def _get_neighbors(self, pos):
        y, x = pos
        nbs = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < self.h and 0 <= nx < self.w and self.passable_map[ny, nx]:
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
    def __init__(self, start_x, start_y, zone_map, passable_map, scale_x, scale_y):
        self.zone_map, self.passable_map = zone_map, passable_map
        self.scale_x, self.scale_y = scale_x, scale_y
        self.h, self.w = zone_map.shape
        self.x, self.y = float(start_x), float(start_y)
        self.energy = CAT_CONFIG["initial_energy"]
        self.satisfaction = CAT_CONFIG["initial_satisfaction"]
        self.angle = random.uniform(0, 2*np.pi)
        self.trajectory = [(self.x, self.y)]
        self.visit_count = np.zeros((self.h, self.w), dtype=float)
        self._add_heatmap(int(self.y), int(self.x), CAT_CONFIG["heatmap_weight"]*3)
        self.goal_x = self.goal_y = None
        self.steps_since_goal_change = 0
        self.is_running = False
        self.run_steps_remaining = 0
        self.choose_new_goal()

    def _add_heatmap(self, y, x, weight):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.visit_count[y, x] += weight
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    self.visit_count[ny, nx] += weight * (0.5 / (1 + np.sqrt(dy*dy+dx*dx)))

    def choose_new_goal(self):
        coords = np.where(self.passable_map)
        if len(coords[0]) == 0: return
        cur_zone = self.zone_map[int(self.y), int(self.x)]
        targets = (["cat_rest","cat_feeding","shared","empty"] if self.energy < 0.3 else
                   ["window","shared","empty","cat_rest"] if self.energy < 0.7 else
                   ["empty","shared","window","cat_rest"])
        for tz in targets:
            if tz == cur_zone: continue
            zc = np.where(self.zone_map == tz)
            if len(zc[0]) > 5:
                idx = random.randint(0, len(zc[0])-1)
                self.goal_x, self.goal_y = float(zc[1][idx]), float(zc[0][idx])
                self.steps_since_goal_change = 0
                return
        idx = random.randint(0, len(coords[0])-1)
        self.goal_y, self.goal_x = float(coords[0][idx]), float(coords[1][idx])
        self.steps_since_goal_change = 0

    def move(self):
        if self.goal_x is None: self.choose_new_goal(); return
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        self.steps_since_goal_change += 1
        if dist < 3.0 or self.steps_since_goal_change > 200:
            self.update_state_at_goal()
            self.choose_new_goal()
            self.is_running = False
            return
        if not self.is_running and random.random() < CAT_CONFIG["run_probability"]:
            self.is_running = True
            self.run_steps_remaining = random.randint(10, 30)
        if self.is_running:
            self.run_steps_remaining -= 1
            if self.run_steps_remaining <= 0: self.is_running = False

        ta = np.arctan2(dy, dx)
        ad = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
        mt = np.radians(CAT_CONFIG["max_turn_angle"])
        self.angle += np.sign(ad) * min(abs(ad), mt) if abs(ad) > mt else ad

        step = CAT_CONFIG["step_length_m"] * (CAT_CONFIG["jump_distance_multiplier"] if self.is_running else 1)
        step_px = step / self.scale_x
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
            self.energy = min(1.0, self.energy + CAT_CONFIG["rest_energy_recover"])
            self.satisfaction = min(1.0, self.satisfaction + 0.02)
        elif cz == "cat_feeding":
            self.energy = min(1.0, self.energy + CAT_CONFIG["feed_energy_recover"])
            self.satisfaction = min(1.0, self.satisfaction + 0.03)
        elif cz == "window":  self.satisfaction = min(1.0, self.satisfaction + 0.02)
        elif cz == "shared":  self.satisfaction = min(1.0, self.satisfaction + 0.01)

    def get_behavior(self):
        if self.is_running:
            return "奔跑"
        cz = self.zone_map[int(self.y), int(self.x)]
        if cz == "cat_rest":
            return "休息"
        if cz == "cat_feeding":
            return "进食"
        if cz == "window":
            return "观望"
        return "游走"

    def step(self):
        self.energy = max(0.1, self.energy - CAT_CONFIG["energy_consume_per_tick"])
        self.move()


# ===================== 人类智能体 =====================
class HumanAgent:
    def __init__(self, start_x, start_y, zone_map, passable_map, scale_x, scale_y, total_ticks=5000):
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
        self.wander_steps = 0
        self.wander_target_x = self.wander_target_y = None
        self.choose_new_goal()

    def _add_heatmap(self, y, x, weight):
        if 0 <= y < self.h and 0 <= x < self.w: self.visit_count[y, x] += weight
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    self.visit_count[ny, nx] += weight * (0.4 / (1 + np.sqrt(dy*dy+dx*dx)))

    def _get_activity_pattern(self):
        p = self.current_tick / self.total_ticks
        return (HUMAN_CONFIG["activity_patterns"]["morning"] if p < 0.3 else
                HUMAN_CONFIG["activity_patterns"]["daytime"] if p < 0.7 else
                HUMAN_CONFIG["activity_patterns"]["evening"])

    def choose_new_goal(self):
        pattern = self._get_activity_pattern()
        zones = pattern["zones"].copy()
        random.shuffle(zones)
        for tz in zones:
            zc = np.where(self.zone_map == tz)
            if len(zc[0]) > 5:
                cy, cx = np.mean(zc[0]), np.mean(zc[1])
                best_idx, best_dist = None, float('inf')
                for _ in range(min(50, len(zc[0]))):
                    idx = random.randint(0, len(zc[0])-1)
                    py, px = zc[0][idx], zc[1][idx]
                    d = (py-cy)**2 + (px-cx)**2
                    if d < best_dist: best_dist, best_idx = d, idx
                if best_idx is not None:
                    gx, gy = float(zc[1][best_idx]), float(zc[0][best_idx])
                else:
                    idx = random.randint(0, len(zc[0])-1)
                    gx, gy = float(zc[1][idx]), float(zc[0][idx])
                self.path = self.path_finder.find_path(self.x, self.y, gx, gy, use_internal=HUMAN_CONFIG["use_internal_path"])
                self.path_index = 0
                if len(self.path) > 0: self.state = "moving"; return
        coords = np.where(self.passable_map)
        if len(coords[0]) > 0:
            idx = random.randint(0, len(coords[0])-1)
            self.path = self.path_finder.find_path(self.x, self.y, float(coords[1][idx]), float(coords[0][idx]))
            self.path_index = 0; self.state = "moving"

    def _start_wandering(self):
        self.state = "wandering"
        mn, mx = HUMAN_CONFIG["random_walk_steps"]
        self.wander_steps = random.randint(mn, mx)
        cz = self.zone_map[int(self.y), int(self.x)]
        zc = np.where(self.zone_map == cz)
        if len(zc[0]) > 0:
            idx = random.randint(0, len(zc[0])-1)
            self.wander_target_y, self.wander_target_x = float(zc[0][idx]), float(zc[1][idx])
        else:
            self.wander_target_x = self.x + random.randint(-30, 30)
            self.wander_target_y = self.y + random.randint(-30, 30)

    def _wander_move(self):
        if self.wander_steps <= 0: self.state = "moving"; self.choose_new_goal(); return
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
        self.trajectory.append((self.x, self.y))
        self._add_heatmap(int(self.y), int(self.x), HUMAN_CONFIG["heatmap_weight"])

    def move(self):
        self.current_tick += 1
        if self.state == "wandering": self._wander_move(); return
        if len(self.path) == 0 or self.path_index >= len(self.path):
            self.update_state_at_goal(); self._start_wandering(); return
        nx, ny = self.path[self.path_index]
        dx, dy = nx - self.x, ny - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 2.0:
            self.path_index += 1
            if self.path_index >= len(self.path): self.update_state_at_goal(); self._start_wandering()
            return
        ta = np.arctan2(dy, dx)
        ad = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
        mt = np.radians(HUMAN_CONFIG["max_turn_angle"])
        self.angle += np.sign(ad) * min(abs(ad), mt) if abs(ad) > mt else ad
        sp = HUMAN_CONFIG["step_length_m"] / self.scale_x
        if dist < sp: nx, ny = nx, ny
        else: nx = self.x + np.cos(self.angle)*sp; ny = self.y + np.sin(self.angle)*sp
        nx = np.clip(nx, 1, self.w-2); ny = np.clip(ny, 1, self.h-2)
        if self.passable_map[int(ny), int(nx)]: self.x, self.y = nx, ny
        else: self.choose_new_goal()
        self.trajectory.append((self.x, self.y))
        self._add_heatmap(int(self.y), int(self.x), HUMAN_CONFIG["heatmap_weight"])

    def update_state_at_goal(self):
        cz = self.zone_map[int(self.y), int(self.x)]
        if cz == "human_sleep":   self.satisfaction = min(1.0, self.satisfaction + 0.03)
        elif cz == "human_work":  self.satisfaction = min(1.0, self.satisfaction + 0.02)
        elif cz == "shared":      self.satisfaction = min(1.0, self.satisfaction + 0.03)
        elif cz == "window":      self.satisfaction = min(1.0, self.satisfaction + 0.02)

    def get_behavior(self):
        cz = self.zone_map[int(self.y), int(self.x)]
        if cz == "human_sleep":
            return "睡眠"
        if cz == "human_work":
            return "工作"
        if self.state == "wandering":
            return "闲逛"
        return "移动"

    def step(self): self.move()


# ===================== 模拟主控 =====================
class Simulation:
    def __init__(self, floor_plan_path, total_ticks=5000):
        self.total_ticks = total_ticks
        parser = FloorPlanParser(floor_plan_path)
        self.img, self.zone_map, self.passable_maps, self.zone_stats, self.scale_x, self.scale_y = parser.parse()

        cp = np.where(self.passable_maps["cat"])
        if len(cp[0]) == 0: raise ValueError("没有猫可通行的区域")
        idx = random.randint(0, len(cp[0])-1)
        self.cat = CatAgent(cp[1][idx], cp[0][idx], self.zone_map, self.passable_maps["cat"], self.scale_x, self.scale_y)

        hp = np.where(self.passable_maps["human"])
        if len(hp[0]) == 0: raise ValueError("没有人类可通行的区域")
        idx = random.randint(0, len(hp[0])-1)
        self.human = HumanAgent(hp[1][idx], hp[0][idx], self.zone_map, self.passable_maps["human"], self.scale_x, self.scale_y, total_ticks)

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
                "cat_behavior": self.cat.get_behavior(),
                "human_x": self.human.x,
                "human_y": self.human.y,
                "human_behavior": self.human.get_behavior(),
            })
        print("[模拟] 运行完成!")

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
        phase = ("Morning" if self.total_ticks/5000 < 0.3 else
                 "Daytime" if self.total_ticks/5000 < 0.7 else "Evening")
        score = self.cat.satisfaction*0.4 + self.human.satisfaction*0.4 + 0.2
        status = "Good layout" if self.cat.satisfaction > 0.5 and self.human.satisfaction > 0.5 else "Needs optimization"
        report = f"""Simulation Report ({self.total_ticks} Steps)
{'='*50}

Parameters:
- Total Steps: {self.total_ticks}
- Cat Energy: {self.cat.energy:.2f}
- Cat Satisfaction: {self.cat.satisfaction:.2f}
- Human Satisfaction: {self.human.satisfaction:.2f}

Activity Phase: {phase}
Overall Score: {score:.2f}

Status:
{status}
"""
        ax.text(0.1, 0.5, report, fontsize=13, verticalalignment="center",
                fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
            print(f"[输出] 结果图已保存: {save_path}")
        plt.close()


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("="*60)
    print(" Cat-Human Co-living Simulation Engine v9.0")
    print(" Standalone Edition - Zero Dependencies")
    print("="*60)

    # 步骤1：自动生成标准户型图（与你的原始户型完全一致）
    floor_plan = generate_floor_plan("floor_plan.png")

    # ★ 如果你想替换成自己的户型图，注释掉上面一行，改为：
    # floor_plan = "你的户型图.png"

    # 步骤2：创建模拟（默认1000步，可改为5000步获得更精确结果）
    sim = Simulation(floor_plan, total_ticks=1000)

    # 步骤3：运行模拟
    sim.run()

    # 步骤4：输出结果（四宫格图：轨迹 + 猫热力图 + 人热力图 + 报告）
    sim.visualize(save_path="simulation_result.png")

    print("\n" + "="*60)
    print(" ✅ 全部完成！请查看当前文件夹下的 simulation_result.png")
    print("="*60)
