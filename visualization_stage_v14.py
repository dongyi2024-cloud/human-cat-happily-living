import os
import warnings
from collections import Counter
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/human-cat-mplconfig")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN

from metrics_calculator import SpaceMetricsCalculator
from simulation_v9 import FloorPlanParser, Simulation, generate_floor_plan
from trajectory_analyzer import TrajectoryAnalyzer, summarize_cat_behavior

for font_path in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)

# Matplotlib resolves the installed Noto CJK TTC as JP even though fontconfig
# exposes SC aliases; the glyph coverage is shared across the CJK family.
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")


BEHAVIOR_COLORS = {
    "休息": "#B7B7A4",
    "进食": "#F15BB5",
    "玩耍": "#F2CC8F",
    "抓挠": "#8B4513",
    "观察": "#9B5DE5",
    "观望": "#9B5DE5",
    "躲藏": "#3D405B",
    "亲近": "#4D908E",
    "探索": "#81B29A",
    "奔跑": "#E76F51",
    "游走": "#43AA8B",
    "占位": "#8B4513",
    "睡眠": "#7A8FA6",
    "移动": "#277DA1",
    "闲逛": "#43AA8B",
    "外出": "#6C757D",
}

NODE_COLORS = {
    "冲突节点": "#FF4D4D",
    "共享节点": "#F4A261",
    "猫专属": "#2A9D8F",
    "人专属": "#4A90E2",
    "低利用": "#8D99AE",
}

STRATEGY = {
    "冲突节点": "建议分流人猫动线或错峰使用",
    "共享节点": "建议保持开放并强化共享功能",
    "猫专属": "建议补充爬架、猫道和停驻点",
    "人专属": "建议保持人类功能完整并限制猫干扰",
    "低利用": "建议改作过渡或储物空间",
}

RISK_COLORS = {
    "高风险": "#FF3333",
    "中风险": "#FF6666",
    "低风险": "#FF9999",
}

BEHAVIOR_LABELS = {
    "休息": "休息",
    "进食": "进食",
    "玩耍": "玩耍",
    "抓挠": "抓挠",
    "观察": "观察",
    "观望": "观察",
    "躲藏": "躲藏",
    "亲近": "亲近",
    "睡眠": "睡眠",
    "移动": "移动",
    "闲逛": "闲逛",
    "外出": "外出",
    "探索": "探索",
    "游走": "游走",
    "奔跑": "奔跑",
    "占位": "抓挠",
}

BEHAVIOR_ABBR = {
    "休息": "休",
    "进食": "食",
    "玩耍": "玩",
    "抓挠": "抓",
    "观察": "观",
    "观望": "观",
    "躲藏": "躲",
    "亲近": "亲",
    "睡眠": "眠",
    "移动": "移",
    "闲逛": "逛",
    "外出": "外",
    "探索": "探",
    "游走": "游",
    "奔跑": "跑",
    "占位": "抓",
}

NODE_LABELS = {
    "冲突节点": "冲突节点",
    "共享节点": "共享节点",
    "猫专属": "猫专属",
    "人专属": "人专属",
    "低利用": "低利用",
}

RISK_LABELS = {
    "高风险": "高风险",
    "中风险": "中风险",
    "低风险": "低风险",
    "无风险": "无风险",
}

STRATEGY_LABELS = {
    "冲突节点": "建议分流人猫动线或错峰使用",
    "共享节点": "建议保持开放并强化共享功能",
    "猫专属": "建议补充爬架、猫道和停驻点",
    "人专属": "建议保持人类功能完整并限制猫干扰",
    "低利用": "建议改作过渡或储物空间",
}

BEHAVIOR_INDEX_ORDER = ["奔跑", "玩耍", "探索", "躲藏", "观察", "休息", "进食", "抓挠"]


@dataclass
class VisualNodeProfile:
    node_id: str
    node_type: str
    centroid_y: int
    centroid_x: int
    member_cells: list[tuple[int, int]]
    dominant_behavior: str
    member_bbox: tuple[int, int, int, int]
    cfs: float
    entropy: float
    dcd_avg: float
    dcd_max: float
    risk: str


@dataclass
class VisualizationContext:
    floor_plan_path: str
    trajectory_csv: str
    output_dir: str
    zone_map: np.ndarray
    passable_map: np.ndarray
    analyzer: TrajectoryAnalyzer
    metrics: dict
    dcd_matrix: np.ndarray
    node_profiles: list[VisualNodeProfile]


class GridCanvas:
    def __init__(self, zone_map: np.ndarray, passable_map: np.ndarray, analyzer: TrajectoryAnalyzer):
        self.zone_map = zone_map
        self.passable_map = passable_map
        self.analyzer = analyzer
        self.grid_height = analyzer.grid_height
        self.grid_width = analyzer.grid_width
        self.zone_grid, self.passable_grid = self._aggregate_grid()
        self.base_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        self.wall_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if not self.passable_grid[gy, gx]:
                    self.base_grid[gy, gx] = [35, 35, 38]
                    self.wall_mask[gy, gx] = True
                else:
                    self.base_grid[gy, gx] = [245, 245, 240]

    def _aggregate_grid(self) -> tuple[np.ndarray, np.ndarray]:
        src_h, src_w = self.zone_map.shape
        zone_grid = np.full((self.grid_height, self.grid_width), "empty", dtype=object)
        passable_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for gy in range(self.grid_height):
            y0 = int(np.floor(gy * src_h / self.grid_height))
            y1 = int(np.floor((gy + 1) * src_h / self.grid_height))
            y1 = max(y0 + 1, min(src_h, y1))
            for gx in range(self.grid_width):
                x0 = int(np.floor(gx * src_w / self.grid_width))
                x1 = int(np.floor((gx + 1) * src_w / self.grid_width))
                x1 = max(x0 + 1, min(src_w, x1))
                region_passable = self.passable_map[y0:y1, x0:x1]
                region_zones = self.zone_map[y0:y1, x0:x1]
                pass_ratio = float(np.count_nonzero(region_passable)) / float(region_passable.size)
                passable_grid[gy, gx] = pass_ratio > 0.3
                if passable_grid[gy, gx]:
                    zones = region_zones[region_passable]
                    zone_grid[gy, gx] = Counter(zones).most_common(1)[0][0] if len(zones) else "empty"
                else:
                    zone_grid[gy, gx] = Counter(region_zones.ravel()).most_common(1)[0][0]
        return zone_grid, passable_grid

    def draw_bg(self, ax, show_grid: bool = False) -> None:
        ax.imshow(self.base_grid, extent=[0, self.grid_width, self.grid_height, 0], interpolation="nearest")
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if not self.wall_mask[gy, gx]:
                    continue
                has_passable_neighbor = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width and self.passable_grid[ny, nx]:
                        has_passable_neighbor = True
                        break
                if has_passable_neighbor:
                    rect = plt.Rectangle((gx, gy), 1, 1, fill=False, edgecolor="white", linewidth=0.7, alpha=0.75)
                    ax.add_patch(rect)
        if show_grid:
            for gy in range(self.grid_height + 1):
                ax.axhline(gy, color="#D8D8D8", linewidth=0.25, alpha=0.22, zorder=2)
            for gx in range(self.grid_width + 1):
                ax.axvline(gx, color="#D8D8D8", linewidth=0.25, alpha=0.22, zorder=2)
        ax.set_xlim(-0.2, self.grid_width + 0.2)
        ax.set_ylim(self.grid_height + 0.2, -0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#1A1A1A")

    def draw_cell(self, ax, gy: int, gx: int, color: str, alpha: float = 0.8, edgecolor: str = "#555555", lw: float = 0.35) -> None:
        rect = plt.Rectangle((gx, gy), 1, 1, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=lw)
        ax.add_patch(rect)

    def draw_text(self, ax, gy: int, gx: int, text: str, color: str = "white", size: int = 7) -> None:
        ax.text(gx + 0.5, gy + 0.5, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold", zorder=10)

    def draw_node_rect(self, ax, bbox: tuple[int, int, int, int], color: str, alpha: float = 0.65, linewidth: float = 2.0) -> None:
        gy0, gx0, gy1, gx1 = bbox
        rect = plt.Rectangle((gx0, gy0), gx1 - gx0 + 1, gy1 - gy0 + 1, facecolor=color, alpha=alpha, edgecolor="white", linewidth=linewidth)
        ax.add_patch(rect)

    def draw_node_label(self, ax, profile: VisualNodeProfile, label: str) -> None:
        x0 = profile.centroid_x + 0.5
        y0 = profile.centroid_y + 0.5
        dy = -0.75 if y0 > 1.6 else 0.75
        x1 = min(x0 + 0.75, self.grid_width - 1.0)
        y1 = float(np.clip(y0 + dy, 0.6, self.grid_height - 0.6))
        x = min(x1 + 0.18, self.grid_width - 4.0)
        y = y1
        ax.plot([x0, x1], [y0, y1], color="#111111", linewidth=1.0, alpha=0.9, zorder=5)
        ax.text(
            x,
            y,
            label,
            fontsize=8,
            color="white",
            ha="left",
            va="center",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="white", linewidth=0.5),
        )


def _green_gradient(ratio: float) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    r = int(230 - 196 * ratio)
    g = int(245 - 106 * ratio)
    b = int(230 - 196 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _warm_entropy_color(ratio: float) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    r = int(220 + 35 * ratio)
    g = int(220 - 100 * ratio)
    b = int(220 - 170 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dcd_color(ratio: float) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    r = 255
    g = int(255 * (1.0 - ratio * 0.85))
    b = int(255 * (1.0 - ratio * 0.90))
    return f"#{r:02x}{g:02x}{b:02x}"


def behavior_label(name: str) -> str:
    return BEHAVIOR_LABELS.get(name, str(name))


def behavior_abbr(name: str) -> str:
    return BEHAVIOR_ABBR.get(name, behavior_label(name)[:1].upper())


def node_label(name: str) -> str:
    return NODE_LABELS.get(name, str(name))


def risk_label(name: str) -> str:
    return RISK_LABELS.get(name, str(name))


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, "建议继续观察")


def set_split_title(ax, main: str, subtitle: str | None = None, size: int = 18) -> None:
    title = main if not subtitle else f"{main}\n{subtitle}"
    ax.set_title(title, fontsize=size, fontweight="bold", color="white", pad=18, loc="center")


def add_color_index(ax, title: str, entries: list[tuple[str, str]], width: float = 0.18, height: float | None = None) -> None:
    height = height or min(max(0.15, 0.025 * len(entries) + 0.11), 0.34) * 0.5
    index_ax = ax.inset_axes([0.805, 0.985 - height, width, height], transform=ax.transAxes)
    index_ax.set_facecolor((0, 0, 0, 0.58))
    for spine in index_ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.6)
    index_ax.set_xlim(0, 1)
    index_ax.set_ylim(0, 1)
    index_ax.set_xticks([])
    index_ax.set_yticks([])
    index_ax.text(0.5, 0.88, title, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    if not entries:
        return
    row_ys = np.linspace(0.68, 0.18, len(entries)) if len(entries) > 1 else [0.46]
    for y, (color, label) in zip(row_ys, entries):
        index_ax.add_patch(plt.Rectangle((0.12, y - 0.022), 0.13, 0.044, facecolor=color, edgecolor="white", linewidth=0.55))
        index_ax.text(0.32, y, label, ha="left", va="center", color="white", fontsize=11)


def add_behavior_index(ax) -> None:
    add_color_index(
        ax,
        "行为类型",
        [(BEHAVIOR_COLORS.get(name, "#A9A9A9"), behavior_label(name)) for name in BEHAVIOR_INDEX_ORDER],
        width=0.18,
    )


def add_cfs_index(ax) -> None:
    add_color_index(
        ax,
        "CFS空间强度",
        [
            (_green_gradient(0.20), "低(0-30)"),
            (_green_gradient(0.55), "中(30-60)"),
            (_green_gradient(0.90), "高(>60)"),
        ],
        width=0.18,
    )


def add_entropy_index(ax) -> None:
    index_ax = ax.inset_axes([0.805, 0.9075, 0.18, 0.0775], transform=ax.transAxes)
    index_ax.set_facecolor((0, 0, 0, 0.58))
    for spine in index_ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.6)
    index_ax.set_xlim(0, 1)
    index_ax.set_ylim(0, 1)
    index_ax.set_xticks([])
    index_ax.set_yticks([])
    index_ax.text(0.5, 0.86, "功能复合", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    index_ax.add_patch(plt.Rectangle((0.12, 0.55), 0.13, 0.055, facecolor="#DCDCDC", edgecolor="#777777", linewidth=0.55))
    index_ax.text(0.32, 0.58, "单一功能", ha="left", va="center", color="white", fontsize=11)
    index_ax.add_patch(plt.Rectangle((0.12, 0.34), 0.13, 0.055, facecolor="#F0A25B", edgecolor="#FF3333", linewidth=1.2))
    index_ax.text(0.32, 0.37, "复合功能", ha="left", va="center", color="white", fontsize=11)


def add_dcd_index(ax) -> None:
    add_color_index(
        ax,
        "DCD风险",
        [
            (_dcd_color(0.20), "低风险(0-47.6)"),
            (_dcd_color(0.55), "中风险(47.6-95.1)"),
            (_dcd_color(0.90), "高风险(>95.1)"),
        ],
        width=0.18,
    )


def add_risk_index(ax) -> None:
    add_color_index(
        ax,
        "风险等级",
        [
            (RISK_COLORS["低风险"], "低风险"),
            (RISK_COLORS["中风险"], "中风险"),
            (RISK_COLORS["高风险"], "高风险"),
        ],
        width=0.18,
    )


def add_vertical_scale(ax, title: str, colors: list[str], labels: list[str], width: float = 0.10) -> None:
    height = 0.44
    index_ax = ax.inset_axes([1.015, 0.5 - height / 2, width, height], transform=ax.transAxes)
    index_ax.set_facecolor((0, 0, 0, 0.58))
    for spine in index_ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.6)
    index_ax.set_xlim(0, 1)
    index_ax.set_ylim(0, 1)
    index_ax.set_xticks([])
    index_ax.set_yticks([])
    index_ax.text(0.18, 0.50, title, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold", rotation=90)

    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    cmap = LinearSegmentedColormap.from_list(f"{title}_scale", colors)
    index_ax.imshow(gradient, cmap=cmap, extent=[0.40, 0.58, 0.13, 0.87], aspect="auto")
    index_ax.add_patch(plt.Rectangle((0.40, 0.13), 0.18, 0.74, fill=False, edgecolor="white", linewidth=0.6))
    for idx, label in enumerate(labels):
        y = 0.87 - idx * (0.74 / (len(labels) - 1))
        index_ax.plot([0.58, 0.66], [y, y], color="white", linewidth=0.55)
        index_ax.text(0.70, y, label, ha="left", va="center", color="white", fontsize=8.5)


def _default_paths(output_dir: str) -> tuple[str, str]:
    return os.path.join(output_dir, "floor_plan.png"), os.path.join(output_dir, "trajectory.csv")


def ensure_analysis_inputs(output_dir: str, floor_plan_path: str | None = None, trajectory_csv: str | None = None, total_ticks: int = 1000, random_seed: int | None = 7) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    default_floor_plan, default_trajectory = _default_paths(output_dir)
    floor_plan_path = floor_plan_path or default_floor_plan
    trajectory_csv = trajectory_csv or default_trajectory
    if os.path.exists(floor_plan_path) and os.path.exists(trajectory_csv):
        return floor_plan_path, trajectory_csv

    if not os.path.exists(floor_plan_path):
        generate_floor_plan(floor_plan_path)

    sim = Simulation(floor_plan_path, total_ticks=total_ticks, output_dir=output_dir, random_seed=random_seed)
    sim.run()
    sim.export_tick_records_csv(trajectory_csv)
    return floor_plan_path, trajectory_csv


def compute_dcd_matrix(analyzer: TrajectoryAnalyzer) -> np.ndarray:
    if analyzer.df is None:
        raise RuntimeError("TrajectoryAnalyzer 尚未加载轨迹数据")

    dcd = np.zeros(analyzer.grid_shape, dtype=np.float32)
    for row in analyzer.df.itertuples(index=False):
        human_state = getattr(row, "human_state", "")
        if human_state == "outside" or np.isnan(row.human_x) or np.isnan(row.human_y):
            continue
        if human_state not in {"moving", "wandering"}:
            continue

        cat_behavior_group = getattr(row, "cat_behavior_group", None)
        if cat_behavior_group is None or str(cat_behavior_group).strip() == "":
            cat_behavior_group = summarize_cat_behavior(row.cat_behavior)
        if cat_behavior_group != "玩耍":
            continue

        cat_cell = analyzer._to_grid(row.cat_x, row.cat_y)
        human_cell = analyzer._to_grid(row.human_x, row.human_y)
        dist_m = analyzer._grid_distance_m(cat_cell, human_cell)
        if dist_m > analyzer.proximity_m:
            continue

        cat_vx = getattr(row, "cat_vx_m_per_tick", np.nan)
        cat_vy = getattr(row, "cat_vy_m_per_tick", np.nan)
        human_vx = getattr(row, "human_vx_m_per_tick", np.nan)
        human_vy = getattr(row, "human_vy_m_per_tick", np.nan)
        if any(np.isnan(v) for v in [cat_vx, cat_vy, human_vx, human_vy]):
            continue

        cat_speed = float(np.hypot(cat_vx, cat_vy))
        human_speed = float(np.hypot(human_vx, human_vy))
        if cat_speed <= 1e-8 or human_speed <= 1e-8:
            continue

        cos_angle = (cat_vx * human_vx + cat_vy * human_vy) / (cat_speed * human_speed)
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg >= 90.0:
            continue

        distance_weight = max(0.1, 1.0 - dist_m / max(analyzer.proximity_m, 1e-8))
        speed_weight = max(1.0, (cat_speed + human_speed) * 0.5)
        weight = distance_weight * speed_weight
        dcd[cat_cell] += weight
        if human_cell != cat_cell:
            dcd[human_cell] += weight * 0.5

    return dcd


def _find_visual_peaks(matrix: np.ndarray, passable: np.ndarray, percentile: float) -> list[dict]:
    valid = passable & (matrix > 0)
    if not np.any(valid):
        return []
    threshold = float(np.percentile(matrix[valid], percentile))
    peaks: list[dict] = []
    height, width = matrix.shape
    for gy in range(height):
        for gx in range(width):
            if not valid[gy, gx] or float(matrix[gy, gx]) < threshold:
                continue
            is_peak = True
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < height and 0 <= nx < width and valid[ny, nx] and float(matrix[ny, nx]) >= float(matrix[gy, gx]):
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                peaks.append({"gy": gy, "gx": gx, "value": float(matrix[gy, gx])})
    return peaks


def _cluster_visual_peaks(peaks: list[dict], eps: float = 2.0, min_samples: int = 1) -> list[dict]:
    if len(peaks) < min_samples:
        return []
    points = np.array([[peak["gy"], peak["gx"]] for peak in peaks], dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    clusters: list[dict] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = [peaks[idx] for idx in range(len(peaks)) if labels[idx] == label]
        if len(members) < min_samples:
            continue
        ys = [member["gy"] for member in members]
        xs = [member["gx"] for member in members]
        clusters.append(
            {
                "cluster_id": len(clusters),
                "centroid_y": int(round(float(np.mean(ys)))),
                "centroid_x": int(round(float(np.mean(xs)))),
                "bbox": (min(ys), min(xs), max(ys), max(xs)),
                "member_cells": [(member["gy"], member["gx"]) for member in members],
            }
        )
    return clusters


def _weighted_behavior_for_cells(cells: list[tuple[int, int]], analyzer: TrajectoryAnalyzer, cat_intensity: np.ndarray) -> tuple[str, float]:
    counter: Counter[str] = Counter()
    for gy, gx in cells:
        weight = max(1, int(round(float(cat_intensity[gy, gx]))))
        for behavior, count in analyzer.cat_behavior_grid.get((gy, gx), {}).items():
            counter[behavior] += int(count) * weight
    dominant_behavior = counter.most_common(1)[0][0] if counter else "休息"
    total = sum(counter.values())
    entropy = 0.0
    if total > 0:
        entropy = float(-sum((count / total) * np.log2(count / total) for count in counter.values() if count > 0))
    return dominant_behavior, entropy


def _profile_cells_from_bbox(bbox: tuple[int, int, int, int], passable: np.ndarray, fallback: list[tuple[int, int]]) -> list[tuple[int, int]]:
    gy0, gx0, gy1, gx1 = bbox
    cells = [(gy, gx) for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1) if passable[gy, gx]]
    return cells or fallback


def _risk_for_value(value: float, nonzero: np.ndarray) -> str:
    if value <= 0:
        return "无风险"
    if len(nonzero) >= 5:
        p80 = float(np.percentile(nonzero, 80))
        p60 = float(np.percentile(nonzero, 60))
    elif len(nonzero):
        p80 = float(nonzero.max() * 0.8)
        p60 = float(nonzero.max() * 0.5)
    else:
        return "无风险"
    if value >= p80:
        return "高风险"
    if value >= p60:
        return "中风险"
    return "低风险"


def _build_node_profiles(analyzer: TrajectoryAnalyzer, metrics: dict, dcd_matrix: np.ndarray, canvas: GridCanvas) -> list[VisualNodeProfile]:
    cat_intensity = metrics["cat_intensity"]
    cat_entropy = metrics["cat_entropy"]
    passable = canvas.passable_grid
    cat_peaks = _find_visual_peaks(cat_intensity, passable, 55)
    conflict_peaks = _find_visual_peaks(dcd_matrix, passable, 25)
    cat_clusters = _cluster_visual_peaks(cat_peaks, eps=2.0, min_samples=1)
    conflict_clusters = _cluster_visual_peaks(conflict_peaks, eps=2.0, min_samples=1)
    nonzero_dcd = dcd_matrix[dcd_matrix > 0]

    profiles: list[VisualNodeProfile] = []
    for cluster in cat_clusters:
        cells = _profile_cells_from_bbox(cluster["bbox"], passable, cluster["member_cells"])
        dominant_behavior, entropy = _weighted_behavior_for_cells(cells, analyzer, cat_intensity)
        cfs = float(np.mean([cat_intensity[gy, gx] for gy, gx in cells])) if cells else 0.0
        if entropy == 0.0 and cells:
            entropy = float(np.mean([cat_entropy[gy, gx] for gy, gx in cells]))
        dcd_values = [float(dcd_matrix[gy, gx]) for gy, gx in cells]
        dcd_avg = float(np.mean(dcd_values)) if dcd_values else 0.0
        dcd_max = float(np.max(dcd_values)) if dcd_values else 0.0
        profiles.append(
            VisualNodeProfile(
                node_id=f"H{cluster['cluster_id']}",
                node_type="猫节点",
                centroid_y=cluster["centroid_y"],
                centroid_x=cluster["centroid_x"],
                member_cells=cluster["member_cells"],
                dominant_behavior=dominant_behavior,
                member_bbox=cluster["bbox"],
                cfs=cfs,
                entropy=entropy,
                dcd_avg=dcd_avg,
                dcd_max=dcd_max,
                risk=_risk_for_value(dcd_max, nonzero_dcd),
            )
        )

    for cluster in conflict_clusters:
        cells = _profile_cells_from_bbox(cluster["bbox"], passable, cluster["member_cells"])
        dcd_values = [float(dcd_matrix[gy, gx]) for gy, gx in cells]
        dcd_avg = float(np.mean(dcd_values)) if dcd_values else 0.0
        dcd_max = float(np.max(dcd_values)) if dcd_values else 0.0
        profiles.append(
            VisualNodeProfile(
                node_id=f"S{cluster['cluster_id']}",
                node_type="共现节点",
                centroid_y=cluster["centroid_y"],
                centroid_x=cluster["centroid_x"],
                member_cells=cluster["member_cells"],
                dominant_behavior="动线交叉",
                member_bbox=cluster["bbox"],
                cfs=0.0,
                entropy=0.0,
                dcd_avg=dcd_avg,
                dcd_max=dcd_max,
                risk=_risk_for_value(dcd_max, nonzero_dcd),
            )
        )
    return profiles


def build_context(output_dir: str = "result", floor_plan_path: str | None = None, trajectory_csv: str | None = None, total_ticks: int = 1000, random_seed: int | None = 7) -> VisualizationContext:
    floor_plan_path, trajectory_csv = ensure_analysis_inputs(output_dir, floor_plan_path, trajectory_csv, total_ticks, random_seed)
    parser = FloorPlanParser(floor_plan_path)
    _, zone_map, passable_maps, _, _, _ = parser.parse()
    analyzer = TrajectoryAnalyzer()
    analyzer.load_from_csv(trajectory_csv)
    calculator = SpaceMetricsCalculator(analyzer)
    metrics = calculator.compute_all()
    dcd_matrix = compute_dcd_matrix(analyzer)
    canvas = GridCanvas(zone_map, passable_maps["cat"], analyzer)
    node_profiles = _build_node_profiles(analyzer, metrics, dcd_matrix, canvas)
    return VisualizationContext(
        floor_plan_path=floor_plan_path,
        trajectory_csv=trajectory_csv,
        output_dir=output_dir,
        zone_map=zone_map,
        passable_map=passable_maps["cat"],
        analyzer=analyzer,
        metrics=metrics,
        dcd_matrix=dcd_matrix,
        node_profiles=node_profiles,
    )


def render_grid_stage(ctx: VisualizationContext, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    dominant_behavior = SpaceMetricsCalculator(ctx.analyzer).get_dominant_behavior_matrix("cat")
    outputs: list[str] = []

    cat_intensity = ctx.metrics["cat_intensity"]
    cat_entropy = ctx.metrics["cat_entropy"]
    passable = canvas.passable_grid
    vmax = float(cat_intensity[passable].max()) if np.any(passable) else 1.0
    vmax = max(vmax, 1.0)

    path = os.path.join(output_dir, "grid_cfs_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=True)
    for gy, gx in zip(*np.where(passable)):
        value = float(cat_intensity[gy, gx])
        if value <= 0:
            continue
        ratio = value / vmax
        canvas.draw_cell(ax, gy, gx, _green_gradient(ratio), alpha=0.88, edgecolor="#555555", lw=0.35)
        if value > vmax * 0.35:
            canvas.draw_text(ax, gy, gx, f"{value:.0f}", color="white" if ratio > 0.55 else "#2F4F2F", size=7)
    set_split_title(ax, "栅格级 CFS空间强度图", "离散着色·绿色深浅")
    add_cfs_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "grid_behavior_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=True)
    threshold = float(np.percentile(cat_intensity[passable], 75)) if np.any(passable) else 0.0
    present_behaviors = set()
    for gy, gx in zip(*np.where(passable)):
        if float(cat_intensity[gy, gx]) < threshold:
            continue
        behavior = dominant_behavior[gy, gx] or "休息"
        present_behaviors.add(behavior)
        canvas.draw_cell(ax, gy, gx, BEHAVIOR_COLORS.get(behavior, "#A9A9A9"), alpha=0.82, edgecolor="white", lw=0.45)
        canvas.draw_text(ax, gy, gx, behavior_abbr(behavior), size=8)
    set_split_title(ax, "栅格级主导行为类型图", "仅高频栅格P75")
    add_behavior_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "grid_entropy_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=True)
    entropy_values = [float(cat_entropy[gy, gx]) for gy, gx in zip(*np.where(passable)) if float(cat_intensity[gy, gx]) >= threshold]
    emax = max(entropy_values) if entropy_values else 1.0
    emax = max(emax, 1.0)
    for gy, gx in zip(*np.where(passable)):
        if float(cat_intensity[gy, gx]) < threshold:
            continue
        ent = float(cat_entropy[gy, gx])
        ratio = ent / emax
        canvas.draw_cell(ax, gy, gx, _warm_entropy_color(ratio), alpha=0.68, edgecolor="#888888", lw=0.3)
        rect = plt.Rectangle((gx, gy), 1, 1, fill=False, edgecolor=_dcd_color(ratio), linewidth=0.4 + ratio * 2.5)
        ax.add_patch(rect)
        canvas.draw_text(ax, gy, gx, f"{ent:.1f}", color="white" if ratio > 0.5 else "#333333", size=7)
    set_split_title(ax, "栅格级行为熵图", "功能复合度·边框粗细映射")
    add_entropy_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "grid_dcd_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=True)
    dcd = ctx.dcd_matrix
    dmax = float(dcd[passable].max()) if np.any(passable) else 0.0
    if dmax > 0:
        for gy, gx in zip(*np.where(passable)):
            value = float(dcd[gy, gx])
            ratio = value / dmax if dmax > 0 else 0.0
            if ratio < 0.015:
                continue
            canvas.draw_cell(ax, gy, gx, _dcd_color(ratio), alpha=0.72, edgecolor="#555555", lw=0.35)
            if value > dmax * 0.12:
                canvas.draw_text(ax, gy, gx, f"{value:.1f}", color="white" if ratio > 0.5 else "#4A0000", size=7)
        set_split_title(ax, "栅格级DCD安全图", "动态冲突密度·红色深浅")
    else:
        set_split_title(ax, "栅格级DCD安全图", "动态冲突密度·红色深浅")
    add_dcd_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    return outputs


def render_node_stage(ctx: VisualizationContext, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    outputs: list[str] = []

    profiles = sorted(ctx.node_profiles, key=lambda item: item.cfs, reverse=True)
    cat_profiles = [profile for profile in profiles if profile.node_type == "猫节点"]
    conflict_profiles = sorted([profile for profile in profiles if profile.node_type == "共现节点"], key=lambda item: item.dcd_max, reverse=True)
    max_cat = max([profile.cfs for profile in cat_profiles] + [1.0])
    max_entropy = max([profile.entropy for profile in cat_profiles] + [1.0])

    path = os.path.join(output_dir, "node_importance_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=False)
    for idx, profile in enumerate(cat_profiles, start=1):
        ratio = profile.cfs / max_cat
        color = plt.cm.YlGn(0.3 + ratio * 0.7)
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=0.75, linewidth=2.3)
        canvas.draw_node_label(ax, profile, f"{profile.node_id}")
    set_split_title(ax, "节点级空间重要性图")
    add_vertical_scale(ax, "空间重要性", ["#E6F5E6", "#228B22"], ["100", "80", "60", "40", "20", "0"])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "node_function_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=False)
    present = set()
    for idx, profile in enumerate(cat_profiles, start=1):
        present.add(profile.dominant_behavior)
        color = BEHAVIOR_COLORS.get(profile.dominant_behavior, "#A9A9A9")
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=0.72, linewidth=2.0)
        canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{behavior_label(profile.dominant_behavior)}")
    set_split_title(ax, "节点级 功能定位图")
    add_behavior_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "node_entropy_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=False)
    for idx, profile in enumerate(cat_profiles, start=1):
        ratio = profile.entropy / max_entropy
        color = plt.cm.YlOrRd(0.2 + ratio * 0.7)
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=0.72, linewidth=2.0)
        canvas.draw_node_label(ax, profile, f"{profile.node_id}\n熵={profile.entropy:.1f}")
    set_split_title(ax, "节点级功能复合度图")
    add_vertical_scale(ax, "行为熵", ["#FFE1CF", "#D9480F"], ["3.0", "2.5", "2.0", "1.5", "1.0", "0.0"])
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    path = os.path.join(output_dir, "node_safety_v14.png")
    fig, ax = plt.subplots(1, 1, figsize=(13, 13), facecolor="#1A1A1A")
    canvas.draw_bg(ax, show_grid=False)
    for profile in conflict_profiles:
        color = RISK_COLORS.get(profile.risk)
        if color is None:
            canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{risk_label(profile.risk)}")
            continue
        alpha = 0.7 if profile.risk == "高风险" else 0.55 if profile.risk == "中风险" else 0.4 if profile.risk == "低风险" else 0.25
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=alpha, linewidth=2.3)
        canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{risk_label(profile.risk)}")
    set_split_title(ax, "节点级 动态冲突风险图")
    add_risk_index(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    cat_peaks = _find_visual_peaks(ctx.metrics["cat_intensity"], canvas.passable_grid, 55)
    cat_clusters = _cluster_visual_peaks(cat_peaks, eps=2.0, min_samples=1)
    candidate_coords = np.array([[peak["gy"], peak["gx"]] for peak in cat_peaks], dtype=int) if cat_peaks else np.empty((0, 2), dtype=int)
    path = os.path.join(output_dir, "node_process_v14.png")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor="#1A1A1A")
    cat_intensity = ctx.metrics["cat_intensity"]
    passable = canvas.passable_grid
    vmax = float(cat_intensity[passable].max()) if np.any(passable) else 1.0
    vmax = max(vmax, 1.0)

    ax = axes[0, 0]
    canvas.draw_bg(ax, show_grid=True)
    for gy, gx in zip(*np.where(passable)):
        value = float(cat_intensity[gy, gx])
        if value <= 0:
            continue
        canvas.draw_cell(ax, gy, gx, _green_gradient(value / vmax), alpha=0.85)
    ax.set_title("(a) 离散栅格 CFS", fontsize=14, fontweight="bold", color="white")

    ax = axes[0, 1]
    canvas.draw_bg(ax, show_grid=True)
    for gy, gx in zip(*np.where(passable)):
        value = float(cat_intensity[gy, gx])
        if value <= 0:
            continue
        canvas.draw_cell(ax, gy, gx, _green_gradient(value / vmax), alpha=0.28)
    for gy, gx in candidate_coords:
        ax.plot(gx + 0.5, gy + 0.5, "r*", markersize=10, zorder=5)
    ax.set_title("(b) 峰值候选", fontsize=14, fontweight="bold", color="white")

    ax = axes[1, 0]
    canvas.draw_bg(ax, show_grid=True)
    if cat_clusters:
        palette = plt.cm.Set3(np.linspace(0, 1, max(len(cat_clusters), 1)))
        for color_idx, cluster in enumerate(cat_clusters):
            canvas.draw_node_rect(ax, cluster["bbox"], palette[color_idx], alpha=0.65, linewidth=2.0)
    ax.set_title("(c) DBSCAN 聚类", fontsize=14, fontweight="bold", color="white")

    ax = axes[1, 1]
    canvas.draw_bg(ax, show_grid=False)
    for idx, profile in enumerate(cat_profiles, start=1):
        ratio = profile.cfs / max_cat
        color = plt.cm.YlGn(0.3 + ratio * 0.7)
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=0.7, linewidth=2.0)
        canvas.draw_node_label(ax, profile, f"H{idx}")
    ax.set_title("(d) 最终节点画像", fontsize=14, fontweight="bold", color="white")
    plt.suptitle("空间诊断演变: 离散栅格 -> 峰值检测 -> DBSCAN 聚类 -> 节点画像", fontsize=18, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    return outputs


def render_final_dashboard(ctx: VisualizationContext, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    profiles = sorted(ctx.node_profiles, key=lambda item: item.cfs, reverse=True)
    cat_profiles = [profile for profile in profiles if profile.node_type == "猫节点"]
    conflict_profiles = sorted([profile for profile in profiles if profile.node_type == "共现节点"], key=lambda item: item.dcd_max, reverse=True)
    fig = plt.figure(figsize=(18, 14), facecolor="#1A1A1A")
    grid = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.18)

    ax = fig.add_subplot(grid[0, 0])
    canvas.draw_bg(ax, show_grid=False)
    if os.path.exists(ctx.trajectory_csv):
        cat_points = ctx.analyzer.df[["cat_x", "cat_y"]].dropna().to_numpy()
        human_points = ctx.analyzer.df[["human_x", "human_y"]].dropna().to_numpy()
        if len(human_points):
            human_cells = np.array([ctx.analyzer._to_grid(x, y)[::-1] for x, y in human_points], dtype=float)
            ax.plot(human_cells[:, 0] + 0.5, human_cells[:, 1] + 0.5, color="#4A90E2", linewidth=0.5, alpha=0.35, label="人")
        if len(cat_points):
            cat_cells = np.array([ctx.analyzer._to_grid(x, y)[::-1] for x, y in cat_points], dtype=float)
            ax.plot(cat_cells[:, 0] + 0.5, cat_cells[:, 1] + 0.5, color="#F4A261", linewidth=0.7, alpha=0.55, label="猫")
        ax.legend(loc="upper right", fontsize=8, facecolor="#222222", edgecolor="white", labelcolor="white")
    ax.set_title("轨迹图", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[0, 1])
    canvas.draw_bg(ax, show_grid=True)
    cat_intensity = ctx.metrics["cat_intensity"]
    passable = canvas.passable_grid
    vmax = float(cat_intensity[passable].max()) if np.any(passable) else 1.0
    vmax = max(vmax, 1.0)
    for gy, gx in zip(*np.where(passable)):
        value = float(cat_intensity[gy, gx])
        if value <= 0:
            continue
        canvas.draw_cell(ax, gy, gx, _green_gradient(value / vmax), alpha=0.88)
    ax.set_title("猫 CFS 栅格", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[0, 2])
    canvas.draw_bg(ax, show_grid=True)
    dcd = ctx.dcd_matrix
    dmax = float(dcd[passable].max()) if np.any(passable) else 0.0
    for gy, gx in zip(*np.where(passable)):
        value = float(dcd[gy, gx])
        ratio = value / dmax if dmax > 0 else 0.0
        if ratio < 0.015:
            continue
        canvas.draw_cell(ax, gy, gx, _dcd_color(ratio), alpha=0.75)
    ax.set_title("DCD 安全栅格", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[1, 0])
    canvas.draw_bg(ax, show_grid=False)
    for profile in profiles:
        color = NODE_COLORS.get(profile.node_type, "#2A9D8F" if profile.node_type == "猫节点" else "#FF4D4D")
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=0.68, linewidth=2.0)
        canvas.draw_node_label(ax, profile, profile.node_id)
    legend = [
        mpatches.Patch(facecolor="#2A9D8F", edgecolor="white", label="猫节点"),
        mpatches.Patch(facecolor="#FF4D4D", edgecolor="white", label="共现节点"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, facecolor="#222222", edgecolor="white", labelcolor="white")
    ax.set_title("节点分类", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[1, 1])
    canvas.draw_bg(ax, show_grid=False)
    for profile in conflict_profiles:
        color = RISK_COLORS.get(profile.risk)
        if color is None:
            canvas.draw_node_label(ax, profile, f"{risk_label(profile.risk)}\n{profile.dcd_max:.1f}")
            continue
        alpha = 0.7 if profile.risk == "高风险" else 0.55 if profile.risk == "中风险" else 0.4 if profile.risk == "低风险" else 0.25
        canvas.draw_node_rect(ax, profile.member_bbox, color, alpha=alpha, linewidth=2.2)
        canvas.draw_node_label(ax, profile, f"{risk_label(profile.risk)}\n{profile.dcd_max:.1f}")
    ax.set_title("节点安全", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    rows = cat_profiles[:8]
    if rows:
        table_rows = [
            [
                profile.node_id,
                profile.node_type,
                behavior_label(profile.dominant_behavior),
                risk_label(profile.risk),
                f"{profile.dcd_max:.1f}",
                strategy_label("猫专属"),
            ]
            for profile in rows
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=["节点", "类型", "行为", "风险", "DCDmax", "建议"],
            loc="center",
            cellLoc="left",
            colWidths=[0.08, 0.14, 0.12, 0.10, 0.10, 0.38],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
        for col in range(6):
            table[0, col].set_facecolor("#2C3E50")
            table[0, col].set_text_props(color="white", fontweight="bold")
    ax.set_title("节点摘要", fontsize=14, fontweight="bold", color="white", pad=8)

    plt.suptitle("最终综合仪表盘 v14", fontsize=18, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#1A1A1A")
    plt.close(fig)
    return output_path
