import os
import warnings
from collections import Counter
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/human-cat-mplconfig")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from metrics_calculator import SpaceMetricsCalculator
from node_profile_builder import (
    CAT_NODE_KIND,
    COEXIST_NODE_KIND,
    NodeProfile,
    build_node_profiles,
    extract_cat_clusters,
)
from project_paths import ensure_project_dir, resolve_project_path
from simulation_v9 import (
    CAT_BEHAVIOR_LABELS,
    CAT_CONFIG,
    HUMAN_CONFIG,
    FloorPlanParser,
    RAINBOW_CMAP,
    Simulation,
    generate_floor_plan,
)
from trajectory_analyzer import TrajectoryAnalyzer
from visual_config import (
    BEHAVIOR_COLORS,
    BEHAVIOR_INDEX_ORDER,
    DEFAULT_FACE_COLOR,
    RISK_COLORS,
    behavior_abbr,
    behavior_label,
    configure_matplotlib_fonts,
    risk_label,
)

# Matplotlib resolves the installed Noto CJK TTC as JP even though fontconfig
# exposes SC aliases; the glyph coverage is shared across the CJK family.
configure_matplotlib_fonts(plt, font_manager)
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")


@dataclass
class VisualizationContext:
    floor_plan_path: str
    trajectory_csv: str
    output_dir: str
    floor_plan_image: np.ndarray
    zone_map: np.ndarray
    passable_map: np.ndarray
    analyzer: TrajectoryAnalyzer
    metrics: dict
    dcd_matrix: np.ndarray
    node_profiles: list[NodeProfile]


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

    def draw_node_label(self, ax, profile: NodeProfile, label: str) -> None:
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


def set_split_title(ax, main: str, subtitle: str | None = None, size: int = 18) -> None:
    title = main if not subtitle else f"{main}\n{subtitle}"
    ax.set_title(title, fontsize=size, fontweight="bold", color="white", pad=18, loc="center")


def get_figsize_from_grid(grid_width: int, grid_height: int, base_height: float = 13.0) -> tuple[float, float]:
    """根据 grid 宽高比生成单图画布，避免强行塞进正方形。"""
    safe_height = max(grid_height, 1)
    ratio = max(grid_width, 1) / safe_height
    width = base_height * ratio
    return (max(8.0, width), base_height)


def render_single_map(
    output_path: str,
    canvas: GridCanvas,
    title: str,
    draw_fn,
    legend_fn=None,
    subtitle: str | None = None,
    show_grid: bool = False,
    figsize: tuple[float, float] | None = None,
    facecolor: str = DEFAULT_FACE_COLOR,
    dpi: int = 200,
    use_tight_bbox: bool = False,
) -> str:
    """统一单张空间图的画布、标题、保存和关闭流程。"""
    figsize = figsize or get_figsize_from_grid(canvas.grid_width, canvas.grid_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=facecolor)
    canvas.draw_bg(ax, show_grid=show_grid)
    draw_fn(ax)
    set_split_title(ax, title, subtitle)
    if legend_fn is not None:
        legend_fn(ax)
    plt.tight_layout()
    savefig_kwargs = {"dpi": dpi, "facecolor": facecolor}
    if use_tight_bbox:
        savefig_kwargs["bbox_inches"] = "tight"
    plt.savefig(output_path, **savefig_kwargs)
    plt.close(fig)
    return output_path


def create_overlay_card_axes(ax, bounds: list[float]) -> plt.Axes:
    """创建浮在主图之上的图例卡片轴，避免被底图文字遮挡。"""
    index_ax = ax.inset_axes(bounds, transform=ax.transAxes)
    index_ax.set_zorder(50)
    index_ax.patch.set_facecolor((0, 0, 0, 0.58))
    index_ax.patch.set_zorder(50)
    return index_ax


def _add_heatmap_weight(visit_count: np.ndarray, y: float, x: float, weight: float) -> None:
    iy = int(y)
    ix = int(x)
    h, w = visit_count.shape
    if 0 <= iy < h and 0 <= ix < w:
        visit_count[iy, ix] += weight
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < h and 0 <= nx < w:
                visit_count[ny, nx] += weight * (0.5 / (1 + np.sqrt(dy * dy + dx * dx)))


def build_legacy_visit_heatmaps(ctx: VisualizationContext) -> tuple[np.ndarray, np.ndarray]:
    """从轨迹 CSV 复原旧版像素级热力图计数语义。"""
    img_h, img_w = ctx.zone_map.shape
    cat_visit_count = np.zeros((img_h, img_w), dtype=float)
    human_visit_count = np.zeros((img_h, img_w), dtype=float)
    df = ctx.analyzer.df
    if df is None or df.empty:
        return cat_visit_count, human_visit_count

    first_cat = df[["cat_x", "cat_y"]].dropna().head(1)
    if not first_cat.empty:
        x, y = first_cat.iloc[0]
        _add_heatmap_weight(cat_visit_count, y, x, CAT_CONFIG["heatmap_weight"] * 3)

    first_human = df[["human_x", "human_y"]].dropna().head(1)
    if not first_human.empty:
        x, y = first_human.iloc[0]
        _add_heatmap_weight(human_visit_count, y, x, HUMAN_CONFIG["heatmap_weight"] * 2)

    for row in df.itertuples(index=False):
        if not np.isnan(row.cat_x) and not np.isnan(row.cat_y):
            cat_weight = CAT_CONFIG["run_heatmap_weight"] if row.cat_behavior == CAT_BEHAVIOR_LABELS["run"] else CAT_CONFIG["heatmap_weight"]
            _add_heatmap_weight(cat_visit_count, row.cat_y, row.cat_x, cat_weight)
        if not np.isnan(row.human_x) and not np.isnan(row.human_y):
            _add_heatmap_weight(human_visit_count, row.human_y, row.human_x, HUMAN_CONFIG["heatmap_weight"])

    return cat_visit_count, human_visit_count


def render_legacy_heatmap_pair(ctx: VisualizationContext, output_path: str) -> str:
    """输出旧版像素级猫/人热力图，并放到同一张平面图里。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cat_visit_count, human_visit_count = build_legacy_visit_heatmaps(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.8), facecolor=DEFAULT_FACE_COLOR)

    panels = [
        ("猫热力图", cat_visit_count, axes[0]),
        ("人热力图", human_visit_count, axes[1]),
    ]
    for title, visit_count, ax in panels:
        heat = gaussian_filter(visit_count, sigma=1.2)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat = np.power(heat, 0.4)
        im = ax.imshow(heat, cmap=RAINBOW_CMAP, alpha=1.0, vmin=0, vmax=1)
        ax.imshow(ctx.floor_plan_image, alpha=0.15)
        ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=14)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="white", labelsize=9)
        cbar.outline.set_edgecolor("white")
        cbar.ax.set_facecolor(DEFAULT_FACE_COLOR)

    plt.suptitle("旧版像素级热力图对照", fontsize=18, fontweight="bold", color="white", y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, facecolor=DEFAULT_FACE_COLOR)
    plt.close(fig)
    return output_path


def add_color_index(ax, title: str, entries: list[tuple[str, str]], width: float = 0.18, height: float | None = None) -> None:
    height = height or min(max(0.15, 0.025 * len(entries) + 0.11), 0.34) * 0.5
    index_ax = create_overlay_card_axes(ax, [0.805, 0.985 - height, width, height])
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
    index_ax = create_overlay_card_axes(ax, [0.805, 0.9075, 0.18, 0.0775])
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


def add_dcd_index(ax, thresholds: dict[str, float]) -> None:
    p60 = float(thresholds.get("p60", 0.0))
    p80 = float(thresholds.get("p80", 0.0))
    add_color_index(
        ax,
        "DCD风险",
        [
            (_dcd_color(0.20), f"低风险(0-{p60:.1f})"),
            (_dcd_color(0.55), f"中风险({p60:.1f}-{p80:.1f})"),
            (_dcd_color(0.90), f"高风险(>{p80:.1f})"),
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
    index_ax = create_overlay_card_axes(ax, [1.015, 0.5 - height / 2, width, height])
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


def ensure_analysis_inputs(output_dir: str, floor_plan_path: str | None = None, trajectory_csv: str | None = None, total_ticks: int = 1440, random_seed: int | None = 7) -> tuple[str, str]:
    output_dir = str(ensure_project_dir(output_dir))
    default_floor_plan, default_trajectory = _default_paths(output_dir)
    floor_plan_path = str(resolve_project_path(floor_plan_path or default_floor_plan))
    trajectory_csv = str(resolve_project_path(trajectory_csv or default_trajectory))
    if os.path.exists(floor_plan_path) and os.path.exists(trajectory_csv):
        return floor_plan_path, trajectory_csv

    if not os.path.exists(floor_plan_path):
        generate_floor_plan(floor_plan_path)

    sim = Simulation(floor_plan_path, total_ticks=total_ticks, output_dir=output_dir, random_seed=random_seed)
    sim.run()
    sim.export_tick_records_csv(trajectory_csv)
    return floor_plan_path, trajectory_csv


def build_context(output_dir: str = "result", floor_plan_path: str | None = None, trajectory_csv: str | None = None, total_ticks: int = 1440, random_seed: int | None = 7) -> VisualizationContext:
    floor_plan_path, trajectory_csv = ensure_analysis_inputs(output_dir, floor_plan_path, trajectory_csv, total_ticks, random_seed)
    parser = FloorPlanParser(floor_plan_path)
    floor_plan_image, zone_map, passable_maps, _, _, _ = parser.parse()
    analyzer = TrajectoryAnalyzer(
        house_width_m=parser.house_width_m,
        house_depth_m=parser.house_depth_m,
        source_width_px=parser.img_width,
        source_height_px=parser.img_height,
    )
    analyzer.load_from_csv(trajectory_csv)
    calculator = SpaceMetricsCalculator(analyzer)
    metrics = calculator.compute_all()
    dcd_matrix = metrics["dcd_matrix"]
    canvas = GridCanvas(zone_map, passable_maps["cat"], analyzer)
    node_profiles = build_node_profiles(analyzer, metrics, dcd_matrix, canvas.passable_grid)
    return VisualizationContext(
        floor_plan_path=floor_plan_path,
        trajectory_csv=trajectory_csv,
        output_dir=output_dir,
        floor_plan_image=floor_plan_image,
        zone_map=zone_map,
        passable_map=passable_maps["cat"],
        analyzer=analyzer,
        metrics=metrics,
        dcd_matrix=dcd_matrix,
        node_profiles=node_profiles,
    )


def render_grid_stage(ctx: VisualizationContext, output_dir: str) -> list[str]:
    output_dir = str(ensure_project_dir(output_dir))
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    cat_intensity = ctx.metrics["cat_intensity"]
    cat_entropy = ctx.metrics["cat_entropy"]
    dominant_behavior = ctx.metrics["dominant_behavior"]
    dcd_thresholds = ctx.metrics["dcd_thresholds"]
    passable = canvas.passable_grid
    vmax = float(cat_intensity[passable].max()) if np.any(passable) else 1.0
    vmax = max(vmax, 1.0)
    threshold = float(np.percentile(cat_intensity[passable], 75)) if np.any(passable) else 0.0
    entropy_values = [float(cat_entropy[gy, gx]) for gy, gx in zip(*np.where(passable)) if float(cat_intensity[gy, gx]) >= threshold]
    emax = max(entropy_values) if entropy_values else 1.0
    emax = max(emax, 1.0)
    dcd = ctx.dcd_matrix
    dmax = float(dcd[passable].max()) if np.any(passable) else 0.0

    def draw_grid_cfs(ax) -> None:
        for gy, gx in zip(*np.where(passable)):
            value = float(cat_intensity[gy, gx])
            if value <= 0:
                continue
            ratio = value / vmax
            canvas.draw_cell(ax, gy, gx, _green_gradient(ratio), alpha=0.88, edgecolor="#555555", lw=0.35)
            if value > vmax * 0.35:
                canvas.draw_text(ax, gy, gx, f"{value:.0f}", color="white" if ratio > 0.55 else "#2F4F2F", size=7)

    def draw_grid_behavior(ax) -> None:
        for gy, gx in zip(*np.where(passable)):
            if float(cat_intensity[gy, gx]) < threshold:
                continue
            behavior = dominant_behavior[gy, gx]
            if not behavior:
                continue
            canvas.draw_cell(ax, gy, gx, BEHAVIOR_COLORS.get(behavior, "#A9A9A9"), alpha=0.82, edgecolor="white", lw=0.45)
            canvas.draw_text(ax, gy, gx, behavior_abbr(behavior), size=8)

    def draw_grid_entropy(ax) -> None:
        for gy, gx in zip(*np.where(passable)):
            if float(cat_intensity[gy, gx]) < threshold:
                continue
            ent = float(cat_entropy[gy, gx])
            ratio = ent / emax
            canvas.draw_cell(ax, gy, gx, _warm_entropy_color(ratio), alpha=0.68, edgecolor="#888888", lw=0.3)
            ax.add_patch(
                plt.Rectangle(
                    (gx, gy),
                    1,
                    1,
                    fill=False,
                    edgecolor=_dcd_color(ratio),
                    linewidth=0.4 + ratio * 2.5,
                )
            )
            canvas.draw_text(ax, gy, gx, f"{ent:.1f}", color="white" if ratio > 0.5 else "#333333", size=7)

    def draw_grid_dcd(ax) -> None:
        if dmax <= 0:
            return
        for gy, gx in zip(*np.where(passable)):
            value = float(dcd[gy, gx])
            ratio = value / dmax
            if ratio < 0.015:
                continue
            canvas.draw_cell(ax, gy, gx, _dcd_color(ratio), alpha=0.72, edgecolor="#555555", lw=0.35)
            if value > dmax * 0.12:
                canvas.draw_text(ax, gy, gx, f"{value:.1f}", color="white" if ratio > 0.5 else "#4A0000", size=7)

    grid_specs = [
        {
            "filename": "grid_cfs_v14.png",
            "title": "栅格级 CFS空间强度图",
            "subtitle": "离散着色·绿色深浅",
            "show_grid": True,
            "legend_fn": add_cfs_index,
            "draw_fn": draw_grid_cfs,
        },
        {
            "filename": "grid_behavior_v14.png",
            "title": "栅格级主导行为类型图",
            "subtitle": "仅高频栅格P75",
            "show_grid": True,
            "legend_fn": add_behavior_index,
            "draw_fn": draw_grid_behavior,
        },
        {
            "filename": "grid_entropy_v14.png",
            "title": "栅格级行为熵图",
            "subtitle": "功能复合度·边框粗细映射",
            "show_grid": True,
            "legend_fn": add_entropy_index,
            "draw_fn": draw_grid_entropy,
        },
        {
            "filename": "grid_dcd_v14.png",
            "title": "栅格级DCD安全图",
            "subtitle": "动态冲突密度·红色深浅",
            "show_grid": True,
            "legend_fn": lambda ax: add_dcd_index(ax, dcd_thresholds),
            "draw_fn": draw_grid_dcd,
        },
    ]

    outputs = [
        render_single_map(
            output_path=os.path.join(output_dir, spec["filename"]),
            canvas=canvas,
            title=spec["title"],
            subtitle=spec["subtitle"],
            show_grid=spec["show_grid"],
            legend_fn=spec["legend_fn"],
            draw_fn=spec["draw_fn"],
        )
        for spec in grid_specs
    ]
    outputs.append(render_legacy_heatmap_pair(ctx, os.path.join(output_dir, "human_cat_heatmaps_v14.png")))
    return outputs


def render_node_stage(ctx: VisualizationContext, output_dir: str) -> list[str]:
    output_dir = str(ensure_project_dir(output_dir))
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    profiles = sorted(ctx.node_profiles, key=lambda item: item.cfs, reverse=True)
    cat_profiles = [profile for profile in profiles if profile.node_kind == CAT_NODE_KIND]
    conflict_profiles = sorted([profile for profile in profiles if profile.node_kind == COEXIST_NODE_KIND], key=lambda item: item.dcd_max, reverse=True)
    max_cat = max([profile.cfs for profile in cat_profiles] + [1.0])
    max_entropy = max([profile.entropy for profile in cat_profiles] + [1.0])

    def draw_node_importance(ax) -> None:
        for profile in cat_profiles:
            ratio = profile.cfs / max_cat
            color = plt.cm.YlGn(0.3 + ratio * 0.7)
            canvas.draw_node_rect(ax, profile.bbox, color, alpha=0.75, linewidth=2.3)
            canvas.draw_node_label(ax, profile, profile.node_id)

    def draw_node_function(ax) -> None:
        for profile in cat_profiles:
            color = BEHAVIOR_COLORS.get(profile.function_type, "#A9A9A9")
            canvas.draw_node_rect(ax, profile.bbox, color, alpha=0.72, linewidth=2.0)
            canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{behavior_label(profile.function_type)}")

    def draw_node_entropy(ax) -> None:
        for profile in cat_profiles:
            ratio = profile.entropy / max_entropy
            color = plt.cm.YlOrRd(0.2 + ratio * 0.7)
            canvas.draw_node_rect(ax, profile.bbox, color, alpha=0.72, linewidth=2.0)
            canvas.draw_node_label(ax, profile, f"{profile.node_id}\n熵={profile.entropy:.1f}")

    def draw_node_safety(ax) -> None:
        for profile in conflict_profiles:
            color = RISK_COLORS.get(profile.risk_level)
            if color is None:
                canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{risk_label(profile.risk_level)}")
                continue
            alpha = 0.7 if profile.risk_level == "高风险" else 0.55 if profile.risk_level == "中风险" else 0.4 if profile.risk_level == "低风险" else 0.25
            canvas.draw_node_rect(ax, profile.bbox, color, alpha=alpha, linewidth=2.3)
            canvas.draw_node_label(ax, profile, f"{profile.node_id}\n{risk_label(profile.risk_level)}")

    node_specs = [
        {
            "filename": "node_importance_v14.png",
            "title": "节点级空间重要性图",
            "legend_fn": lambda ax: add_vertical_scale(ax, "空间重要性", ["#E6F5E6", "#228B22"], ["100", "80", "60", "40", "20", "0"]),
            "draw_fn": draw_node_importance,
        },
        {
            "filename": "node_function_v14.png",
            "title": "节点级 功能定位图",
            "legend_fn": add_behavior_index,
            "draw_fn": draw_node_function,
        },
        {
            "filename": "node_entropy_v14.png",
            "title": "节点级功能复合度图",
            "legend_fn": lambda ax: add_vertical_scale(ax, "行为熵", ["#FFE1CF", "#D9480F"], ["3.0", "2.5", "2.0", "1.5", "1.0", "0.0"]),
            "draw_fn": draw_node_entropy,
        },
        {
            "filename": "node_safety_v14.png",
            "title": "节点级 动态冲突风险图",
            "legend_fn": add_risk_index,
            "draw_fn": draw_node_safety,
        },
    ]

    outputs = [
        render_single_map(
            output_path=os.path.join(output_dir, spec["filename"]),
            canvas=canvas,
            title=spec["title"],
            draw_fn=spec["draw_fn"],
            legend_fn=spec["legend_fn"],
        )
        for spec in node_specs
    ]

    cat_peaks, cat_clusters = extract_cat_clusters(ctx.metrics["cat_intensity"], canvas.passable_grid)
    candidate_coords = np.array([[peak["gy"], peak["gx"]] for peak in cat_peaks], dtype=int) if cat_peaks else np.empty((0, 2), dtype=int)
    path = os.path.join(output_dir, "node_process_v14.png")
    cat_intensity = ctx.metrics["cat_intensity"]
    passable = canvas.passable_grid
    vmax = float(cat_intensity[passable].max()) if np.any(passable) else 1.0
    vmax = max(vmax, 1.0)
    process_size = get_figsize_from_grid(canvas.grid_width, canvas.grid_height, base_height=7.5)
    fig, axes = plt.subplots(2, 2, figsize=(process_size[0] * 2, process_size[1] * 2), facecolor="#1A1A1A")

    def draw_process_cfs_panel(ax) -> None:
        canvas.draw_bg(ax, show_grid=True)
        for gy, gx in zip(*np.where(passable)):
            value = float(cat_intensity[gy, gx])
            if value <= 0:
                continue
            canvas.draw_cell(ax, gy, gx, _green_gradient(value / vmax), alpha=0.85)
        ax.set_title("(a) 离散栅格 CFS", fontsize=14, fontweight="bold", color="white")

    def draw_process_peak_panel(ax) -> None:
        canvas.draw_bg(ax, show_grid=True)
        for gy, gx in zip(*np.where(passable)):
            value = float(cat_intensity[gy, gx])
            if value <= 0:
                continue
            canvas.draw_cell(ax, gy, gx, _green_gradient(value / vmax), alpha=0.28)
        for gy, gx in candidate_coords:
            ax.plot(gx + 0.5, gy + 0.5, "r*", markersize=10, zorder=5)
        ax.set_title("(b) 峰值候选", fontsize=14, fontweight="bold", color="white")

    def draw_process_cluster_panel(ax) -> None:
        canvas.draw_bg(ax, show_grid=True)
        if cat_clusters:
            palette = plt.cm.Set3(np.linspace(0, 1, max(len(cat_clusters), 1)))
            for color_idx, cluster in enumerate(cat_clusters):
                canvas.draw_node_rect(ax, cluster["bbox"], palette[color_idx], alpha=0.65, linewidth=2.0)
        ax.set_title("(c) DBSCAN 聚类", fontsize=14, fontweight="bold", color="white")

    def draw_process_profile_panel(ax) -> None:
        canvas.draw_bg(ax, show_grid=False)
        for profile in cat_profiles:
            ratio = profile.cfs / max_cat
            color = plt.cm.YlGn(0.3 + ratio * 0.7)
            canvas.draw_node_rect(ax, profile.bbox, color, alpha=0.7, linewidth=2.0)
            canvas.draw_node_label(ax, profile, profile.node_id)
        ax.set_title("(d) 最终节点画像", fontsize=14, fontweight="bold", color="white")

    draw_process_cfs_panel(axes[0, 0])
    draw_process_peak_panel(axes[0, 1])
    draw_process_cluster_panel(axes[1, 0])
    draw_process_profile_panel(axes[1, 1])
    plt.suptitle("空间诊断演变: 离散栅格 -> 峰值检测 -> DBSCAN 聚类 -> 节点画像", fontsize=18, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=200, facecolor="#1A1A1A")
    plt.close(fig)
    outputs.append(path)

    return outputs
