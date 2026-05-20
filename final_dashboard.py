import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/human-cat-mplconfig")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from node_profile_builder import CAT_NODE_KIND, COEXIST_NODE_KIND
from visual_config import (
    NODE_KIND_COLORS,
    RISK_COLORS,
    behavior_label,
    node_kind_label,
    risk_label,
    strategy_label,
)
from visualization_stage_v14 import (
    GridCanvas,
    VisualizationContext,
    _dcd_color,
    _green_gradient,
)


def elevate_legend(legend) -> None:
    """确保 legend 浮在节点标签等主图标注之上。"""
    if legend is None:
        return
    legend.set_zorder(60)
    frame = legend.get_frame()
    if frame is not None:
        frame.set_zorder(60)


def render_final_dashboard(ctx: VisualizationContext, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas = GridCanvas(ctx.zone_map, ctx.passable_map, ctx.analyzer)
    profiles = sorted(ctx.node_profiles, key=lambda item: item.cfs, reverse=True)
    cat_profiles = [profile for profile in profiles if profile.node_kind == CAT_NODE_KIND]
    conflict_profiles = sorted(
        [profile for profile in profiles if profile.node_kind == COEXIST_NODE_KIND],
        key=lambda item: item.dcd_max,
        reverse=True,
    )
    fig = plt.figure(figsize=(18, 14), facecolor="#1A1A1A")
    grid = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.18)

    ax = fig.add_subplot(grid[0, 0])
    canvas.draw_bg(ax, show_grid=False)
    if os.path.exists(ctx.trajectory_csv):
        cat_points = ctx.analyzer.df[["cat_x", "cat_y"]].dropna().to_numpy()
        human_points = ctx.analyzer.df[["human_x", "human_y"]].dropna().to_numpy()
        if len(human_points):
            human_cells = np.array([ctx.analyzer._to_grid(x, y)[::-1] for x, y in human_points], dtype=float)
            ax.plot(
                human_cells[:, 0] + 0.5,
                human_cells[:, 1] + 0.5,
                color="#4A90E2",
                linewidth=0.5,
                alpha=0.35,
                label="人",
            )
        if len(cat_points):
            cat_cells = np.array([ctx.analyzer._to_grid(x, y)[::-1] for x, y in cat_points], dtype=float)
            ax.plot(
                cat_cells[:, 0] + 0.5,
                cat_cells[:, 1] + 0.5,
                color="#F4A261",
                linewidth=0.7,
                alpha=0.55,
                label="猫",
            )
        legend = ax.legend(loc="upper right", fontsize=8, facecolor="#222222", edgecolor="white", labelcolor="white")
        elevate_legend(legend)
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
        color = NODE_KIND_COLORS.get(profile.node_kind, "#2A9D8F" if profile.node_kind == CAT_NODE_KIND else "#FF4D4D")
        canvas.draw_node_rect(ax, profile.bbox, color, alpha=0.68, linewidth=2.0)
        canvas.draw_node_label(ax, profile, profile.node_id)
    legend = [
        mpatches.Patch(facecolor=NODE_KIND_COLORS[CAT_NODE_KIND], edgecolor="white", label=node_kind_label(CAT_NODE_KIND)),
        mpatches.Patch(facecolor=NODE_KIND_COLORS[COEXIST_NODE_KIND], edgecolor="white", label=node_kind_label(COEXIST_NODE_KIND)),
    ]
    legend_artist = ax.legend(handles=legend, loc="upper right", fontsize=8, facecolor="#222222", edgecolor="white", labelcolor="white")
    elevate_legend(legend_artist)
    ax.set_title("节点分类", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[1, 1])
    canvas.draw_bg(ax, show_grid=False)
    for profile in conflict_profiles:
        color = RISK_COLORS.get(profile.risk_level)
        if color is None:
            canvas.draw_node_label(ax, profile, f"{risk_label(profile.risk_level)}\n{profile.dcd_max:.1f}")
            continue
        alpha = 0.7 if profile.risk_level == "高风险" else 0.55 if profile.risk_level == "中风险" else 0.4 if profile.risk_level == "低风险" else 0.25
        canvas.draw_node_rect(ax, profile.bbox, color, alpha=alpha, linewidth=2.2)
        canvas.draw_node_label(ax, profile, f"{risk_label(profile.risk_level)}\n{profile.dcd_max:.1f}")
    ax.set_title("节点安全", fontsize=14, fontweight="bold", color="white")

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    rows = cat_profiles[:8]
    if rows:
        table_rows = [
            [
                profile.node_id,
                profile.node_kind,
                behavior_label(profile.function_type),
                risk_label(profile.risk_level),
                f"{profile.dcd_max:.1f}",
                strategy_label(profile.strategy_type or "猫专属"),
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
