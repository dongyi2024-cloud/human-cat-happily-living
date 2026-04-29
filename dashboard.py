"""
模块 D：多通道可视化仪表盘
生成 2×3 六通道对比图，论文品质输出。
支持无界面后端，可在无 GUI 环境下运行。
"""

import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from trajectory_analyzer import TrajectoryAnalyzer
from metrics_calculator import SpaceMetricsCalculator
from node_detector import NodeDetector, SpaceNode

# 节点类型 → 颜色 / 标记映射（中文 key 对应 node_detector 的分类结果）
# 中文 → 英文类型名映射
_CN2EN_TYPE = {
    "冲突节点": "Conflict",
    "共享节点": "Shared",
    "猫专属":   "Cat-only",
    "人专属":   "Human-only",
    "低利用":   "Low-use",
}

NODE_STYLE: dict[str, dict] = {
    "Conflict":   {"color": "#E74C3C", "marker": "X",  "size": 180, "zorder": 6},
    "Shared":     {"color": "#F39C12", "marker": "D",  "size": 140, "zorder": 5},
    "Cat-only":   {"color": "#3498DB", "marker": "o",  "size": 100, "zorder": 4},
    "Human-only": {"color": "#2ECC71", "marker": "s",  "size": 100, "zorder": 4},
    "Low-use":    {"color": "#95A5A6", "marker": "^",  "size":  80, "zorder": 3},
}

STRATEGY = {
    "Conflict":   "Add partition / time-divide active periods",
    "Shared":     "Keep open; add shared leisure amenities",
    "Cat-only":   "Add catwalks & vertical climbing paths",
    "Human-only": "Reinforce single function; block cat access",
    "Low-use":    "Repurpose as storage / transition zone",
}

TR_CN2EN = _CN2EN_TYPE  # shorthand alias


def _add_colorbar(fig, ax, cmap, vmin, vmax, label: str) -> None:
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)


def _plot_heatmap(ax, matrix: np.ndarray, title: str, cmap: str, fig,
                  vmax=None, label: str = "Intensity") -> None:
    """通用热力图绘制。"""
    v = vmax if vmax is not None else matrix.max() or 1
    ax.imshow(matrix, origin="upper", cmap=cmap, vmin=0, vmax=v, aspect="equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")
    _add_colorbar(fig, ax, cmap, 0, v, label)


def _plot_node_symbols(ax, nodes: list[SpaceNode], grid_size: int, title: str) -> None:
    """通道 5：节点分类符号图。"""
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")

    for node in nodes:
        en_type = TR_CN2EN.get(node.node_type, "Low-use")
        style = NODE_STYLE.get(en_type, NODE_STYLE["Low-use"])
        ax.scatter(
            node.centroid_x, node.centroid_y,
            c=style["color"], marker=style["marker"],
            s=style["size"], zorder=style["zorder"],
            edgecolors="white", linewidths=0.5, alpha=0.9,
        )
        ax.annotate(
            f"#{node.node_id}",
            (node.centroid_x, node.centroid_y),
            textcoords="offset points", xytext=(4, -4),
            fontsize=5, color="white",
        )

    legend_handles = [
        mpatches.Patch(color=v["color"], label=en_type)
        for en_type, v in NODE_STYLE.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=6, framealpha=0.6, facecolor="#1a1a2e", labelcolor="white")


def _plot_design_table(ax, nodes: list[SpaceNode]) -> None:
    """通道 6：自动化设计策略建议表格。"""
    ax.axis("off")
    ax.set_title("Design Strategy Suggestions (Node Profiles)", fontsize=11, fontweight="bold", pad=6)

    display_nodes = nodes[:12]
    col_labels = ["Node", "Type", "Centroid(x,y)", "Cat Int", "Hum Int", "Design Suggestion"]
    rows = []
    for node in display_nodes:
        en = TR_CN2EN.get(node.node_type, "Low-use")
        rows.append([
            f"#{node.node_id:02d}",
            en,
            f"({node.centroid_x:.0f},{node.centroid_y:.0f})",
            f"{node.avg_cat_intensity:.1f}",
            f"{node.avg_human_intensity:.1f}",
            STRATEGY.get(en, "—"),
        ])

    if not rows:
        ax.text(0.5, 0.5, "No nodes detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        return

    col_widths = [0.06, 0.10, 0.12, 0.08, 0.08, 0.56]
    tbl = ax.table(
        cellText=rows, colLabels=col_labels, colWidths=col_widths,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.35)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    type_colors = {
        "Conflict":   "#FADBD8",
        "Shared":     "#FDEBD0",
        "Cat-only":   "#D6EAF8",
        "Human-only": "#D5F5E3",
        "Low-use":    "#F2F3F4",
    }
    for i, node in enumerate(display_nodes):
        en = TR_CN2EN.get(node.node_type, "Low-use")
        bg = type_colors.get(en, "#FFFFFF")
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)


def generate_dashboard(
    metrics: dict,
    nodes: list[SpaceNode],
    grid_size: int = 200,
    output_path: str = "dashboard.png",
) -> None:
    """
    生成六通道 2×3 可视化仪表盘。

    通道布局:
      [0,0] Cat Activity Intensity
      [0,1] Human Activity Intensity
      [0,2] Cat Behavioral Entropy
      [1,0] Active Co-occurrence (Conflict)
      [1,1] Spatial Node Classification
      [1,2] Design Strategy Table
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Human-Cat Cohabitation Space — Multi-Dimensional Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#F8F9FA")

    cat_int = metrics["cat_intensity"]
    human_int = metrics["human_intensity"]
    cat_ent = metrics["cat_entropy"]
    cooc_active = metrics["cooccurrence_active"]

    # [0,0] Cat intensity
    _plot_heatmap(axes[0, 0], cat_int, "Cat — Activity Intensity Heatmap", "YlOrRd", fig, label="Score S")

    # [0,1] Human intensity
    _plot_heatmap(axes[0, 1], human_int, "Human — Activity Intensity", "Blues", fig, label="Score S")

    # [0,2] Cat entropy
    _plot_heatmap(axes[0, 2], cat_ent, "Cat — Behavioral Entropy", "plasma", fig, label="H (nats)")

    # [1,0] Co-occurrence
    ax_cooc = axes[1, 0]
    cooc_vmax = cooc_active.max()
    if cooc_vmax > 0:
        _plot_heatmap(ax_cooc, cooc_active, "Active Co-occurrence (Conflict Points)",
                      "hot", fig, vmax=max(cooc_vmax, 1), label="Ticks")
    else:
        ax_cooc.imshow(np.zeros((grid_size, grid_size)), cmap="hot",
                       origin="upper", aspect="equal")
        ax_cooc.set_title("Active Co-occurrence\n(no active co-occurrence in this run)",
                          fontsize=11, fontweight="bold", pad=6)
        ax_cooc.axis("off")
        ax_cooc.text(grid_size / 2, grid_size / 2,
                     "No active co-occurrence\n(separated activity zones)",
                     ha="center", va="center", fontsize=10, color="white", alpha=0.8)

    # [1,1] Node symbols
    _plot_node_symbols(axes[1, 1], nodes, grid_size, "Spatial Node Classification")

    # [1,2] Design table
    _plot_design_table(axes[1, 2], nodes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[模块D] Dashboard saved: {output_path}")


# ===================== 独立测试入口 =====================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print(" Module D — Visualization Dashboard Test")
    print("=" * 60)

    analyzer = TrajectoryAnalyzer(grid_size=200)
    analyzer.load_from_csv("trajectory.csv")

    calc = SpaceMetricsCalculator(analyzer)
    metrics = calc.compute_all()

    detector = NodeDetector(metrics, intensity_pct=80, cooc_pct=90,
                            dbscan_eps=5, dbscan_min_samples=3)
    nodes = detector.detect()

    generate_dashboard(metrics, nodes, grid_size=200, output_path="dashboard.png")
    print("[Module D] Test complete")