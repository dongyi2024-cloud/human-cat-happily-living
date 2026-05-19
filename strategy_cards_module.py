"""
strategy_cards_module.py

从 simulation_v14.py 中归纳出来的「健康策略卡 / 安全策略卡」独立模块。

用途：
1. 输入 diagnose() 之后得到的 nodes 列表；
2. 根据节点类型自动筛选：
   - 猫节点(type == "猫节点") → 健康策略摘要卡
   - 共现节点(type == "共现节点") → 安全策略卡
3. 输出两张 PNG 图片：
   - health_card_v14.png
   - safety_card_v14.png

依赖：
    pip install matplotlib

说明：
    这个文件只保留策略卡相关代码，不包含仿真、寻路、栅格聚合、节点诊断等上游逻辑。
    你可以把它作为后处理模块接到主程序里，也可以直接运行本文件生成示例图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

# 后台渲染模式：服务器 / 命令行环境下不需要弹窗，直接保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# 1. 字体设置
# ============================================================

# 中文字体兜底顺序：
# - Windows 常见：SimHei, Microsoft YaHei
# - macOS 常见：Arial Unicode MS
# - 最后退回 DejaVu Sans，至少保证英文和数字正常。
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 策略映射表
# ============================================================

# FUNC_A 是健康策略卡和安全策略卡共同依赖的“功能类型 → 改造建议”映射。
# 在原程序中，猫节点会先由主导行为映射到功能类型：
#   奔跑 → 运动型
#   观察 → 观察型
#   躲藏 → 庇护型
# 再由 FUNC_A 输出具体空间改造建议。
FUNC_A: Dict[str, str] = {
    "运动型": "铺设跑道垫，保持通道畅通",
    "玩耍型": "预留开阔地，配置互动玩具区",
    "探索型": "保持路径转折，配置嗅觉丰容",
    "庇护型": "配置隐藏箱/半封闭猫窝",
    "观察型": "配置爬架(150-195cm)/窗台加宽",
    "休息型": "配置记忆棉软垫/加热垫",
    "进食型": "配置食盆，远离人流>=1.5m",
    "抓挠型": "配置垂直抓柱(>=91cm)",
    "动线交叉": "增设墙面猫道(1.2-1.5m)，立体分流",
}

# 安全策略卡当前使用统一的动线分流建议。
# 后续如果要细分“相向交汇 / 走廊瓶颈 / 门口交叉”，可以扩展这个字典。
SAFETY_ADVICE_BY_MECHANISM: Dict[str, str] = {
    "相向交汇": "增设墙面猫道(1.2-1.5m)，实现立体分流",
    "走廊瓶颈": "拓宽有效通行面或减少障碍物，降低人猫同线会车概率",
    "门口交叉": "在门口侧边设置猫跳台/等待点，避开开门和人移动路径",
    "默认": "增设墙面猫道(1.2-1.5m)，实现立体分流",
}


# ============================================================
# 3. nodes 数据结构约定
# ============================================================

"""
健康策略卡需要的猫节点字段：

{
    "nid": "H0",
    "type": "猫节点",
    "zone": "共享空间",
    "cfs": 83.2,
    "dom_bh": "观察",
    "func": "观察型",
    "comp": "功能复合型"
}

安全策略卡需要的共现节点字段：

{
    "nid": "S0",
    "type": "共现节点",
    "zone": "共享空间",
    "dcd_max": 12.8,
    "risk": "高风险",
    "conflict_mechanism": "走廊瓶颈"
}

说明：
    上面这些字段来自主程序 diagnose() 生成的节点画像。
    本模块不会重新计算 CFS / DCD / 风险等级，只负责把已有节点画像排版成策略卡。
"""


Node = Mapping[str, Any]


# ============================================================
# 4. 工具函数
# ============================================================

def _as_list(nodes: Iterable[Node]) -> List[Node]:
    """把任意可迭代 nodes 转成 list，避免后面重复遍历时被耗尽。"""
    return list(nodes)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把节点字段安全转成 float，缺字段或格式错误时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_parent_dir(path: str | Path) -> Path:
    """保存图片前，自动创建输出目录。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _draw_round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.0,
    zorder: int = 1,
) -> FancyBboxPatch:
    """统一绘制圆角框。策略卡所有标题栏、节点卡片、说明框都用它。"""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _setup_card_canvas(total_h: float, fig_h: float) -> tuple[plt.Figure, plt.Axes]:
    """创建策略卡画布：横向 14 英寸，高度随节点数量动态变化。"""
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_h), facecolor="#FAFAFA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    return fig, ax


# ============================================================
# 5. 健康策略卡
# ============================================================

def draw_health_card(
    nodes: Iterable[Node],
    path: str | Path = "health_card_v14.png",
    sim_steps: int | None = 10000,
) -> Path:
    """
    绘制健康策略摘要卡。

    逻辑来源：
        原 Visualizer.health_card()

    输入：
        nodes:
            diagnose() 输出的节点列表。
            本函数只读取 type == "猫节点" 且 cfs > 0.1 的节点。
        path:
            输出 PNG 路径。
        sim_steps:
            页脚中的仿真步数说明。如果不想显示步数，可以传 None。

    排序规则：
        按 CFS 从高到低排序，CFS 越高，健康改造优先级越靠前。

    视觉规则：
        前三优先级使用不同强调色；
        第 4 个及以后使用灰色普通卡片；
        每个节点固定行高，画布高度随节点数自动增加，避免文字重叠。
    """
    out_path = _ensure_parent_dir(path)
    all_nodes = _as_list(nodes)

    # 只保留猫节点；CFS 很低的节点不进入策略卡，避免“路过噪声”把卡片撑爆。
    health_nodes = sorted(
        [
            n
            for n in all_nodes
            if n.get("type") == "猫节点" and _safe_float(n.get("cfs")) > 0.1
        ],
        key=lambda n: _safe_float(n.get("cfs")),
        reverse=True,
    )
    n_nodes = len(health_nodes)

    # 布局参数。
    # unit_h 是每个节点卡片占用的纵向高度；
    # total_h 决定坐标系高度；
    # fig_h 决定最终图片实际高度。
    unit_h = 0.20
    title_h = 0.18
    header_h = 0.10
    footer_h = 0.08
    total_h = title_h + header_h + max(n_nodes, 1) * unit_h + footer_h
    fig_h = max(10, total_h * 8)

    fig, ax = _setup_card_canvas(total_h, fig_h)

    # 标题栏：说明这张卡基于 CFS 节点画像。
    _draw_round_box(
        ax,
        (0.04, total_h - title_h),
        0.92,
        title_h * 0.85,
        facecolor="#E8F5E9",
        edgecolor="#2E7D32",
        linewidth=2.5,
    )
    ax.text(
        0.5,
        total_h - title_h / 2,
        "健康策略摘要卡",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
        color="#1B5E20",
    )
    ax.text(
        0.5,
        total_h - title_h + 0.02,
        f"基于 CFS 节点画像 · 全部 {n_nodes} 个节点 · 按优先级降序",
        fontsize=11,
        ha="center",
        va="top",
        color="#666",
    )

    # 表头。
    ax.text(
        0.5,
        total_h - title_h - header_h / 2,
        "优先级 | 节点 | 功能区 | 主导行为 | CFS | 复合度 | 建议措施",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#C8E6C9",
            edgecolor="#66BB6A",
            linewidth=1.2,
        ),
    )

    # 没有猫节点时，优雅降级，不报错。
    if n_nodes == 0:
        y_center = total_h - title_h - header_h - unit_h / 2
        _draw_round_box(
            ax,
            (0.05, y_center - 0.08),
            0.90,
            0.16,
            facecolor="#F5F5F5",
            edgecolor="#BBBBBB",
            linewidth=1,
        )
        ax.text(
            0.5,
            y_center,
            "当前未检测到有效猫节点\n请检查 diagnose() 输出或 CFS 阈值",
            fontsize=14,
            ha="center",
            va="center",
            color="#666",
            linespacing=1.5,
        )

    # 节点卡片：前三名重点突出。
    colors_top3 = [
        ("#FFF8E1", "#FF8F00"),
        ("#E3F2FD", "#1976D2"),
        ("#E8F5E9", "#388E3C"),
    ]

    for i, node in enumerate(health_nodes):
        y_top = total_h - title_h - header_h - i * unit_h
        y_bottom = y_top - unit_h
        y_center = (y_top + y_bottom) / 2

        if i < 3:
            bg, edge = colors_top3[i]
            priority = ["第一优先", "第二优先", "第三优先"][i]
            title_c = "#1A1A1A"
            lw = 2.5
        else:
            bg, edge = "#F5F5F5", "#BBBBBB"
            priority = f"第{i + 1}优先"
            title_c = "#333333"
            lw = 1.2

        _draw_round_box(
            ax,
            (0.05, y_bottom + 0.01),
            0.90,
            unit_h * 0.88,
            facecolor=bg,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )

        func_type = str(node.get("func", "未知"))
        advice = FUNC_A.get(func_type, "根据节点行为类型补充具体改造措施")
        line1 = (
            f"{priority}  |  {node.get('nid', '?')}  |  {node.get('zone', '未知区域')}  |  "
            f"{node.get('dom_bh', '未知行为')} ({func_type})"
        )
        line2 = (
            f"CFS:{_safe_float(node.get('cfs')):.1f}  |  "
            f"复合度:{node.get('comp', '未知')}  |  建议: {advice}"
        )

        fs1 = 13 if i < 3 else 11
        fs2 = 11 if i < 3 else 9
        ax.text(
            0.08,
            y_center + 0.03,
            line1,
            fontsize=fs1,
            fontweight="bold",
            ha="left",
            va="center",
            color=title_c,
            zorder=3,
        )
        ax.text(
            0.08,
            y_center - 0.03,
            line2,
            fontsize=fs2,
            ha="left",
            va="center",
            color="#444444",
            zorder=3,
        )

    # 页脚说明。
    y_footer = total_h - title_h - header_h - max(n_nodes, 1) * unit_h - footer_h / 2
    step_text = f" | 仿真: {sim_steps}步" if sim_steps is not None else ""
    ax.text(
        0.5,
        y_footer,
        f"共 {n_nodes} 个猫节点 | H1=最重要(最高CFS) | AAFP/ISFM五支柱{step_text}",
        fontsize=10,
        ha="center",
        style="italic",
        color="#999",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"[策略卡] {out_path}")
    return out_path


# ============================================================
# 6. 安全策略卡
# ============================================================

def draw_safety_card(
    nodes: Iterable[Node],
    path: str | Path = "safety_card_v14.png",
) -> Path:
    """
    绘制安全策略卡。

    逻辑来源：
        原 Visualizer.safety_card()

    输入：
        nodes:
            diagnose() 输出的节点列表。
            本函数只读取 type == "共现节点" 的节点。
        path:
            输出 PNG 路径。

    排序规则：
        按 dcd_max 从高到低排序，DCDmax 越高，风险优先级越靠前。

    安全解释：
        DCD = Dynamic Conflict Density，动态冲突密度。
        在原主程序中，它由三层过滤得到：
        1. 时空重叠：人猫距离小于阈值；
        2. 动态位移：人处于“移动”，猫处于“奔跑 / 探索 / 玩耍”；
        3. 路径交叉：速度向量夹角小于 90 度，排除同向跟随。
    """
    out_path = _ensure_parent_dir(path)
    all_nodes = _as_list(nodes)

    risk_nodes = sorted(
        [n for n in all_nodes if n.get("type") == "共现节点"],
        key=lambda n: _safe_float(n.get("dcd_max")),
        reverse=True,
    )
    n_nodes = len(risk_nodes)

    # 安全卡单节点文字更多，所以 unit_h 比健康卡略高。
    unit_h = 0.24
    title_h = 0.18
    header_h = 0.10
    footer_h = 0.20
    total_h = title_h + header_h + max(n_nodes, 1) * unit_h + footer_h
    fig_h = max(8, total_h * 8)

    fig, ax = _setup_card_canvas(total_h, fig_h)

    # 标题栏。
    _draw_round_box(
        ax,
        (0.04, total_h - title_h),
        0.92,
        title_h * 0.85,
        facecolor="#FFEBEE",
        edgecolor="#C62828",
        linewidth=2.5,
    )
    ax.text(
        0.5,
        total_h - title_h / 2,
        "安全策略卡",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
        color="#B71C1C",
    )
    ax.text(
        0.5,
        total_h - title_h + 0.02,
        f"基于 DCD 三层过滤 · 全部 {n_nodes} 个风险节点 · 按 DCDmax 降序",
        fontsize=11,
        ha="center",
        va="top",
        color="#666",
    )

    # 表头。
    ax.text(
        0.5,
        total_h - title_h - header_h / 2,
        "节点 | 功能区 | 风险等级 | DCDmax | 冲突机制 | 改造建议",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFCDD2",
            edgecolor="#EF5350",
            linewidth=1.2,
        ),
    )

    if n_nodes == 0:
        # 无风险节点时，仍然输出一张可读的卡，方便报告排版保持完整。
        y_center = total_h - title_h - header_h - unit_h / 2
        _draw_round_box(
            ax,
            (0.05, y_center - 0.08),
            0.90,
            0.16,
            facecolor="#F5F5F5",
            edgecolor="#BBBBBB",
            linewidth=1,
        )
        ax.text(
            0.5,
            y_center,
            "当前户型未检测到高风险冲突节点\n人猫动线交叉风险较低",
            fontsize=14,
            ha="center",
            va="center",
            color="#666",
            linespacing=1.5,
        )
    else:
        for i, node in enumerate(risk_nodes):
            y_top = total_h - title_h - header_h - i * unit_h
            y_bottom = y_top - unit_h
            y_center = (y_top + y_bottom) / 2

            risk = str(node.get("risk", "无风险"))
            mech = str(node.get("conflict_mechanism", "相向交汇"))
            advice = SAFETY_ADVICE_BY_MECHANISM.get(
                mech,
                SAFETY_ADVICE_BY_MECHANISM["默认"],
            )

            _draw_round_box(
                ax,
                (0.05, y_bottom + 0.01),
                0.90,
                unit_h * 0.88,
                facecolor="#FFF3F3",
                edgecolor="#F44336",
                linewidth=1.5,
            )

            ax.text(
                0.08,
                y_center + 0.03,
                (
                    f"{node.get('nid', '?')}  |  {node.get('zone', '未知区域')}  |  "
                    f"{risk}  |  DCDmax:{_safe_float(node.get('dcd_max')):.1f}  |  {mech}"
                ),
                fontsize=12,
                fontweight="bold",
                ha="left",
                va="center",
                color="#C62828",
            )

            txt = (
                f"功能区: {node.get('zone', '未知区域')}  |  冲突: {mech}\n"
                f"人移动路径与猫动态路径在 {node.get('zone', '未知区域')} 发生 {mech}\n"
                f"改造: {advice}"
            )
            ax.text(
                0.08,
                y_center - 0.04,
                txt,
                fontsize=10,
                ha="left",
                va="center",
                linespacing=1.6,
                color="#444",
            )

    # 底部说明框：解释 DCD 如何筛出来，方便论文 / 汇报里说清指标逻辑。
    yb = total_h - title_h - header_h - max(n_nodes, 1) * unit_h
    info_box_h = max(0.10, yb - 0.04)
    _draw_round_box(
        ax,
        (0.05, 0.02),
        0.90,
        info_box_h,
        facecolor="#F5F5F5",
        edgecolor="#BBBBBB",
        linewidth=1,
    )

    info_text = (
        "DCD三层过滤机制:\n"
        "① 时空重叠: 人猫距离 < 阈值  |  "
        "② 动态位移: 人='移动' AND 猫∈{奔跑,探索,玩耍}\n"
        "③ 路径交叉: 速度向量夹角 < 90度 (相向/交叉，排除同向跟随)"
    )
    ax.text(
        0.08,
        0.02 + info_box_h / 2,
        info_text,
        fontsize=9,
        ha="left",
        va="center",
        color="#555",
        linespacing=1.6,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"[策略卡] {out_path}")
    return out_path


# ============================================================
# 7. 一键输出两张策略卡
# ============================================================

def draw_strategy_cards(
    nodes: Iterable[Node],
    output_dir: str | Path = ".",
    sim_steps: int | None = 10000,
) -> tuple[Path, Path]:
    """
    一键生成健康策略卡和安全策略卡。

    推荐在主程序中这样调用：

        nodes, *_ = diagnose(...)
        draw_strategy_cards(nodes, output_dir="outputs")

    返回：
        (health_card_path, safety_card_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_nodes = _as_list(nodes)
    health_path = draw_health_card(
        all_nodes,
        output_dir / "health_card_v14.png",
        sim_steps=sim_steps,
    )
    safety_path = draw_safety_card(
        all_nodes,
        output_dir / "safety_card_v14.png",
    )
    return health_path, safety_path


# ============================================================
# 8. 示例数据
# ============================================================

DEMO_NODES: List[Dict[str, Any]] = [
    {
        "nid": "H0",
        "type": "猫节点",
        "zone": "共享空间",
        "cfs": 92.4,
        "dom_bh": "观察",
        "func": "观察型",
        "comp": "功能复合型",
    },
    {
        "nid": "H1",
        "type": "猫节点",
        "zone": "窗户",
        "cfs": 78.6,
        "dom_bh": "探索",
        "func": "探索型",
        "comp": "功能高度复合型",
    },
    {
        "nid": "H2",
        "type": "猫节点",
        "zone": "猫休息区",
        "cfs": 61.3,
        "dom_bh": "休息",
        "func": "休息型",
        "comp": "功能单一型",
    },
    {
        "nid": "S0",
        "type": "共现节点",
        "zone": "共享空间",
        "dcd_max": 13.8,
        "risk": "高风险",
        "conflict_mechanism": "走廊瓶颈",
    },
    {
        "nid": "S1",
        "type": "共现节点",
        "zone": "工作区",
        "dcd_max": 8.2,
        "risk": "中风险",
        "conflict_mechanism": "相向交汇",
    },
]


if __name__ == "__main__":
    # 直接运行本文件时，会在当前目录的 demo_strategy_cards 文件夹里生成两张示例卡。
    draw_strategy_cards(DEMO_NODES, output_dir="demo_strategy_cards", sim_steps=10000)
