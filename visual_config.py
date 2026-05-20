import os


FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
FONT_FAMILY = "Noto Sans CJK JP"
FONT_FALLBACKS = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
DEFAULT_FACE_COLOR = "#1A1A1A"


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

NODE_KIND_COLORS = {
    "猫活动节点": "#2A9D8F",
    "人猫共现节点": "#FF4D4D",
    "低利用节点": "#8D99AE",
    "人类活动节点": "#4A90E2",
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

NODE_KIND_LABELS = {
    "猫活动节点": "猫活动节点",
    "人猫共现节点": "人猫共现节点",
    "低利用节点": "低利用节点",
    "人类活动节点": "人类活动节点",
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
    "健康优化": "建议优化休息、躲藏与进食支持",
    "安全避让": "建议降低交叉干扰并优化避让动线",
    "共享强化": "建议强化共享停驻与柔性互动空间",
    "功能补充": "建议补充对应行为所需的空间构件",
    "低利用转化": "建议重新定义该区域的功能用途",
}

BEHAVIOR_INDEX_ORDER = ["奔跑", "玩耍", "探索", "躲藏", "观察", "休息", "进食", "抓挠"]


def behavior_label(name: str) -> str:
    return BEHAVIOR_LABELS.get(name, str(name))


def behavior_abbr(name: str) -> str:
    return BEHAVIOR_ABBR.get(name, behavior_label(name)[:1].upper())


def node_kind_label(name: str) -> str:
    return NODE_KIND_LABELS.get(name, str(name))


def risk_label(name: str) -> str:
    return RISK_LABELS.get(name, str(name))


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, "建议继续观察")


def configure_matplotlib_fonts(plt_module, font_manager_module) -> None:
    """集中配置可视化阶段使用的 CJK 字体。"""
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            font_manager_module.fontManager.addfont(font_path)
    plt_module.rcParams["font.family"] = FONT_FAMILY
    plt_module.rcParams["font.sans-serif"] = FONT_FALLBACKS
    plt_module.rcParams["axes.unicode_minus"] = False
