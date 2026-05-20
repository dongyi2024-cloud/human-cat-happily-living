from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from metrics_calculator import classify_dcd_risk
from trajectory_analyzer import TrajectoryAnalyzer


CAT_NODE_KIND = "猫活动节点"
COEXIST_NODE_KIND = "人猫共现节点"


@dataclass
class NodeProfile:
    node_id: str
    node_kind: str
    risk_level: str
    function_type: str
    strategy_type: str | None
    centroid_y: int
    centroid_x: int
    member_cells: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    cfs: float
    entropy: float
    dcd_avg: float
    dcd_max: float

    @property
    def node_type(self) -> str:
        if self.node_kind == CAT_NODE_KIND:
            return "猫节点"
        if self.node_kind == COEXIST_NODE_KIND:
            return "共现节点"
        return self.node_kind

    @property
    def risk(self) -> str:
        return self.risk_level

    @property
    def dominant_behavior(self) -> str:
        return self.function_type

    @property
    def member_bbox(self) -> tuple[int, int, int, int]:
        return self.bbox


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


def _infer_strategy_type(node_kind: str, risk_level: str, function_type: str) -> str | None:
    if node_kind == COEXIST_NODE_KIND:
        if risk_level in {"高风险", "中风险"}:
            return "安全避让"
        return "共享强化"
    if node_kind == CAT_NODE_KIND:
        if function_type in {"休息", "睡眠", "躲藏", "进食"}:
            return "健康优化"
        if function_type in {"玩耍", "探索", "观察", "抓挠", "奔跑"}:
            return "功能补充"
        return "健康优化"
    return None


def build_node_profiles(
    analyzer: TrajectoryAnalyzer,
    metrics: dict,
    dcd_matrix: np.ndarray,
    passable_grid: np.ndarray,
) -> list[NodeProfile]:
    """从指标矩阵构建可复用的节点画像。"""
    cat_intensity = metrics["cat_intensity"]
    cat_entropy = metrics["cat_entropy"]
    dcd_thresholds = metrics.get("dcd_thresholds", {})
    cat_peaks = _find_visual_peaks(cat_intensity, passable_grid, 55)
    conflict_peaks = _find_visual_peaks(dcd_matrix, passable_grid, 25)
    cat_clusters = _cluster_visual_peaks(cat_peaks, eps=2.0, min_samples=1)
    conflict_clusters = _cluster_visual_peaks(conflict_peaks, eps=2.0, min_samples=1)

    profiles: list[NodeProfile] = []
    for cluster in cat_clusters:
        cells = _profile_cells_from_bbox(cluster["bbox"], passable_grid, cluster["member_cells"])
        function_type, entropy = _weighted_behavior_for_cells(cells, analyzer, cat_intensity)
        cfs = float(np.mean([cat_intensity[gy, gx] for gy, gx in cells])) if cells else 0.0
        if entropy == 0.0 and cells:
            entropy = float(np.mean([cat_entropy[gy, gx] for gy, gx in cells]))
        dcd_values = [float(dcd_matrix[gy, gx]) for gy, gx in cells]
        dcd_avg = float(np.mean(dcd_values)) if dcd_values else 0.0
        dcd_max = float(np.max(dcd_values)) if dcd_values else 0.0
        risk_level = classify_dcd_risk(dcd_max, dcd_thresholds)
        profiles.append(
            NodeProfile(
                node_id=f"H{cluster['cluster_id']}",
                node_kind=CAT_NODE_KIND,
                risk_level=risk_level,
                function_type=function_type,
                strategy_type=_infer_strategy_type(CAT_NODE_KIND, risk_level, function_type),
                centroid_y=cluster["centroid_y"],
                centroid_x=cluster["centroid_x"],
                member_cells=cluster["member_cells"],
                bbox=cluster["bbox"],
                cfs=cfs,
                entropy=entropy,
                dcd_avg=dcd_avg,
                dcd_max=dcd_max,
            )
        )

    for cluster in conflict_clusters:
        cells = _profile_cells_from_bbox(cluster["bbox"], passable_grid, cluster["member_cells"])
        dcd_values = [float(dcd_matrix[gy, gx]) for gy, gx in cells]
        dcd_avg = float(np.mean(dcd_values)) if dcd_values else 0.0
        dcd_max = float(np.max(dcd_values)) if dcd_values else 0.0
        risk_level = classify_dcd_risk(dcd_max, dcd_thresholds)
        function_type = "动线交叉"
        profiles.append(
            NodeProfile(
                node_id=f"S{cluster['cluster_id']}",
                node_kind=COEXIST_NODE_KIND,
                risk_level=risk_level,
                function_type=function_type,
                strategy_type=_infer_strategy_type(COEXIST_NODE_KIND, risk_level, function_type),
                centroid_y=cluster["centroid_y"],
                centroid_x=cluster["centroid_x"],
                member_cells=cluster["member_cells"],
                bbox=cluster["bbox"],
                cfs=0.0,
                entropy=0.0,
                dcd_avg=dcd_avg,
                dcd_max=dcd_max,
            )
        )
    return profiles


def extract_cat_clusters(cat_intensity: np.ndarray, passable_grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    """供诊断图复用的猫节点峰值与聚类结果。"""
    peaks = _find_visual_peaks(cat_intensity, passable_grid, 55)
    clusters = _cluster_visual_peaks(peaks, eps=2.0, min_samples=1)
    return peaks, clusters
