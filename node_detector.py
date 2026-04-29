"""
模块 C：节点峰值检测与聚类
NodeDetector — 从五维指标矩阵中提取高分格栅，聚合为具有设计意义的空间节点。

流程：阈值过滤 → DBSCAN 空间聚类 → 质心计算 → 节点分类
"""

import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field


@dataclass
class SpaceNode:
    """一个聚合空间节点（设计意义上的功能区域）。"""
    node_id: int
    centroid_x: float          # 格栅列索引
    centroid_y: float          # 格栅行索引
    cell_count: int            # 该聚类包含的格栅数量
    node_type: str             # "冲突节点" / "共享节点" / "猫专属" / "人专属" / "低利用"
    avg_cat_intensity: float
    avg_human_intensity: float
    avg_cooc_active: float
    avg_cat_entropy: float
    member_cells: list = field(default_factory=list)  # [(gy, gx), ...]


class NodeDetector:
    """
    从 SpaceMetricsCalculator 产出的指标矩阵中检测空间节点。
    """

    def __init__(
        self,
        metrics: dict,
        intensity_pct: float = 80.0,   # 强度阈值百分位（前20%）
        cooc_pct: float = 90.0,        # 共现阈值百分位（前10%）
        dbscan_eps: float = 5.0,       # DBSCAN 邻域半径（格栅单位）
        dbscan_min_samples: int = 3,   # DBSCAN 最小样本数
    ):
        self.metrics = metrics
        self.intensity_pct = intensity_pct
        self.cooc_pct = cooc_pct
        self.eps = dbscan_eps
        self.min_samples = dbscan_min_samples

        self.nodes: list[SpaceNode] = []

    # ------------------------------------------------------------------
    # 步骤 1：阈值过滤，提取候选高分格栅
    # ------------------------------------------------------------------

    def _filter_high_score_cells(self) -> np.ndarray:
        """
        合并猫/人强度矩阵，提取前 (100-intensity_pct)% 的格栅坐标。
        同时额外纳入共现密度前 (100-cooc_pct)% 的格栅。
        返回候选格栅坐标数组 shape=(N, 2)，每行为 [gy, gx]。
        """
        cat_int = self.metrics["cat_intensity"]
        human_int = self.metrics["human_intensity"]
        cooc = self.metrics["cooccurrence_active"]

        # 合并强度：取两者最大值（保留任意一方高活跃格栅）
        combined = np.maximum(cat_int, human_int)

        thr_intensity = np.percentile(combined[combined > 0], self.intensity_pct)
        thr_cooc = (np.percentile(cooc[cooc > 0], self.cooc_pct)
                    if cooc.max() > 0 else np.inf)

        mask_intensity = combined >= thr_intensity
        mask_cooc = cooc >= thr_cooc

        candidate_mask = mask_intensity | mask_cooc
        gy_arr, gx_arr = np.where(candidate_mask)
        coords = np.column_stack([gy_arr, gx_arr])

        print(f"[模块C] 候选格栅: {len(coords)}  (强度≥{thr_intensity:.1f} OR 活跃共现≥{thr_cooc:.1f})")
        return coords

    # ------------------------------------------------------------------
    # 步骤 2：DBSCAN 空间聚类
    # ------------------------------------------------------------------

    def _cluster(self, coords: np.ndarray) -> np.ndarray:
        """对候选格栅坐标做 DBSCAN，返回标签数组（-1 为噪声）。"""
        if len(coords) == 0:
            return np.array([], dtype=int)

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = db.fit_predict(coords)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"[模块C] DBSCAN 聚类: {n_clusters} 个节点  噪声点: {n_noise}")
        return labels

    # ------------------------------------------------------------------
    # 步骤 3：质心计算 + 节点分类
    # ------------------------------------------------------------------

    def _classify_node(self, avg_cat: float, avg_human: float, avg_cooc: float) -> str:
        """
        基于该聚类的平均强度与共现密度进行分类。
        - 冲突节点：活跃共现密度高（人猫双方都活跃且同时出现）
        - 共享节点：人猫强度均较高但共现低（分时共享）
        - 猫专属：猫强度显著高于人
        - 人专属：人强度显著高于猫
        - 低利用：强度均低
        """
        if avg_cooc > 0.5:
            return "冲突节点"
        ratio = (avg_cat + 1e-6) / (avg_human + 1e-6)
        if avg_cat > 5 and avg_human > 5:
            return "共享节点"
        elif ratio > 2.5:
            return "猫专属"
        elif ratio < 0.4:
            return "人专属"
        else:
            return "低利用"

    def _build_nodes(self, coords: np.ndarray, labels: np.ndarray) -> None:
        """将聚类标签转换为 SpaceNode 列表。"""
        self.nodes = []
        cat_int = self.metrics["cat_intensity"]
        human_int = self.metrics["human_intensity"]
        cooc_active = self.metrics["cooccurrence_active"]
        cat_ent = self.metrics["cat_entropy"]

        unique_labels = sorted(set(labels) - {-1})
        for nid, lbl in enumerate(unique_labels):
            mask = labels == lbl
            members = coords[mask]  # shape=(k, 2): [[gy, gx], ...]

            centroid_y = float(members[:, 0].mean())
            centroid_x = float(members[:, 1].mean())

            avg_cat = float(np.mean([cat_int[gy, gx] for gy, gx in members]))
            avg_human = float(np.mean([human_int[gy, gx] for gy, gx in members]))
            avg_cooc = float(np.mean([cooc_active[gy, gx] for gy, gx in members]))
            avg_ent = float(np.mean([cat_ent[gy, gx] for gy, gx in members]))

            node_type = self._classify_node(avg_cat, avg_human, avg_cooc)

            self.nodes.append(SpaceNode(
                node_id=nid,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                cell_count=len(members),
                node_type=node_type,
                avg_cat_intensity=avg_cat,
                avg_human_intensity=avg_human,
                avg_cooc_active=avg_cooc,
                avg_cat_entropy=avg_ent,
                member_cells=[(int(gy), int(gx)) for gy, gx in members],
            ))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def detect(self) -> list[SpaceNode]:
        """执行完整检测流程，返回节点列表。"""
        coords = self._filter_high_score_cells()
        if len(coords) == 0:
            print("[模块C] 无候选格栅，检测结束")
            return []

        labels = self._cluster(coords)
        self._build_nodes(coords, labels)

        # 打印节点摘要
        type_counts: dict[str, int] = {}
        for node in self.nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
        print(f"[模块C] 节点分类统计: {type_counts}")
        print(f"[模块C] 共识别 {len(self.nodes)} 个空间节点")
        return self.nodes

    def print_node_profiles(self) -> None:
        """打印每个节点的画像摘要。"""
        if not self.nodes:
            print("[模块C] 尚未运行 detect()，无节点数据")
            return

        print("\n" + "=" * 60)
        print(" 空间节点画像")
        print("=" * 60)
        for node in self.nodes:
            print(f"节点 #{node.node_id:02d}  [{node.node_type}]")
            print(f"  质心: ({node.centroid_x:.1f}, {node.centroid_y:.1f})  "
                  f"格栅数: {node.cell_count}")
            print(f"  猫强度均值: {node.avg_cat_intensity:.2f}  "
                  f"人强度均值: {node.avg_human_intensity:.2f}")
            print(f"  活跃共现均值: {node.avg_cooc_active:.3f}  "
                  f"猫熵均值: {node.avg_cat_entropy:.3f}")
            print()


# ===================== 独立测试入口 =====================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from trajectory_analyzer import TrajectoryAnalyzer
    from metrics_calculator import SpaceMetricsCalculator

    print("=" * 60)
    print(" 模块 C — 节点峰值检测与聚类 独立测试")
    print("=" * 60)

    analyzer = TrajectoryAnalyzer(grid_size=200)
    analyzer.load_from_csv("trajectory.csv")

    calc = SpaceMetricsCalculator(analyzer)
    metrics = calc.compute_all()

    detector = NodeDetector(metrics, intensity_pct=80, cooc_pct=90, dbscan_eps=5, dbscan_min_samples=3)
    nodes = detector.detect()
    detector.print_node_profiles()

    print(f"\n[模块C] ✅ 测试完成，共 {len(nodes)} 个节点")
