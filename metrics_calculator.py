"""
模块 B：五维评价指标算法
SpaceMetricsCalculator — 基于 TrajectoryAnalyzer 提供的 tick 级别原始数据
计算每个格栅单元的五维评价指标。

核心原则：所有指标追溯到原始 tick 数据，禁止对已聚合热力图二次运算。
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy

from trajectory_analyzer import TrajectoryAnalyzer, BEHAVIOR_WEIGHTS, PASSIVE_BEHAVIORS


class SpaceMetricsCalculator:
    """
    五维格栅评价指标计算器。
    输入：已完成格栅化的 TrajectoryAnalyzer 实例。
    """

    def __init__(self, analyzer: TrajectoryAnalyzer):
        self.analyzer = analyzer
        self.grid_size = analyzer.grid_size

        self._cat_intensity: np.ndarray | None = None
        self._human_intensity: np.ndarray | None = None
        self._cat_entropy: np.ndarray | None = None
        self._human_entropy: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 指标 1：空间功能强度 S = Σ(count_behavior × weight_behavior)
    # ------------------------------------------------------------------

    def compute_intensity(self, subject: str = "cat") -> np.ndarray:
        """
        subject: "cat" 或 "human"
        返回 grid_size × grid_size 的强度矩阵。
        """
        grid = (self.analyzer.cat_behavior_grid if subject == "cat"
                else self.analyzer.human_behavior_grid)

        mat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (gy, gx), behdict in grid.items():
            score = sum(cnt * BEHAVIOR_WEIGHTS.get(beh, 1.0) for beh, cnt in behdict.items())
            mat[gy, gx] = score

        if subject == "cat":
            self._cat_intensity = mat
        else:
            self._human_intensity = mat
        return mat

    # ------------------------------------------------------------------
    # 指标 2：行为熵 H = -Σ p_i * log(p_i)
    # ------------------------------------------------------------------

    def compute_entropy(self, subject: str = "cat") -> np.ndarray:
        """
        对每个格栅单元的行为分布应用香农熵。
        返回 grid_size × grid_size 的熵矩阵。
        """
        grid = (self.analyzer.cat_behavior_grid if subject == "cat"
                else self.analyzer.human_behavior_grid)

        mat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (gy, gx), behdict in grid.items():
            counts = np.array(list(behdict.values()), dtype=np.float64)
            if counts.sum() > 0:
                mat[gy, gx] = float(scipy_entropy(counts))

        if subject == "cat":
            self._cat_entropy = mat
        else:
            self._human_entropy = mat
        return mat

    # ------------------------------------------------------------------
    # 指标 3 & 4：共现密度（直接从 analyzer tick 级别矩阵读取）
    # ------------------------------------------------------------------

    def get_cooccurrence_all(self) -> np.ndarray:
        """全状态共现密度矩阵。"""
        return self.analyzer.cooccurrence_all.astype(np.float32)

    def get_cooccurrence_active(self) -> np.ndarray:
        """活跃共现密度矩阵（人猫均非被动状态）。"""
        return self.analyzer.cooccurrence_active.astype(np.float32)

    # ------------------------------------------------------------------
    # 指标 5：主导行为 Top-N
    # ------------------------------------------------------------------

    def get_top_behaviors(self, subject: str = "cat", top_n: int = 3) -> dict:
        """返回每个非空格栅的 Top-N 行为列表。格式: {(gy, gx): [("行为", count), ...]}"""
        grid = (self.analyzer.cat_behavior_grid if subject == "cat"
                else self.analyzer.human_behavior_grid)
        return {
            (gy, gx): sorted(behdict.items(), key=lambda x: -x[1])[:top_n]
            for (gy, gx), behdict in grid.items()
        }

    def get_dominant_behavior_matrix(self, subject: str = "cat") -> np.ndarray:
        """返回每个格栅单元最高频行为字符串的矩阵（空格栅为空字符串）。"""
        grid = (self.analyzer.cat_behavior_grid if subject == "cat"
                else self.analyzer.human_behavior_grid)
        mat = np.full((self.grid_size, self.grid_size), "", dtype=object)
        for (gy, gx), behdict in grid.items():
            if behdict:
                mat[gy, gx] = max(behdict, key=behdict.get)
        return mat

    # ------------------------------------------------------------------
    # 汇总接口：一次性计算所有指标
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """计算全部五维指标，返回包含所有矩阵的字典。"""
        cat_intensity = self.compute_intensity("cat")
        human_intensity = self.compute_intensity("human")
        cat_entropy = self.compute_entropy("cat")
        human_entropy = self.compute_entropy("human")
        cooc_all = self.get_cooccurrence_all()
        cooc_active = self.get_cooccurrence_active()

        print("[模块B] 五维指标计算完成")
        print(f"  猫强度  — 最大值: {cat_intensity.max():.1f}  非零格栅: {np.count_nonzero(cat_intensity)}")
        print(f"  人强度  — 最大值: {human_intensity.max():.1f}  非零格栅: {np.count_nonzero(human_intensity)}")
        cat_ent_nz = cat_entropy[cat_entropy > 0]
        human_ent_nz = human_entropy[human_entropy > 0]
        cat_ent_mean = cat_ent_nz.mean() if len(cat_ent_nz) > 0 else 0
        human_ent_mean = human_ent_nz.mean() if len(human_ent_nz) > 0 else 0
        print(f"  猫熵    — 最大值: {cat_entropy.max():.3f}  均值(非零): {cat_ent_mean:.3f}")
        print(f"  人熵    — 最大值: {human_entropy.max():.3f}  均值(非零): {human_ent_mean:.3f}")
        print(f"  全状态共现 — 总计: {int(cooc_all.sum())}  峰值: {int(cooc_all.max())}")
        print(f"  活跃共现   — 总计: {int(cooc_active.sum())}  峰值: {int(cooc_active.max())}")

        return {
            "cat_intensity": cat_intensity,
            "human_intensity": human_intensity,
            "cat_entropy": cat_entropy,
            "human_entropy": human_entropy,
            "cooccurrence_all": cooc_all,
            "cooccurrence_active": cooc_active,
        }


# ===================== 独立测试入口 =====================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print(" 模块 B — 五维评价指标算法 独立测试")
    print("=" * 60)

    analyzer = TrajectoryAnalyzer(grid_size=200)
    analyzer.load_from_csv("trajectory.csv")

    calc = SpaceMetricsCalculator(analyzer)
    metrics = calc.compute_all()

    cat_visits = analyzer.get_cat_visit_matrix()
    top_cells = np.argpartition(cat_visits.ravel(), -5)[-5:]
    print("\n[猫] 高频格栅 Top-3 行为：")
    top_behaviors = calc.get_top_behaviors("cat", top_n=3)
    for flat_idx in top_cells:
        gy, gx = divmod(int(flat_idx), 200)
        behs = top_behaviors.get((gy, gx), [])
        print(f"  格栅({gx:3d},{gy:3d})  访问={int(cat_visits[gy, gx])}  Top3={behs}")

    print("\n[模块B] ✅ 测试完成")
