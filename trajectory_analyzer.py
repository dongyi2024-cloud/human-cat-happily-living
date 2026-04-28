"""
模块 A：轨迹持久化与格栅化引擎
TrajectoryAnalyzer — 读取 tick 级别轨迹数据，导出 CSV，构建行为频次矩阵。

核心原则：所有指标计算追溯到原始 tick 数据，禁止对已聚合结果二次运算。
"""

import numpy as np
import pandas as pd

# 行为权重（供模块 B 空间功能强度使用）
BEHAVIOR_WEIGHTS = {
    "奔跑": 8.0,
    "游走": 4.0,
    "移动": 4.0,
    "闲逛": 3.0,
    "观望": 2.0,
    "休息": 2.0,
    "睡眠": 1.0,
    "工作": 3.0,
    "进食": 3.0,
}

# 静止/睡眠行为集合（用于活跃共现判断）
PASSIVE_BEHAVIORS = {"休息", "睡眠"}


class TrajectoryAnalyzer:
    """
    从仿真 tick 记录构建 grid_size × grid_size 的行为频次矩阵。
    所有指标均从原始 tick 级别数据计算，避免二次聚合误差。
    """

    def __init__(self, grid_size: int = 200, proximity_cells: int = 5):
        self.grid_size = grid_size
        # 共现判断的邻域半径（欧氏距离，格栅单位），5格≈0.35m，与DBSCAN eps一致
        self.proximity_cells = proximity_cells
        self.df: pd.DataFrame | None = None

        # 每个 cell 的行为频次字典：(gy, gx) -> {behavior: count}
        self.cat_behavior_grid: dict = {}
        self.human_behavior_grid: dict = {}

        # 共现计数矩阵（tick 级别）
        self.cooccurrence_all = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.cooccurrence_active = np.zeros((grid_size, grid_size), dtype=np.int32)

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------

    def load_from_records(self, tick_records: list[dict]) -> None:
        """从仿真 tick_records 列表加载数据（直接对接 Simulation.run()）。"""
        self.df = pd.DataFrame(tick_records)
        self._build_grids()

    def load_from_csv(self, csv_path: str) -> None:
        """从已保存的 CSV 文件加载轨迹。"""
        self.df = pd.read_csv(csv_path)
        self._build_grids()

    # ------------------------------------------------------------------
    # 2. CSV 导出
    # ------------------------------------------------------------------

    def export_csv(self, csv_path: str) -> None:
        if self.df is None:
            raise RuntimeError("尚未加载数据，请先调用 load_from_records() 或 load_from_csv()")
        self.df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[模块A] 轨迹 CSV 已保存: {csv_path}  ({len(self.df)} ticks)")

    # ------------------------------------------------------------------
    # 3. 格栅化核心
    # ------------------------------------------------------------------

    def _to_grid(self, x: float, y: float) -> tuple[int, int]:
        """将连续坐标映射到格栅索引，坐标已是像素单位（0~grid_size-1）。"""
        gx = int(np.clip(x, 0, self.grid_size - 1))
        gy = int(np.clip(y, 0, self.grid_size - 1))
        return gy, gx  # 返回 (row, col) 以对应矩阵索引

    def _build_grids(self) -> None:
        """遍历每个 tick，构建行为频次字典和共现矩阵。"""
        if self.df is None:
            return

        self.cat_behavior_grid = {}
        self.human_behavior_grid = {}
        self.cooccurrence_all = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.cooccurrence_active = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        for row in self.df.itertuples(index=False):
            cgy, cgx = self._to_grid(row.cat_x, row.cat_y)
            hgy, hgx = self._to_grid(row.human_x, row.human_y)

            # 猫行为频次
            key = (cgy, cgx)
            if key not in self.cat_behavior_grid:
                self.cat_behavior_grid[key] = {}
            beh = row.cat_behavior
            self.cat_behavior_grid[key][beh] = self.cat_behavior_grid[key].get(beh, 0) + 1

            # 人行为频次
            key = (hgy, hgx)
            if key not in self.human_behavior_grid:
                self.human_behavior_grid[key] = {}
            beh = row.human_behavior
            self.human_behavior_grid[key][beh] = self.human_behavior_grid[key].get(beh, 0) + 1

            # 全状态共现：同一 tick 人猫欧氏距离 ≤ proximity_cells（格栅单位）
            # 使用邻域而非单格精确匹配，避免浮点截断零共现；5格≈0.35m
            if (cgy - hgy) ** 2 + (cgx - hgx) ** 2 <= self.proximity_cells ** 2:
                # 记录在猫位置格栅（冲突的主体）
                self.cooccurrence_all[cgy, cgx] += 1
                if row.cat_behavior not in PASSIVE_BEHAVIORS and row.human_behavior not in PASSIVE_BEHAVIORS:
                    self.cooccurrence_active[cgy, cgx] += 1

        total_ticks = len(self.df)
        cat_cells = len(self.cat_behavior_grid)
        human_cells = len(self.human_behavior_grid)
        total_cooc = int(self.cooccurrence_all.sum())
        active_cooc = int(self.cooccurrence_active.sum())
        print(f"[模块A] 格栅化完成 | 总Ticks: {total_ticks}")
        print(f"         猫活跃格栅: {cat_cells} | 人活跃格栅: {human_cells}")
        print(f"         全状态共现Ticks: {total_cooc} | 活跃共现Ticks: {active_cooc}")

    # ------------------------------------------------------------------
    # 4. 汇总矩阵（供模块 B 使用）
    # ------------------------------------------------------------------

    def get_cat_visit_matrix(self) -> np.ndarray:
        """返回猫总访问次数矩阵（grid_size × grid_size）。"""
        mat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (gy, gx), behdict in self.cat_behavior_grid.items():
            mat[gy, gx] = sum(behdict.values())
        return mat

    def get_human_visit_matrix(self) -> np.ndarray:
        """返回人总访问次数矩阵（grid_size × grid_size）。"""
        mat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (gy, gx), behdict in self.human_behavior_grid.items():
            mat[gy, gx] = sum(behdict.values())
        return mat

    def get_behavior_summary(self) -> dict:
        """返回人猫各行为的全局统计（行为 → 总次数）。"""
        cat_summary, human_summary = {}, {}
        for behdict in self.cat_behavior_grid.values():
            for beh, cnt in behdict.items():
                cat_summary[beh] = cat_summary.get(beh, 0) + cnt
        for behdict in self.human_behavior_grid.values():
            for beh, cnt in behdict.items():
                human_summary[beh] = human_summary.get(beh, 0) + cnt
        return {"cat": cat_summary, "human": human_summary}


# ===================== 独立测试入口 =====================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from simulation_v9 import generate_floor_plan, Simulation

    print("=" * 60)
    print(" 模块 A — 轨迹格栅化引擎 独立测试")
    print("=" * 60)

    floor_plan = generate_floor_plan("floor_plan.png")
    sim = Simulation(floor_plan, total_ticks=2000)
    sim.run()

    analyzer = TrajectoryAnalyzer(grid_size=200)
    analyzer.load_from_records(sim.tick_records)
    analyzer.export_csv("trajectory.csv")

    summary = analyzer.get_behavior_summary()
    print("\n[猫行为统计]")
    for beh, cnt in sorted(summary["cat"].items(), key=lambda x: -x[1]):
        print(f"  {beh:6s}: {cnt:6d} ticks")
    print("[人行为统计]")
    for beh, cnt in sorted(summary["human"].items(), key=lambda x: -x[1]):
        print(f"  {beh:6s}: {cnt:6d} ticks")

    cat_mat = analyzer.get_cat_visit_matrix()
    human_mat = analyzer.get_human_visit_matrix()
    print(f"\n猫访问矩阵非零格栅: {np.count_nonzero(cat_mat)}")
    print(f"人访问矩阵非零格栅: {np.count_nonzero(human_mat)}")
    print(f"共现矩阵最大值 (全状态): {analyzer.cooccurrence_all.max()}")
    print(f"共现矩阵最大值 (活跃): {analyzer.cooccurrence_active.max()}")
    print("\n[模块A] ✅ 测试完成")
