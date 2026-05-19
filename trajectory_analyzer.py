"""
模块 A：轨迹持久化与格栅化引擎
TrajectoryAnalyzer — 读取 tick 级别轨迹数据，导出 CSV，构建行为频次矩阵。

核心原则：所有指标计算追溯到原始 tick 数据，禁止对已聚合结果二次运算。
"""

import numpy as np
import pandas as pd

# 行为权重（供模块 B 空间功能强度使用）
BEHAVIOR_WEIGHTS = {
    "玩耍": 6.0,
    "抓挠": 3.0,
    "观察": 2.0,
    "奔跑": 8.0,
    "探索": 5.0,
    "游走": 4.0,
    "移动": 4.0,
    "闲逛": 3.0,
    "居家工作": 3.5,
    "外出": 0.0,
    "家务": 3.0,
    "照护": 3.0,
    "休闲": 2.0,
    "学习": 3.0,
    "观望": 2.0,
    "躲藏": 2.0,
    "休息": 2.0,
    "占位": 3.0,
    "亲近": 4.0,
    "睡眠": 1.0,
    "工作": 3.0,
    "进食": 3.0,
}

# 静止/睡眠行为集合（用于活跃共现判断）
PASSIVE_BEHAVIORS = {"休息", "睡眠", "躲藏", "外出", "休闲", "观察", "观望"}

CAT_BEHAVIOR_GROUP_LABELS = {
    "玩耍": {"奔跑", "探索", "游走"},
    "抓挠": {"占位"},
    "观察": {"观望"},
}
CAT_BEHAVIOR_GROUP_LOOKUP = {
    raw_label: group_label
    for group_label, raw_labels in CAT_BEHAVIOR_GROUP_LABELS.items()
    for raw_label in raw_labels
}


def summarize_cat_behavior(label: str) -> str:
    return CAT_BEHAVIOR_GROUP_LOOKUP.get(label, label)


class TrajectoryAnalyzer:
    """
    从仿真 tick 记录构建物理尺度约束下的行为频次矩阵。
    所有指标均从原始 tick 级别数据计算，避免二次聚合误差。
    """

    def __init__(
        self,
        grid_size: int | None = None,
        proximity_cells: int | None = None,
        house_width_m: float = 14.0,
        house_depth_m: float = 16.5,
        source_width_px: int = 200,
        source_height_px: int = 200,
        cell_size_min_m: float = 0.30,
        cell_size_max_m: float = 0.40,
        target_cell_size_m: float = 0.35,
        proximity_m: float = 0.35,
    ):
        self.house_width_m = float(house_width_m)
        self.house_depth_m = float(house_depth_m)
        self.source_width_px = int(source_width_px)
        self.source_height_px = int(source_height_px)

        if grid_size is None:
            self.grid_width, self.cell_width_m = self._choose_axis_cells(
                self.house_width_m, cell_size_min_m, cell_size_max_m, target_cell_size_m
            )
            self.grid_height, self.cell_height_m = self._choose_axis_cells(
                self.house_depth_m, cell_size_min_m, cell_size_max_m, target_cell_size_m
            )
        else:
            # 兼容旧调用；新流程默认不传 grid_size，改用物理尺寸约束自动生成。
            self.grid_width = self.grid_height = int(grid_size)
            self.cell_width_m = self.house_width_m / self.grid_width
            self.cell_height_m = self.house_depth_m / self.grid_height

        self.grid_shape = (self.grid_height, self.grid_width)
        self.grid_size = max(self.grid_shape)
        self.avg_cell_size_m = (self.cell_width_m + self.cell_height_m) / 2.0

        if proximity_cells is not None:
            self.proximity_m = float(proximity_cells) * self.avg_cell_size_m
        else:
            self.proximity_m = float(proximity_m)
        self.proximity_cells = max(1, int(round(self.proximity_m / self.avg_cell_size_m)))
        self.df: pd.DataFrame | None = None

        # 每个 cell 的行为频次字典：(gy, gx) -> {behavior: count}
        self.cat_behavior_grid: dict = {}
        self.human_behavior_grid: dict = {}

        # 共现计数矩阵（tick 级别）
        self.cooccurrence_all = np.zeros(self.grid_shape, dtype=np.int32)
        self.cooccurrence_active = np.zeros(self.grid_shape, dtype=np.int32)

    @staticmethod
    def _choose_axis_cells(length_m: float, min_cell_m: float, max_cell_m: float, target_cell_m: float) -> tuple[int, float]:
        """
        用循环选择某一轴的格栅数量，使单格尺寸落在 [min_cell_m, max_cell_m]，
        并尽量接近 target_cell_m。
        """
        if length_m <= 0:
            raise ValueError("户型物理长度必须大于 0")
        if not (0 < min_cell_m <= target_cell_m <= max_cell_m):
            raise ValueError("格栅尺寸范围必须满足 0 < min <= target <= max")

        best: tuple[float, int, float] | None = None
        max_cells = int(np.ceil(length_m / min_cell_m)) + 1
        for cells in range(1, max_cells + 1):
            cell_m = length_m / cells
            if min_cell_m <= cell_m <= max_cell_m:
                score = abs(cell_m - target_cell_m)
                if best is None or score < best[0]:
                    best = (score, cells, cell_m)

        if best is None:
            raise ValueError(
                f"无法为长度 {length_m:.2f}m 生成 {min_cell_m:.2f}-{max_cell_m:.2f}m 的格栅"
            )
        _, cells, cell_m = best
        return cells, cell_m

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------

    def load_from_records(self, tick_records: list[dict]) -> None:
        """从仿真 tick_records 列表加载数据（直接对接 Simulation.run()）。"""
        self.df = pd.DataFrame(tick_records)
        self._build_grids()

    def load_from_csv(self, csv_path: str) -> None:
        """从已保存的 CSV 文件加载轨迹（自动检测 GBK/UTF-8 编码）。"""
        for enc in ("utf-8-sig", "gbk", "utf-8"):
            try:
                self.df = pd.read_csv(csv_path, encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            raise ValueError(f"无法识别 {csv_path} 的文件编码")
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
        """将仿真像素坐标映射到物理尺度格栅索引。"""
        gx = int(np.clip((float(x) / self.source_width_px) * self.grid_width, 0, self.grid_width - 1))
        gy = int(np.clip((float(y) / self.source_height_px) * self.grid_height, 0, self.grid_height - 1))
        return gy, gx  # 返回 (row, col) 以对应矩阵索引

    def _build_velocity_columns(self) -> None:
        """基于相邻 tick 的坐标差分，推导速度向量与速度标量。"""
        if self.df is None or "tick" not in self.df.columns:
            return

        self.df = self.df.sort_values("tick").reset_index(drop=True)
        x_scale_m = self.house_width_m / self.source_width_px
        y_scale_m = self.house_depth_m / self.source_height_px

        for prefix in ("cat", "human"):
            x = pd.to_numeric(self.df.get(f"{prefix}_x"), errors="coerce")
            y = pd.to_numeric(self.df.get(f"{prefix}_y"), errors="coerce")
            valid = x.notna() & y.notna()
            prev_valid = valid.shift(1, fill_value=False)

            dx_px = x - x.shift(1)
            dy_px = y - y.shift(1)
            invalid = ~(valid & prev_valid)
            dx_px = dx_px.mask(invalid)
            dy_px = dy_px.mask(invalid)

            dx_m = dx_px * x_scale_m
            dy_m = dy_px * y_scale_m

            self.df[f"{prefix}_vx_px_per_tick"] = dx_px
            self.df[f"{prefix}_vy_px_per_tick"] = dy_px
            self.df[f"{prefix}_speed_px_per_tick"] = np.sqrt(dx_px**2 + dy_px**2)
            self.df[f"{prefix}_vx_m_per_tick"] = dx_m
            self.df[f"{prefix}_vy_m_per_tick"] = dy_m
            self.df[f"{prefix}_speed_m_per_tick"] = np.sqrt(dx_m**2 + dy_m**2)

    def get_velocity_vectors(self, subject: str = "cat", unit: str = "m") -> pd.DataFrame:
        """返回指定对象的逐 tick 速度向量，基于相邻 tick 坐标差分。"""
        if self.df is None:
            raise RuntimeError("尚未加载数据，请先调用 load_from_records() 或 load_from_csv()")
        if subject not in {"cat", "human"}:
            raise ValueError("subject 仅支持 'cat' 或 'human'")
        if unit not in {"m", "px"}:
            raise ValueError("unit 仅支持 'm' 或 'px'")

        suffix = "m_per_tick" if unit == "m" else "px_per_tick"
        cols = [
            "tick",
            f"{subject}_x",
            f"{subject}_y",
            f"{subject}_vx_{suffix}",
            f"{subject}_vy_{suffix}",
            f"{subject}_speed_{suffix}",
        ]
        return self.df[cols].copy()

    def _grid_distance_m(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """返回两个格栅中心点之间的近似物理距离。"""
        dy = (a[0] - b[0]) * self.cell_height_m
        dx = (a[1] - b[1]) * self.cell_width_m
        return float(np.sqrt(dx * dx + dy * dy))

    def _build_grids(self) -> None:
        """遍历每个 tick，构建行为频次字典和共现矩阵。"""
        if self.df is None:
            return

        if "cat_behavior_group" not in self.df.columns:
            self.df["cat_behavior_group"] = self.df["cat_behavior"].map(summarize_cat_behavior)
        else:
            fallback = self.df["cat_behavior"].map(summarize_cat_behavior)
            self.df["cat_behavior_group"] = self.df["cat_behavior_group"].fillna(fallback)

        self._build_velocity_columns()
        self.cat_behavior_grid = {}
        self.human_behavior_grid = {}
        self.cooccurrence_all = np.zeros(self.grid_shape, dtype=np.int32)
        self.cooccurrence_active = np.zeros(self.grid_shape, dtype=np.int32)

        for row in self.df.itertuples(index=False):
            cgy, cgx = self._to_grid(row.cat_x, row.cat_y)
            human_state = getattr(row, "human_state", "")
            human_outside = (
                human_state == "outside"
                or pd.isna(row.human_x)
                or pd.isna(row.human_y)
            )

            # 猫行为频次
            key = (cgy, cgx)
            if key not in self.cat_behavior_grid:
                self.cat_behavior_grid[key] = {}
            beh = getattr(row, "cat_behavior_group", None)
            if beh is None or (isinstance(beh, float) and pd.isna(beh)) or str(beh).strip() == "":
                beh = summarize_cat_behavior(row.cat_behavior)
            self.cat_behavior_grid[key][beh] = self.cat_behavior_grid[key].get(beh, 0) + 1

            if human_outside:
                continue

            hgy, hgx = self._to_grid(row.human_x, row.human_y)

            # 人行为频次：outside tick 不写入室内格栅，避免伪造室内位置。
            key = (hgy, hgx)
            if key not in self.human_behavior_grid:
                self.human_behavior_grid[key] = {}
            beh = row.human_behavior
            self.human_behavior_grid[key][beh] = self.human_behavior_grid[key].get(beh, 0) + 1

            # 全状态共现：同一 tick 人猫物理距离 ≤ proximity_m。
            # 使用物理距离而非单格精确匹配，避免格栅尺度调整后共现语义漂移。
            if self._grid_distance_m((cgy, cgx), (hgy, hgx)) <= self.proximity_m:
                # 记录在猫位置格栅（冲突的主体）
                self.cooccurrence_all[cgy, cgx] += 1
                if beh not in PASSIVE_BEHAVIORS and row.human_behavior not in PASSIVE_BEHAVIORS:
                    self.cooccurrence_active[cgy, cgx] += 1

        cat_cells = len(self.cat_behavior_grid)
        human_cells = len(self.human_behavior_grid)
        total_cooc = int(self.cooccurrence_all.sum())
        active_cooc = int(self.cooccurrence_active.sum())
        total_ticks = len(self.df)
        print(f"[模块A] 格栅化完成 | 总Ticks: {total_ticks}")
        print(
            f"         格栅: {self.grid_height}行 x {self.grid_width}列 | "
            f"单元: {self.cell_width_m:.3f}m x {self.cell_height_m:.3f}m | "
            f"共现半径: {self.proximity_m:.2f}m"
        )
        print(f"         猫活跃格栅: {cat_cells} | 人活跃格栅: {human_cells}")
        print(f"         全状态共现Ticks: {total_cooc} | 活跃共现Ticks: {active_cooc}")

    # ------------------------------------------------------------------
    # 4. 汇总矩阵（供模块 B 使用）
    # ------------------------------------------------------------------

    def get_cat_visit_matrix(self) -> np.ndarray:
        """返回猫总访问次数矩阵。"""
        mat = np.zeros(self.grid_shape, dtype=np.float32)
        for (gy, gx), behdict in self.cat_behavior_grid.items():
            mat[gy, gx] = sum(behdict.values())
        return mat

    def get_human_visit_matrix(self) -> np.ndarray:
        """返回人总访问次数矩阵。"""
        mat = np.zeros(self.grid_shape, dtype=np.float32)
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

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    floor_plan = generate_floor_plan(os.path.join(result_dir, "floor_plan.png"))
    sim = Simulation(floor_plan, total_ticks=2000, output_dir=result_dir)
    sim.run()

    analyzer = TrajectoryAnalyzer()
    analyzer.load_from_records(sim.tick_records)
    analyzer.export_csv(os.path.join(result_dir, "trajectory.csv"))

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
