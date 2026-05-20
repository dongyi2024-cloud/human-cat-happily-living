from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from final_dashboard import render_final_dashboard
from metrics_calculator import SpaceMetricsCalculator
from node_detector import NodeDetector
from project_paths import ensure_project_dir, project_relative_display, resolve_project_path
from simulation_v9 import (
    Simulation,
    build_strategy_card_nodes,
    generate_floor_plan,
    get_cat_preset_profiles,
)
from strategy_cards_module import draw_strategy_cards
from trajectory_analyzer import TrajectoryAnalyzer
from visualization_stage_v14 import build_context, render_grid_stage, render_node_stage


def prepare_floor_plan(result_dir: Path, floor_plan_arg: str | None) -> Path:
    target_path = result_dir / "floor_plan.png"
    if floor_plan_arg is None:
        generate_floor_plan(str(target_path))
        return target_path

    source_path = resolve_project_path(floor_plan_arg)
    if source_path is None or not source_path.exists():
        raise FileNotFoundError(f"找不到户型图: {floor_plan_arg}")
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def build_analyzer_from_simulation(sim: Simulation, trajectory_path: Path) -> TrajectoryAnalyzer:
    analyzer = TrajectoryAnalyzer(
        house_width_m=sim.parser.house_width_m,
        house_depth_m=sim.parser.house_depth_m,
        source_width_px=sim.parser.img_width,
        source_height_px=sim.parser.img_height,
    )
    analyzer.load_from_records(sim.tick_records)
    analyzer.export_csv(str(trajectory_path))
    return analyzer


def run_pipeline(
    result_dir: str = "result",
    total_ticks: int = 1440,
    random_seed: int | None = 7,
    floor_plan: str | None = None,
    human_profile_id: str = "default_china",
    cat_preset: str | None = None,
) -> dict[str, object]:
    resolved_result_dir = ensure_project_dir(result_dir)
    resolved_floor_plan = prepare_floor_plan(resolved_result_dir, floor_plan)
    cat_profile = None
    if cat_preset:
        presets = get_cat_preset_profiles()
        if cat_preset not in presets:
            raise ValueError(f"未知猫预设: {cat_preset}，可选: {', '.join(sorted(presets))}")
        cat_profile = presets[cat_preset]

    sim = Simulation(
        str(resolved_floor_plan),
        total_ticks=total_ticks,
        output_dir=str(resolved_result_dir),
        random_seed=random_seed,
        cat_profile=cat_profile,
        human_profile_id=human_profile_id,
    )
    sim.run()

    simulation_figure = resolved_result_dir / "simulation_result.png"
    sim.visualize(save_path=str(simulation_figure))
    exported_data = sim.export_outputs(output_dir=str(resolved_result_dir))

    trajectory_path = resolved_result_dir / "trajectory.csv"
    analyzer = build_analyzer_from_simulation(sim, trajectory_path)
    metrics = SpaceMetricsCalculator(analyzer).compute_all()
    detector = NodeDetector(metrics, intensity_pct=80, cooc_pct=90, dbscan_eps=2, dbscan_min_samples=3)
    nodes = detector.detect()
    strategy_nodes = build_strategy_card_nodes(nodes, analyzer, sim)
    strategy_outputs = draw_strategy_cards(
        strategy_nodes,
        output_dir=resolved_result_dir / "strategy_cards",
        sim_steps=sim.total_ticks,
    )

    ctx = build_context(
        output_dir=str(resolved_result_dir),
        floor_plan_path=str(resolved_floor_plan),
        trajectory_csv=str(trajectory_path),
        total_ticks=total_ticks,
        random_seed=random_seed,
    )
    grid_outputs = render_grid_stage(ctx, str(resolved_result_dir / "grid_stage_v14"))
    node_outputs = render_node_stage(ctx, str(resolved_result_dir / "node_stage_v14"))
    final_dashboard = render_final_dashboard(ctx, str(resolved_result_dir / "final_stage_v14" / "final_dashboard_v14.png"))

    return {
        "result_dir": resolved_result_dir,
        "simulation_figure": simulation_figure,
        "trajectory_csv": trajectory_path,
        "exported_data": exported_data,
        "strategy_cards": strategy_outputs,
        "grid_outputs": grid_outputs,
        "node_outputs": node_outputs,
        "final_dashboard": final_dashboard,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行人宠共居仿真、分析和可视化全流程")
    parser.add_argument("--result-dir", default="result", help="所有输出统一写入该目录，默认 result/")
    parser.add_argument("--ticks", type=int, default=1440, help="仿真总 tick 数，默认 1440")
    parser.add_argument("--random-seed", type=int, default=7, help="随机种子，默认 7")
    parser.add_argument("--floor-plan", default=None, help="可选：自定义户型图路径；若提供，会复制到 result/floor_plan.png")
    parser.add_argument("--human-profile-id", default="default_china", help="人类画像 ID，默认 default_china")
    parser.add_argument("--cat-preset", default=None, help="猫预设名称，可选 sensitive_hiding / curious_active / friendly_companion / senior_arthritis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        result_dir=args.result_dir,
        total_ticks=args.ticks,
        random_seed=args.random_seed,
        floor_plan=args.floor_plan,
        human_profile_id=args.human_profile_id,
        cat_preset=args.cat_preset,
    )

    print("=" * 60)
    print("一键运行完成")
    print("=" * 60)
    print(f"输出目录: {project_relative_display(outputs['result_dir'])}")
    print(f"主仿真图: {project_relative_display(outputs['simulation_figure'])}")
    print(f"轨迹 CSV: {project_relative_display(outputs['trajectory_csv'])}")
    print(f"最终仪表盘: {project_relative_display(outputs['final_dashboard'])}")
    print("其余结果已写入 strategy_cards/、grid_stage_v14/、node_stage_v14/ 子目录")


if __name__ == "__main__":
    main()
