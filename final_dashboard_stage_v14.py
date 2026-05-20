import argparse
import os

from final_dashboard import render_final_dashboard
from visualization_stage_v14 import build_context


def main():
    parser = argparse.ArgumentParser(description="生成 v14 最终综合仪表盘")
    parser.add_argument("--output-path", default=os.path.join("result", "final_stage_v14", "final_dashboard_v14.png"), help="最终仪表盘输出路径")
    parser.add_argument("--analysis-dir", default="result", help="轨迹与户型输入目录")
    parser.add_argument("--floor-plan", default=None, help="户型图路径，默认使用 analysis-dir/floor_plan.png")
    parser.add_argument("--trajectory-csv", default=None, help="轨迹 CSV 路径，默认使用 analysis-dir/trajectory.csv")
    parser.add_argument("--ticks", type=int, default=1440, help="缺少轨迹时自动补跑模拟的 tick 数")
    parser.add_argument("--random-seed", type=int, default=7, help="自动补跑模拟的随机种子")
    args = parser.parse_args()

    ctx = build_context(
        output_dir=args.analysis_dir,
        floor_plan_path=args.floor_plan,
        trajectory_csv=args.trajectory_csv,
        total_ticks=args.ticks,
        random_seed=args.random_seed,
    )
    output_path = render_final_dashboard(ctx, args.output_path)
    print(f"[final_dashboard_stage_v14] 输出完成: {output_path}")


if __name__ == "__main__":
    main()
