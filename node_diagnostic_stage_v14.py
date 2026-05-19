import argparse
import os

from visualization_stage_v14 import build_context, render_node_stage


def main():
    parser = argparse.ArgumentParser(description="生成 v14 节点级可视化诊断图")
    parser.add_argument("--output-dir", default=os.path.join("result", "node_stage_v14"), help="节点级图片输出目录")
    parser.add_argument("--analysis-dir", default="result", help="轨迹与户型输入目录")
    parser.add_argument("--floor-plan", default=None, help="户型图路径，默认使用 analysis-dir/floor_plan.png")
    parser.add_argument("--trajectory-csv", default=None, help="轨迹 CSV 路径，默认使用 analysis-dir/trajectory.csv")
    parser.add_argument("--ticks", type=int, default=1000, help="缺少轨迹时自动补跑模拟的 tick 数")
    parser.add_argument("--random-seed", type=int, default=7, help="自动补跑模拟的随机种子")
    args = parser.parse_args()

    ctx = build_context(
        output_dir=args.analysis_dir,
        floor_plan_path=args.floor_plan,
        trajectory_csv=args.trajectory_csv,
        total_ticks=args.ticks,
        random_seed=args.random_seed,
    )
    outputs = render_node_stage(ctx, args.output_dir)
    print("[node_diagnostic_stage_v14] 输出完成:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
