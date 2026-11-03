"""把 eval_all_trt_single 产出的 summary CSV 回填进 MLflow (事后记录)。

场景: 评估已跑完/正在跑 (launcher 未带 --mlflow_db), 结果 CSV 已落地, 需要
事后记入 MLflow。也可把训练 run 的最终测试评估以 final/ 前缀并入其训练 run。

用法 (把 final 评估记进 s4 训练 run):
python scripts/log_eval_to_mlflow.py \
  --summary_csv eval3_results/final/epoch_10_summary.csv \
  --mlflow_db <run_dir>/mlflow.db \
  --experiment work_260922_subcenter_ohem \
  --run_name subcenter_ohem_260922_s4_full_10-30_0 \
  --prefix final/ --step 10
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', required=True, help='eval_all_trt_single 产出的 *_summary.csv')
    parser.add_argument('--mlflow_db', required=True, help='sqlite 库路径, 如 <run_dir>/mlflow.db')
    parser.add_argument('--experiment', default='eval', help='MLflow experiment 名')
    parser.add_argument('--run_name', default='', help='并入已有 run 的名字 (按 experiment 内搜索); 留空则新建 run')
    parser.add_argument('--prefix', default='', help='指标键前缀, 如 final/')
    parser.add_argument('--step', type=int, default=0, help='指标 step (一般 = epoch)')
    args = parser.parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    df = pd.read_csv(args.summary_csv, index_col=0)
    summary = df['val'].to_dict()
    metrics = {f"{args.prefix}{k.replace('@', '_at_')}": float(v)
               for k, v in summary.items() if isinstance(v, (int, float))}
    if not metrics:
        raise ValueError(f"CSV 里没有可记录的数值指标: {args.summary_csv}")

    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(args.mlflow_db)}")
    mlflow.set_experiment(args.experiment)

    run_id = None
    if args.run_name:
        client = MlflowClient()
        exp = client.get_experiment_by_name(args.experiment)
        if exp is not None:
            runs = client.search_runs([exp.experiment_id],
                                      filter_string=f"tags.mlflow.runName = '{args.run_name}'",
                                      max_results=1)
            if runs:
                run_id = runs[0].info.run_id
                print(f"并入已有 run: {args.run_name} (id={run_id})")

    with mlflow.start_run(**({'run_id': run_id} if run_id else {'run_name': args.run_name or 'eval_backfill'})):
        mlflow.log_metrics(metrics, step=args.step)
        mlflow.set_tag('backfill_source', os.path.abspath(args.summary_csv))
        active = mlflow.active_run()
        print(f"已记录 {len(metrics)} 个指标 (step={args.step}) -> {args.mlflow_db} "
              f"(run={active.info.run_name})")
        for k in sorted(metrics)[:5]:
            print(f"  {k} = {metrics[k]}")


if __name__ == '__main__':
    main()
