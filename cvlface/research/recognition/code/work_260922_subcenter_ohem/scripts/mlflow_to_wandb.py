"""把 MLflow 里的 run (含完整指标历史) 回填到 wandb。

用途: wandb 停用期间仅记录在 mlflow 的数据 (如 s4 续训 run、final 评估),
在 wandb 恢复后补传。按 step 重建指标序列, params 写入 config。

用法 (需先 export https_proxy=http://127.0.0.1:7890):
python scripts/mlflow_to_wandb.py \
  --mlflow_db <中心库>/mlflow_campaign.db \
  --experiment eval --run_name final \
  --wandb_project subcenter_ohem_260922
"""
import argparse
import os
from collections import defaultdict

from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_db', required=True)
    parser.add_argument('--experiment', required=True, help='mlflow experiment 名')
    parser.add_argument('--run_name', required=True, help='mlflow 里的 run 名 (tags.mlflow.runName)')
    parser.add_argument('--wandb_project', default='work_260922_subcenter_ohem')
    parser.add_argument('--wandb_run_name', default='', help='默认同 mlflow run 名')
    args = parser.parse_args()

    import wandb
    from mlflow.entities import ViewType

    client = MlflowClient(f"sqlite:///{os.path.abspath(args.mlflow_db)}")
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise ValueError(f"mlflow experiment 不存在: {args.experiment}")
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"tags.mlflow.runName = '{args.run_name}'",
                              max_results=1)
    if not runs:
        raise ValueError(f"run 不存在: {args.run_name}")
    run = runs[0]

    # 逐 key 取历史, 按 step 聚合成 wandb.log 的批次
    by_step = defaultdict(dict)
    n_points = 0
    for key in run.data.metrics.keys():
        for m in client.get_metric_history(run.info.run_id, key):
            by_step[m.step][key] = m.value
            n_points += 1

    run.finish() if hasattr(run, 'finish') else None
    wandb_run = wandb.init(project=args.wandb_project,
                           name=args.wandb_run_name or args.run_name,
                           config=run.data.params,
                           resume='allow')
    for step in sorted(by_step.keys()):
        wandb.log(by_step[step], step=step)
    wandb.finish()
    print(f"回填完成: {args.run_name} -> wandb {args.wandb_project}/"
          f"{args.wandb_run_name or args.run_name} "
          f"(params={len(run.data.params)}, 指标点={n_points}, steps={len(by_step)})")


if __name__ == '__main__':
    main()
