"""把各 run 的 mlflow.db (train_opt 每 run 一个库) 汇总进中心库, 供统一 UI 查看。

- 源: experiments/work_260922_subcenter_ohem/<run>/mlflow.db 们
- 目标: 中心库 (默认 <campaign 根>/mlflow_campaign.db)
- 幂等: 导入的 run 打 src_run_id 标签, 重跑按标签跳过, 可反复执行增量同步
- 保真: params/tags 全量, metrics 逐 key 取完整 step 历史后 log_batch 回放

用法:
python scripts/merge_mlflow_runs.py \
  --src_glob '<experiments 根>/<campaign>/<run>/mlflow.db' \
  --dst_db <experiments 根>/<campaign>/mlflow_campaign.db
"""
import argparse
import glob
import os

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.tracking import MlflowClient

RESERVED_TAGS = {'mlflow.runName'}


def import_one_db(src_db, dst, seen_filter_fmt):
    src_uri = f"sqlite:///{os.path.abspath(src_db)}"
    src = MlflowClient(src_uri)
    n_runs = 0
    for exp in src.search_experiments():
        if exp.name == 'Default':
            continue
        runs = src.search_runs([exp.experiment_id], max_results=5000)
        if not runs:
            continue
        dst_exp = dst.get_experiment_by_name(exp.name)
        if dst_exp is None:
            dst_exp_id = dst.create_experiment(exp.name)
        else:
            dst_exp_id = dst_exp.experiment_id

        for run in runs:
            src_run_id = run.info.run_id
            run_name = run.data.tags.get('mlflow.runName', src_run_id)
            # 判重需含已删除 run (游离 run 合并后即被删, 不含则会重复导入)
            existed = dst.search_runs(
                [dst_exp_id],
                filter_string=seen_filter_fmt.format(src_uri=src_uri, src_run_id=src_run_id),
                max_results=1, run_view_type=ViewType.ALL)
            if existed:
                print(f"  跳过 (已导入): {exp.name}/{run_name}")
                continue

            dst_run = dst.create_run(
                dst_exp_id, run_name=run_name,
                tags={k: v for k, v in run.data.tags.items() if k not in RESERVED_TAGS}
                | {'src_db': src_uri, 'src_run_id': src_run_id})

            params = [Param(k, v) for k, v in run.data.params.items()]
            tags = [RunTag(k, v) for k, v in run.data.tags.items() if k not in RESERVED_TAGS]
            # tags 在 create_run 已带, 这里只补 src 标记之外无需再写; params 分批
            for i in range(0, len(params), 90):
                dst.log_batch(dst_run.info.run_id, params=params[i:i + 90])

            n_metrics = 0
            for key in run.data.metrics.keys():
                hist = src.get_metric_history(src_run_id, key)
                metrics = [Metric(key, m.value, m.timestamp, m.step) for m in hist]
                for i in range(0, len(metrics), 900):
                    dst.log_batch(dst_run.info.run_id, metrics=metrics[i:i + 900])
                n_metrics += len(metrics)
            n_runs += 1
            print(f"  导入: {exp.name}/{run_name} (params={len(params)}, metric 点={n_metrics})")
    return n_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_glob', required=True, help='源 mlflow.db 的 glob')
    parser.add_argument('--dst_db', required=True, help='中心库路径')
    args = parser.parse_args()

    dst_uri = f"sqlite:///{os.path.abspath(args.dst_db)}"
    dst = MlflowClient(dst_uri)
    # 幂等标记: src_db + src_run_id 联合判重 (注意 filter 字符串转义)
    seen_filter_fmt = "tags.src_db = '{src_uri}' and tags.src_run_id = '{src_run_id}'"

    total = 0
    for src_db in sorted(glob.glob(args.src_glob)):
        if os.path.abspath(src_db) == os.path.abspath(args.dst_db):
            continue
        print(f"[源] {src_db}")
        total += import_one_db(src_db, dst, seen_filter_fmt)
    print(f"完成: 新导入 {total} 个 run -> {args.dst_db}")


if __name__ == '__main__':
    main()
