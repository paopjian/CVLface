"""wandb run 合并: 把两个 run 的完整历史按 step 轴合并成一个新 run。

用途: s4 主 run (10-30_0, epoch 0-10) 与续训 run (11-01_0, epoch 11-15) 在
wandb 上是两个 run, 合并成连续的 subcenter_ohem_260922_s4_full 便于看整条曲线。
两个源 run 不做任何修改; 新 run 的 config 取源 A。

用法 (需代理):
python scripts/merge_wandb_runs.py \
  --project kejian-zhao-tsinghua-university/work_260922_subcenter_ohem \
  --src_a subcenter_ohem_260922_s4_full_10-30_0 \
  --src_b subcenter_ohem_260922_s4_full_11-01_0 \
  --new_name subcenter_ohem_260922_s4_full
"""
import argparse

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='entity/project')
    parser.add_argument('--src_a', required=True, help='前段 run 名 (config 以它为准)')
    parser.add_argument('--src_b', required=True, help='后段 run 名')
    parser.add_argument('--new_name', required=True)
    args = parser.parse_args()

    api = wandb.Api()

    def find_run(name):
        runs = [r for r in api.runs(args.project) if r.name == name]
        if len(runs) != 1:
            raise ValueError(f"run '{name}' 命中 {len(runs)} 个, 需要唯一")
        return runs[0]

    run_a, run_b = find_run(args.src_a), find_run(args.src_b)

    # 全量拉取重放; 边界 step 撞车时先到先得 (A 优先)。
    # 注意: scan_history 的流式分页走代理会被静默掐断 (实测 B run 采到 0 行/A 行数
    # 随机减半), 必须用 REST history 端点 (samples 给大值即全量)。
    seen, merged = set(), []
    for tag, run in (('A', run_a), ('B', run_b)):
        rows = run.history(pandas=False, samples=200000)
        if len(rows) < 500:
            raise RuntimeError(f"源 {tag} ({run.name}) 仅取到 {len(rows)} 行, 数据不完整, 中止")
        n = 0
        for row in rows:
            step = row.get('_step')
            if step is None:
                continue
            step = int(step)
            if step in seen:
                continue
            metrics = {k: v for k, v in row.items() if not k.startswith('_')}
            if not metrics:
                continue
            seen.add(step)
            merged.append((step, metrics))
            n += 1
        print(f"源 {tag} ({run.name}): 采集 {n} 个 step")

    merged.sort(key=lambda x: x[0])
    print(f"合并总 step 数: {len(merged)} (step {merged[0][0]} .. {merged[-1][0]})")

    entity, project = args.project.split('/', 1)
    wandb_run = wandb.init(entity=entity, project=project, name=args.new_name,
                           config=dict(run_a.config))
    # 源 A 的 _step 是 wandb 自动编号 (0..N), 源 B 是全局步号 (476290+), 直接拼
    # 会在图表 x 轴拉出大空档; 行内自带真实 step/epoch 字段, 合并后顺序重编号。
    for i, (step, metrics) in enumerate(merged):
        wandb.log(metrics, step=i)
    wandb.finish()
    print(f"合并完成 -> {args.project}/{args.new_name} (重编号 step 0..{len(merged)-1})")


if __name__ == '__main__':
    main()
