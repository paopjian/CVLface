"""
fix_wandb_summary.py - 将 history 中的 ijbc/001 数据移到 summary 中

用法:
python fix_wandb_summary.py \
    --project_name work_0605_test \
    --run_id irlb4py2
"""
import argparse
from wandb import Api


def fix_summary(project_name, run_id):
    """从 history 中提取最后一个非空的 ijbc/001_* 值，更新到 summary"""
    api = Api()
    run = api.run(f"{project_name}/{run_id}")

    print(f"Run: {run.name} (id={run.id})")
    print(f"URL: {run.url}")

    # 读取完整 history
    print("\n读取 history...")
    history = run.history(samples=10000)

    # 找出所有 ijbc/001 相关的列
    ijbc_001_keys = [col for col in history.columns if col.startswith('ijbc/001_')]

    if not ijbc_001_keys:
        print("错误: history 中没有找到 ijbc/001_* 数据")
        return

    print(f"找到 {len(ijbc_001_keys)} 个 ijbc/001 指标:")
    for key in sorted(ijbc_001_keys):
        print(f"  {key}")

    # 提取每个指标的最后一个非空值
    summary_update = {}
    for key in ijbc_001_keys:
        # 从后往前找第一个非空值
        values = history[key].dropna()
        if len(values) > 0:
            last_value = values.iloc[-1]
            # 将 ijbc/001_xxx 转换为 summary/ijbc_001_xxx 格式
            summary_key = key.replace('ijbc/001_', 'summary/ijbc_001_')
            summary_update[summary_key] = last_value
            print(f"  {key} -> {summary_key}: {last_value:.6f}")
        else:
            print(f"  {key}: 无有效数据")

    if not summary_update:
        print("\n错误: 所有 ijbc/001 指标都没有有效值")
        return

    # 更新 summary
    print(f"\n更新 summary (共 {len(summary_update)} 个指标)...")
    run.summary.update(summary_update)

    print("✓ 完成! 请刷新 wandb 页面查看更新后的 summary")
    print(f"  {run.url}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True,
                        help='wandb project 名称 (格式: entity/project 或 project)')
    parser.add_argument('--run_id', type=str, required=True,
                        help='wandb run id')
    args = parser.parse_args()

    # 处理 project_name 格式
    if '/' not in args.project_name:
        # 如果没有提供 entity，需要从 API 获取
        api = Api()
        # 尝试直接用 run_id 获取，这样可以自动推断 entity
        try:
            test_run = api.run(f"{args.project_name}/{args.run_id}")
            project_path = f"{test_run.entity}/{args.project_name}"
        except:
            print(f"错误: 无法找到 run，请使用完整的 project 路径: entity/project")
            print(f"示例: --project_name kejian-zhao-tsinghua-university/work_0605_test")
            exit(1)
    else:
        project_path = args.project_name

    fix_summary(project_path, args.run_id)
