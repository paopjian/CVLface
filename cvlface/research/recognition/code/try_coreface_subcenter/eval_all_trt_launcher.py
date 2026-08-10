"""
eval_all_3_launcher.py - TRT 多卡评估启动器 (无 fabric/NCCL)

设计:
- 遍历 checkpoint 目录，对每个 ckpt 调用 eval_3_single.py
- 记录结果到 wandb
- 支持断点续评

用法:
python eval_all_3_launcher.py \
  --num_gpu 7 \
  --eval_config_name test_20260605 \
  --ckpt_dir data2/dataset_0605/train_output/s2_body36_0605_06-10_2/checkpoints_every_epoch/ \
  --project_name work_0605_test \
  --name s2_body36_0605_06-10_2_trt
"""
import os
import sys
import subprocess
import argparse
import pandas as pd
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)


def get_epoch_num(path):
    if 'epoch:' in path:
        filename = os.path.basename(path)
        try:
            epoch_part = filename.split('_')[0]
            return int(epoch_part.split(':')[1])
        except (IndexError, ValueError):
            return float('inf')
    if 'adaface' in path:
        filename = os.path.basename(path)
        try:
            epoch_part = filename.split('_')[-1]
            epoch_part = epoch_part.replace('epoch', '')
            return int(epoch_part)
        except (IndexError, ValueError):
            return float(0)
    return float(0)


def find_existing_run(project_name, run_name):
    import wandb
    api = wandb.Api()
    try:
        runs = api.runs(project_name, filters={"display_name": run_name})
    except Exception as e:
        print(f"查询 wandb 失败: {e}")
        return None, set()
    if not runs:
        return None, set()
    target_run = runs[0]
    print(f"找到已有 wandb run: {target_run.name} (id={target_run.id})")
    completed_epochs = set()
    try:
        history = target_run.history(keys=["epoch"], pandas=True)
        if not history.empty and "epoch" in history.columns:
            completed_epochs = set(int(e) for e in history["epoch"].dropna().tolist())
            print(f"已完成 epoch: {sorted(completed_epochs)}")
    except Exception as e:
        print(f"读取 history 失败: {e}")
    return target_run.id, completed_epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=7)
    parser.add_argument('--eval_config_name', type=str, default='test_20260605')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--project_name', type=str, default="work_0605_test")
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp16', 'fp32'],
                        help="TRT engine 精度: fp16(默认, 快) / fp32(更稳, 更接近 PyTorch)")
    parser.add_argument('--timeout_minutes', type=int, default=90)
    parser.add_argument('--max_retries', type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = args.ckpt_dir
    path_list = os.listdir(checkpoint_path)
    full_paths = [os.path.join(checkpoint_path, name) for name in path_list]
    sorted_paths = sorted(full_paths, key=get_epoch_num)

    print(f"共找到 {len(sorted_paths)} 个 checkpoint:")
    for p in sorted_paths:
        print(f"  epoch:{get_epoch_num(p)} - {os.path.basename(p)}")

    # wandb 断点续评
    import wandb
    existing_run_id, completed_epochs = find_existing_run(args.project_name, args.name)

    if existing_run_id and completed_epochs:
        before_count = len(sorted_paths)
        sorted_paths = [p for p in sorted_paths if get_epoch_num(p) not in completed_epochs]
        print(f"\n断点续评: 跳过 {before_count - len(sorted_paths)} 个已完成 epoch")
        if not sorted_paths:
            print("所有 checkpoint 已评估完毕。")
            sys.exit(0)

        wandb_run = wandb.init(
            project=args.project_name, name=args.name,
            id=existing_run_id, resume="must",
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
        )
    else:
        wandb_run = wandb.init(
            project=args.project_name, name=args.name,
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
        )
    print(f"wandb run: {wandb_run.id}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_script = os.path.join(script_dir, 'eval_all_trt_single.py')

    for i, path in enumerate(sorted_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sorted_paths)}] 评估: {os.path.basename(path)}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, single_script,
            '--num_gpu', str(args.num_gpu),
            '--eval_config_name', args.eval_config_name,
            '--ckpt_path', path,
            '--name', args.name,
            '--precision', args.precision,
        ]

        env = os.environ.copy()
        gpu_list = ','.join(str(i) for i in range(args.num_gpu))
        env['CUDA_VISIBLE_DEVICES'] = gpu_list
        env['LD_LIBRARY_PATH'] = f"/root/miniconda3/envs/cvlface/lib:{env.get('LD_LIBRARY_PATH', '')}"

        timeout_sec = args.timeout_minutes * 60
        success = False
        for attempt in range(1, args.max_retries + 1):
            if attempt > 1:
                print(f"  第 {attempt}/{args.max_retries} 次重试...")
            try:
                proc = subprocess.run(cmd, env=env, timeout=timeout_sec)
                if proc.returncode == 0:
                    success = True
                    break
                else:
                    print(f"  失败 (returncode={proc.returncode})")
            except subprocess.TimeoutExpired:
                print(f"  超时 ({args.timeout_minutes}min)")

        if success:
            # 读取结果
            output_dir = os.path.join(script_dir, 'eval3_results', args.name)
            summary_csv = os.path.join(output_dir, f'epoch_{get_epoch_num(path)}_summary.csv')
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv, index_col=0)
                summary_dict = df['val'].to_dict()
                wandb.log(summary_dict)
                print(f"  已记录 wandb: epoch={summary_dict.get('epoch', '?')}")
            else:
                print(f"  警告: 未找到 {summary_csv}")
        else:
            print(f"  警告: {os.path.basename(path)} 最终失败")

    wandb.finish()
    print('\n所有评估完成!')
