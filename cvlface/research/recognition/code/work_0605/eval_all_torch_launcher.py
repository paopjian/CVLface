"""
启动器脚本：为每个 checkpoint 独立启动一个子进程进行评估，
彻底避免内存累积问题。
wandb 统一在 launcher 中记录，每个子进程只保存 CSV 结果。

python eval_all_torch_launcher.py \
  --eval_config_name test_20260213 \
  --ckpt_dir /data2/dataset_0213_rec/train_output/ft_ir101_s3_full_02-21_0/checkpoints_every_epoch \
  --project_name work_0213_eval_all_s2 \
  --name s3_0221 \
  --compile --timing

"""
import os
import sys
import signal
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

def get_runname_and_task(ckpt_dir):
    if 'pretrained_models' in ckpt_dir:
        runname = ckpt_dir.split('/')[-1]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = 'pretrained_models'
    else:
        runname = ckpt_dir.split('/')[-3]
        code_task = ckpt_dir.split('/')[-2]
        save_dir_task = ckpt_dir.split('/')[-1]
    return runname, save_dir_task, code_task


def find_existing_run(project_name, run_name):
    """查询 wandb 中是否存在同名 run，如果有则返回该 run 的 id 和已完成的 epoch 集合。"""
    import wandb
    api = wandb.Api()
    try:
        runs = api.runs(project_name, filters={"display_name": run_name})
    except Exception as e:
        print(f"查询 wandb 失败: {e}")
        return None, set()

    if not runs:
        return None, set()

    # 取最新的同名 run
    target_run = runs[0]
    print(f"找到已有 wandb run: {target_run.name} (id={target_run.id}, state={target_run.state})")

    # 从 history 中获取所有已记录的 epoch
    completed_epochs = set()
    try:
        history = target_run.history(keys=["epoch"], pandas=True)
        if not history.empty and "epoch" in history.columns:
            completed_epochs = set(int(e) for e in history["epoch"].dropna().tolist())
            print(f"已有 run 已完成的 epoch: {sorted(completed_epochs)}")
    except Exception as e:
        print(f"读取 history 失败: {e}")

    return target_run.id, completed_epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--eval_config_name', type=str, default='full')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--ckpt_dir', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--project_name', type=str, default="work_0920_eval_all2")
    parser.add_argument('--timeout_minutes', type=int, default=60, help='单个 checkpoint 评估超时时间(分钟)，超时后 kill 并重试')
    parser.add_argument('--max_retries', type=int, default=2, help='超时后最大重试次数')
    parser.add_argument('--compile', action='store_true', help='启用 torch.compile 加速推理')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile 模式')
    parser.add_argument('--timing', action='store_true', help='启用每个评估器的计时输出')
    args = parser.parse_args()

    if args.name == '' or args.project_name == '':
        print('项目名未填写')
        sys.exit(0)

    checkpoint_path = args.ckpt_dir
    path_list = os.listdir(checkpoint_path)
    full_paths = [os.path.join(checkpoint_path, name) for name in path_list]
    sorted_paths = sorted(full_paths, key=get_epoch_num)

    print(f"共找到 {len(sorted_paths)} 个 checkpoint:")
    for p in sorted_paths:
        print(f"  {p}")

    # 查询 wandb 中是否有同名 run，获取已完成的 epoch 集合
    import wandb
    existing_run_id, completed_epochs = find_existing_run(args.project_name, args.name)

    if existing_run_id is not None and completed_epochs:
        # 用已完成的 epoch 集合过滤，只保留未评估的 checkpoint
        before_count = len(sorted_paths)
        sorted_paths = [p for p in sorted_paths if get_epoch_num(p) not in completed_epochs]
        skipped = before_count - len(sorted_paths)
        print(f"\n断点续评: 跳过已完成的 {skipped} 个 checkpoint (epoch in {sorted(completed_epochs)})")
        print(f"剩余 {len(sorted_paths)} 个 checkpoint 待评估:")
        for p in sorted_paths:
            print(f"  {p}")

        if not sorted_paths:
            print("所有 checkpoint 已评估完毕，无需重复运行。")
            sys.exit(0)

        # resume 已有 run
        wandb_run = wandb.init(
            project=args.project_name,
            name=args.name,
            id=existing_run_id,
            resume="must",
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
        )
        print(f"已恢复 wandb run: {existing_run_id}")
    else:
        # 新建 run
        wandb_run = wandb.init(
            project=args.project_name,
            name=args.name,
            dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
        )
        print(f"创建新 wandb run: {wandb_run.id}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_eval_script = os.path.join(script_dir, 'eval_all_torch_single.py')
    # sorted_paths= [sorted_paths[0]]
    for i, path in enumerate(sorted_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sorted_paths)}] 启动子进程评估: {os.path.basename(path)}")
        print(f"{'='*60}")

        cmd = [
            'fabric', 'run',
            f'--devices={args.num_gpu}',
            f'--precision={args.precision}',
            single_eval_script,
            '--eval_config_name', args.eval_config_name,
            '--pipeline_name', args.pipeline_name,
            '--single_ckpt_path', path,
            '--name', args.name,
            '--project_name', args.project_name,
            '--num_gpu', str(args.num_gpu),
            '--precision', args.precision,
        ]
        if args.compile:
            cmd.append('--compile')
            cmd.extend(['--compile_mode', args.compile_mode])
        if args.timing:
            cmd.append('--timing')
        cmd.extend(['--timeout_minutes', str(args.timeout_minutes)])

        env = os.environ.copy()
        env['LIGHTING_TESTING'] = '1'

        timeout_sec = args.timeout_minutes * 60
        success = False
        for attempt in range(1, args.max_retries + 1):
            if attempt > 1:
                print(f"  第 {attempt}/{args.max_retries} 次重试...")
            try:
                proc = subprocess.Popen(cmd, env=env, start_new_session=True)
                proc.wait(timeout=timeout_sec)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                print(f"  超时 ({args.timeout_minutes}min)，正在 kill 进程组...")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                print(f"  已 kill，{'准备重试' if attempt < args.max_retries else '放弃'}")
                returncode = -1
                continue

            if returncode == 0:
                success = True
                break
            else:
                print(f"  评估失败，返回码 {returncode}，{'准备重试' if attempt < args.max_retries else '放弃'}")

        if not success:
            print(f"警告: checkpoint {path} 评估最终失败 (重试 {args.max_retries} 次)")
        else:
            print(f"checkpoint {path} 评估完成")
            # 读取子进程保存的 summary CSV，log 到 wandb
            runname, save_dir_task, task = get_runname_and_task(path)
            output_dir = os.path.join(root, 'research/recognition/experiments', task, 'eval_' + save_dir_task)
            summary_csv = os.path.join(output_dir, 'result', 'eval_summary_final.csv')
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv, index_col=0)
                summary_dict = df['val'].to_dict()
                wandb.log(summary_dict)
                print(f"已记录到 wandb: epoch={summary_dict.get('epoch', '?')}")
            else:
                print(f"警告: 未找到 summary CSV: {summary_csv}")

    wandb.finish()
    print('\n所有评估完成!')
