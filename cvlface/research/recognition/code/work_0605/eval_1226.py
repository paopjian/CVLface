"""
启动器脚本：为每个 checkpoint 独立启动一个子进程进行评估，
彻底避免内存累积问题。
wandb 统一在 launcher 中记录，每个子进程只保存 CSV 结果。

python eval_all_2_launcher.py \
  --eval_config_name test_20260213 \
  --ckpt_dir /data2/dataset_0213_rec/train_output/ft_ir101_s3_full_02-21_0/checkpoints_every_epoch \
  --project_name work_0213_eval_all_s2 \
  --name s3_0221

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--eval_config_name', type=str, default='full')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--ckpt_dir', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--project_name', type=str, default="work_0920_eval_all2")
    args = parser.parse_args()

    if args.name == '' or args.project_name == '':
        print('项目名未填写')
        sys.exit(0)

    checkpoint_path = args.ckpt_dir
    # path_list = os.listdir(checkpoint_path)
    # path_list = ['s3_0925','adaface_ir101_webface12m','s3_1226_14']
    full_paths = [os.path.join(checkpoint_path, name) for name in path_list]
    sorted_paths = sorted(full_paths, key=get_epoch_num)
    is_single_ckpt = len(sorted_paths) == 1

    print(f"共找到 {len(sorted_paths)} 个 checkpoint:")
    for p in sorted_paths:
        print(f"  {p}")

    # 初始化 wandb（单个 run 记录所有 checkpoint）
    import wandb
    wandb_run = wandb.init(
        project=args.project_name,
        name=args.name,
        dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_eval_script = os.path.join(script_dir, 'eval_all_2_single.py')
    # sorted_paths= [sorted_paths[0]]
    for i, path in enumerate(sorted_paths):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sorted_paths)}] 启动子进程评估: {os.path.basename(path)}")
        print(f"{'='*60}")

        cmd = [
            'lightning', 'run', 'model',
            f'--strategy=ddp',
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

        env = os.environ.copy()
        env['LIGHTING_TESTING'] = '1'

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"警告: checkpoint {path} 评估失败，返回码 {result.returncode}")
        else:
            print(f"checkpoint {path} 评估完成")
            # 读取子进程保存的 summary CSV，log 到 wandb
            runname, save_dir_task, task = get_runname_and_task(path)
            output_dir = os.path.join(root, 'research/recognition/experiments', task, 'eval_' + save_dir_task)
            summary_csv = os.path.join(output_dir, 'result', 'eval_summary_final.csv')
            if os.path.exists(summary_csv):
                df = pd.read_csv(summary_csv, index_col=0)
                summary_dict = df['val'].to_dict()
                if is_single_ckpt:
                    for epoch_i in range(21):
                        summary_dict['epoch'] = epoch_i
                        wandb.log(summary_dict)
                    print(f"已记录基准线到 wandb: epoch=0~20")
                else:
                    wandb.log(summary_dict)
                    print(f"已记录到 wandb: epoch={summary_dict.get('epoch', '?')}")
            else:
                print(f"警告: 未找到 summary CSV: {summary_csv}")

    wandb.finish()
    print('\n所有评估完成!')
