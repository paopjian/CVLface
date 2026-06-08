"""
启动器脚本：指定 `ckpt_dir` 后，直接评估其中的 `model.pt`。
wandb 统一在 launcher 中记录，子进程只保存 CSV 结果。

python eval_one_folder.py \
  --eval_config_name test_20260320 \
  --ckpt_dir /root/zhaokj/CVLface/cvlface/pretrained_models/recognition/s3_1226_14 \
  --project_name work_0320_eval \
  --name s3_1226

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

def get_runname_and_task(ckpt_dir):
    if 'pretrained_models' in ckpt_dir:
        runname = os.path.basename(os.path.dirname(ckpt_dir)) if ckpt_dir.endswith('.pt') else ckpt_dir.split('/')[-1]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = 'pretrained_models'
    else:
        normalized_path = os.path.dirname(ckpt_dir) if ckpt_dir.endswith('.pt') else ckpt_dir
        runname = normalized_path.split('/')[-3]
        code_task = normalized_path.split('/')[-2]
        save_dir_task = normalized_path.split('/')[-1]
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
    if not os.path.exists(checkpoint_path):
        print(f'未找到 checkpoint: {checkpoint_path}')
        sys.exit(1)

    print("将评估 checkpoint:")
    print(f"  {checkpoint_path}")

    # 初始化 wandb（单个 run 记录单个 checkpoint）
    import wandb
    wandb_run = wandb.init(
        project=args.project_name,
        name=args.name,
        dir=os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name),
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    single_eval_script = os.path.join(script_dir, 'eval_all_2_single.py')
    print(f"\n{'='*60}")
    print(f"启动子进程评估: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")

    cmd = [
        'lightning', 'run', 'model',
        f'--strategy=ddp',
        f'--devices={args.num_gpu}',
        f'--precision={args.precision}',
        single_eval_script,
        '--eval_config_name', args.eval_config_name,
        '--pipeline_name', args.pipeline_name,
        '--single_ckpt_path', checkpoint_path,
        '--name', args.name,
        '--project_name', args.project_name,
        '--num_gpu', str(args.num_gpu),
        '--precision', args.precision,
    ]

    env = os.environ.copy()
    env['LIGHTING_TESTING'] = '1'

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"警告: checkpoint {checkpoint_path} 评估失败，返回码 {result.returncode}")
    else:
        print(f"checkpoint {checkpoint_path} 评估完成")
        runname, save_dir_task, task = get_runname_and_task(checkpoint_path)
        output_dir = os.path.join(root, 'research/recognition/experiments', task, 'eval_' + save_dir_task)
        summary_csv = os.path.join(output_dir, 'result', 'eval_summary_final.csv')
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv, index_col=0)
            summary_dict = df['val'].to_dict()
            for epoch_i in range(20):
                summary_dict['epoch'] = epoch_i
                wandb.log(summary_dict)
            print(f"已记录到 wandb: project={args.project_name}, name={args.name}")
        else:
            print(f"警告: 未找到 summary CSV: {summary_csv}")

    wandb.finish()
    print('\n所有评估完成!')
