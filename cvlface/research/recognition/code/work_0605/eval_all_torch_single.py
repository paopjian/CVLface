"""
单 checkpoint 评估脚本，由 eval_all_torch_launcher.py 调用。
每个 checkpoint 在独立进程中运行，进程结束后内存自动全部释放。

开关:
  --compile       启用 torch.compile 加速推理
  --timing        启用每个评估器的计时输出
"""
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
import datetime
sys.path.append(os.path.join(root))
import numpy as np
import time

import torch
import pandas as pd
from models import get_model
from aligners import get_aligner
from evaluations import get_evaluator_by_name
from lightning.fabric.loggers import CSVLogger
from pipelines import pipeline_from_name
from general_utils.config_utils import load_config
from evaluations import summary
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset
import lovely_tensors as lt
lt.monkey_patch()


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--eval_config_name', type=str, default='full')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--single_ckpt_path', type=str, required=True)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--project_name', type=str, default="work_0920_eval_all2")
    parser.add_argument('--compile', action='store_true', help='启用 torch.compile 加速推理')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile 模式')
    parser.add_argument('--timing', action='store_true', help='启用每个评估器的计时输出')
    parser.add_argument('--timeout_minutes', type=int, default=60, help='NCCL 通信超时(分钟)')
    args = parser.parse_args()

    path = args.single_ckpt_path
    print(f"评估 checkpoint: {path}")
    print(f"  compile={args.compile}, timing={args.timing}")

    # setup fabric
    csv_logger_dir = os.path.join(root, 'research/recognition/experiments', 'eval_all', args.name)
    os.makedirs(csv_logger_dir, exist_ok=True)
    csv_logger = CSVLogger(root_dir=csv_logger_dir, flush_logs_every_n_steps=1)
    torch.set_float32_matmul_precision('high')
    ddp_strategy = DDPStrategy(timeout=datetime.timedelta(minutes=args.timeout_minutes))
    fabric = Fabric(
        precision=args.precision,
        accelerator="auto",
        strategy=ddp_strategy,
        devices=args.num_gpu,
        loggers=[csv_logger],
    )

    if args.num_gpu == 1:
        fabric.launch()
    print(f"Fabric launched with {args.num_gpu} GPUS and {args.precision}")
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)

    # setup output dir
    runname, save_dir_task, task = get_runname_and_task(path)
    output_dir = os.path.join(root, 'research/recognition/experiments', task, 'eval_' + save_dir_task)
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model_config = load_config(os.path.join(path, 'model.yaml'))
    model_config.start_from = ''
    model_config.freeze = False
    model = get_model(model_config, task)
    model.load_state_dict_from_path(os.path.join(path, 'model.pt'))

    # torch.compile (必须在 fabric.setup 之前，否则 DDP wrapper 导致 graph break 退化为 eager)
    if args.compile:
        print(f"  torch.compile mode: {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
        # rank 0 先触发编译 (dummy forward 写入 inductor cache)，其他 rank 等待后命中 cache
        if fabric.world_size > 1:
            model.cuda()
            if fabric.global_rank == 0:
                print("Rank 0: 触发 torch.compile 编译...")
                compile_start = time.time()
                dummy = torch.randn(1, 3, 112, 112, device='cuda')
                with torch.no_grad():
                    _ = model(dummy)
                del dummy
                torch.cuda.empty_cache()
                print(f"Rank 0: 编译完成, 耗时 {time.time() - compile_start:.1f}s")
            fabric.barrier()

    # load aligner
    aligner_config = load_config(os.path.join(root, 'research/recognition/code/', 'run_v1', f'aligners/configs/none.yaml'))
    aligner = get_aligner(aligner_config)

    # load pipeline
    eval_config = load_config(f'evaluations/configs/{args.eval_config_name}.yaml')
    if args.pipeline_name == 'default':
        full_config_path = os.path.join(path, 'config.yaml')
        assert os.path.isfile(full_config_path), f"config.yaml not found at {full_config_path}"
        pipeline_name = load_config(full_config_path).pipelines.eval_pipeline_name
    else:
        pipeline_name = args.pipeline_name

    # prepare accelerator
    model = fabric.setup(model)
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)

    # make inference pipe
    eval_pipeline = pipeline_from_name(pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataset_color_space='RGB')

    # evaluation callbacks
    evaluators = []
    for name, info in eval_config.per_epoch_evaluations.items():
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type
        eval_batch_size = info.batch_size
        eval_num_workers = info.num_workers
        evaluator = get_evaluator_by_name(
            eval_type=eval_type, name=name, eval_data_path=eval_data_path,
            transform=eval_pipeline.make_test_transform(),
            fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers
        )
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluators.append(evaluator)

    # Evaluation
    print('Evaluation Started')
    all_result = {}
    timing_results = {}
    path_name = os.path.basename(path)
    epoch = get_epoch_num(path_name)
    total_start = time.time()

    for evaluator in evaluators:
        if fabric.local_rank == 0:
            print(f"Evaluating {evaluator.name}")
        eval_start = time.time()
        result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=0, n_images_seen=0)
        eval_elapsed = time.time() - eval_start

        if args.timing:
            timing_results[evaluator.name] = eval_elapsed
            if fabric.local_rank == 0:
                print(f"  [{evaluator.name}] 耗时: {eval_elapsed:.2f}s")

        if fabric.local_rank == 0:
            print(f"{evaluator.name}")
            print(result)
        all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

    # Combined evaluations (合并多源评估)
    combined_config = getattr(eval_config, 'combined_evaluations', None)
    if combined_config:
        print(f'[Rank {fabric.local_rank}] 等待合并评估 (rank 0 计算中)...')
        if fabric.local_rank == 0:
            from evaluations import run_combined_evaluations
            evaluators_dict = {e.name: e for e in evaluators}
            combined_start = time.time()
            combined_result = run_combined_evaluations(evaluators_dict, combined_config)
            all_result.update(combined_result)
            combined_elapsed = time.time() - combined_start
            print(f'合并评估完成，耗时: {combined_elapsed / 60:.2f} mins')
            if args.timing:
                timing_results['combined_evaluations'] = combined_elapsed
        fabric.barrier()

    total_elapsed = time.time() - total_start

    if fabric.local_rank == 0:
        # 计时汇总
        if args.timing:
            print(f"\n{'='*50}")
            print(f"计时汇总:")
            print(f"{'='*50}")
            for name, elapsed in timing_results.items():
                print(f"  {name}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
            print(f"  {'─'*40}")
            print(f"  总耗时: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)")
            print(f"{'='*50}\n")

        print(f'csv输出目录{output_dir}')
        os.makedirs(os.path.join(output_dir, 'result'), exist_ok=True)
        save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
        save_result.to_csv(os.path.join(output_dir, f'result/eval_final.csv'))
        mean, summary_dict = summary(save_result, epoch=epoch, step=0, n_images_seen=0)
        # 将计时信息也写入 summary
        if args.timing:
            summary_dict['timing/total_sec'] = total_elapsed
            for name, elapsed in timing_results.items():
                summary_dict[f'timing/{name}_sec'] = elapsed
        fabric.log_dict(summary_dict)
        summary_result = pd.DataFrame(pd.Series(summary_dict), columns=['val'])
        summary_result.to_csv(os.path.join(output_dir, f'result/eval_summary_final.csv'))

    print(f'Evaluation Finished for {path_name}')
