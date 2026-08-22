import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys

sys.path.append(os.path.join(root))
import numpy as np

import pandas as pd
import torch
import config
from config import Config
from models import get_model
from classifiers import get_classifier
from aligners import get_aligner
from losses import get_margin_loss
from dataset import get_train_dataset, visualize_dataset, set_epoch
from evaluations import get_evaluator_by_name
from general_utils import random_utils, os_utils
from optims.optims import make_optimizer
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from optims.lr_scheduler import make_scheduler, scheduler_step, get_last_lr
from pipelines import pipeline_from_config, pipeline_from_name
import omegaconf
import lovely_tensors as lt
lt.monkey_patch()
from tqdm import tqdm
from evaluations import IsBestTracker, summary
from evaluations import run_combined_evaluations_distributed
import time
import mlflow
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
import datetime
from pefts import apply_peft
from general_utils.dist_utils import verify_ddp_weights_equal
from functools import partial
from fabric.fabric import setup_dataloader_from_dataset
import threading, queue
import gc

from external_torch_eval import run_external_torch_eval

# MLflow 异步写入线程 (避免 SQLite I/O 阻塞训练循环)
_mlflow_queue = queue.Queue()

def _mlflow_writer():
    while True:
        item = _mlflow_queue.get()
        if item is None:
            break
        metrics, step = item
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

_mlflow_thread = threading.Thread(target=_mlflow_writer, daemon=True)
_mlflow_thread.start()


def get_norm(module):
    """计算模块梯度的 L2 norm (GPU 上累积, 只一次 sync)."""
    params = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return 0.0
    # 利用 clip_grad_norm_ 的 fused multi-tensor kernel, max_norm=inf 不做裁剪
    return torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf')).item()
    
def compute_update_ratio(module, param_cache):
    if module is None or len(param_cache) == 0:
        return 0.0
    device = next(module.parameters()).device
    total_delta_sq = torch.zeros(1, device=device)
    total_param_sq = torch.zeros(1, device=device)
    for name, p in module.named_parameters():
        if name in param_cache:
            delta = p.data - param_cache[name]
            total_delta_sq += delta.norm(2) ** 2
            total_param_sq += param_cache[name].norm(2) ** 2
    if total_param_sq.item() > 0:
        return (total_delta_sq.sqrt() / total_param_sq.sqrt()).item()
    else:
        return 0.0

def log_classifier(classifier, log_dict):
    if classifier is not None and hasattr(classifier, 'partial_fc') and hasattr(classifier.partial_fc, 'batch_mean') and hasattr(classifier.partial_fc, 'batch_std'): 
        try:
            mean_t = classifier.partial_fc.batch_mean.detach().float()
            std_t = classifier.partial_fc.batch_std.detach().float()
            
            # 确保张量是标量
            if mean_t.numel() > 1:
                mean_t = mean_t.mean()
            if std_t.numel() > 1:
                std_t = std_t.mean()
                
            # 直接记录全局的batch_mean和batch_std
            log_dict['train/adaface_batch_mean'] = float(mean_t.item())
            log_dict['train/adaface_batch_std'] = float(std_t.item())
            
        except Exception as e:
            print(f"Error occurred while processing classifier statistics: {e}")
    return log_dict

# 添加用于早停的类
class EarlyStoppingMonitor:
    def __init__(self, patience=5):
        self.patience = patience  # 连续多少个epoch没有改进后停止
        self.best_metric = -float('inf')  # 对于准确率等指标，初始值设为负无穷
        self.no_improvement_count = 0
    
    def update(self, metric):
        """更新指标并检查是否需要早停"""
        if metric > self.best_metric:
            self.best_metric = metric
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            # 如果连续patience个epoch没有改进，返回True表示需要早停
            print(f'EarlyStoppingMonitor: metric={metric:.4f}, best_metric={self.best_metric:.4f}, no_improvement_count={self.no_improvement_count}/{self.patience}')
            return self.no_improvement_count >= self.patience

class MetricImprovementMonitor:
    """监控特定指标是否在连续N个epoch内有足够的改进，否则触发早停"""
    def __init__(self, metric_name, patience=5, min_improvement=0.01, mode='max', relative=True):
        """
        Args:
            metric_name: 要监控的指标名称
            patience: 连续多少个epoch没有足够改进后触发早停
            min_improvement: 最小改进量（relative=True时为比例，relative=False时为绝对值）
            mode: 'max' 表示指标越大越好, 'min' 表示指标越小越好
            relative: True表示min_improvement是相对比例（如0.01=1%），False表示绝对值
        """
        self.metric_name = metric_name
        self.patience = patience
        self.min_improvement = min_improvement
        self.mode = mode
        self.relative = relative
        self.best_value = None
        self.no_improvement_count = 0

    def update(self, value):
        """
        更新指标值并检查是否需要早停
        Returns:
            should_stop: 是否应该停止训练
        """
        if self.best_value is None:
            self.best_value = value
            self.no_improvement_count = 0
            print(f"[{self.metric_name}] 初始值: {value:.4f}")
            return False

        # 计算改进量
        if self.relative:
            # 相对改进：相对于best的比例
            if self.mode == 'max':
                improved = value > self.best_value * (1 + self.min_improvement)
            else:
                improved = value < self.best_value * (1 - self.min_improvement)
        else:
            # 绝对改进
            if self.mode == 'max':
                improved = value > self.best_value + self.min_improvement
            else:
                improved = value < self.best_value - self.min_improvement

        if improved:
            old_best = self.best_value
            self.best_value = value
            self.no_improvement_count = 0
            print(f"[{self.metric_name}] 改进: {old_best:.4f} -> {value:.4f}")
        else:
            self.no_improvement_count += 1
            print(f"[{self.metric_name}] 无足够改进: 当前={value:.4f}, 最佳={self.best_value:.4f} ({self.no_improvement_count}/{self.patience})")

        if self.no_improvement_count >= self.patience:
            print(f"[{self.metric_name}] 早停触发: 连续{self.patience}个epoch无足够改进")
            return True
        return False

    def check(self, all_result):
        """从评估结果字典中提取指标并更新"""
        if self.metric_name not in all_result:
            print(f"Warning: Metric '{self.metric_name}' not found in results, keys: {list(all_result.keys())}")
            return False
        return self.update(all_result[self.metric_name])


def broadcast_should_stop(fabric, should_stop):
    """广播早停信号到所有 rank，确保 DDP 同步退出"""
    stop_tensor = torch.tensor(int(should_stop), device=fabric.device)
    fabric.broadcast(stop_tensor, src=0)
    return stop_tensor.item()

if __name__ == '__main__':
    cfg: Config = config.init(root)
    # print(f"cfg:{cfg}")
    torch.set_float32_matmul_precision(cfg.trainers.float32_matmul_precision)
    torch.backends.cudnn.benchmark = True
    # print('matmul precision', cfg.trainers.float32_matmul_precision)
    # print('precision', cfg.trainers.precision)

    random_utils.setup_seed(seed=cfg.trainers.seed, cuda_deterministic=False)

    loggers = []
    csv_logger = CSVLogger(root_dir=cfg.trainers.output_dir, flush_logs_every_n_steps=1)
    loggers.append(csv_logger)
    if cfg.trainers.using_wandb:
        wandb_logger = WandbLogger(project=cfg.trainers.task, save_dir=cfg.trainers.output_dir,
                                   name=os.path.basename(cfg.trainers.output_dir),
                                   log_model=False)
        loggers.append(wandb_logger)

    # grad_max_norm?
    nccl_timeout_min = getattr(cfg.trainers, 'timeout_minutes', 120)
    ddp_strategy = DDPStrategy(timeout=datetime.timedelta(minutes=nccl_timeout_min))
    fabric = Fabric(precision=cfg.trainers.precision,
                    loggers=loggers,
                    accelerator="auto",
                    strategy=ddp_strategy,
                    devices=cfg.trainers.num_gpu)
    
    
    fabric.seed_everything(cfg.trainers.seed)
    if cfg.trainers.num_gpu == 1:
        fabric.launch()
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=cfg.trainers.seed)

    cfg.trainers.local_rank = fabric.local_rank
    cfg.trainers.world_size = fabric.world_size
    # Bind each worker before constructing CUDA-backed objects. This avoids
    # creating idle CUDA contexts on GPU 0 in non-zero ranks.
    if torch.cuda.is_available():
        torch.cuda.set_device(fabric.local_rank)
    # print = fabric.print

    # MLflow: 仅 rank 0 启动 run 并记录超参数 (离线 SQLite, 无需启动服务)
    if fabric.local_rank == 0:
        mlflow_db_path = os.path.join(cfg.trainers.output_dir, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        mlflow.set_experiment(cfg.trainers.task)
        mlflow_run = mlflow.start_run(run_name=os.path.basename(cfg.trainers.output_dir))
        mlflow_params = {
            "model": cfg.models.yaml_path,
            "loss": cfg.losses.name if hasattr(cfg.losses, 'name') else str(cfg.losses),
            "optimizer": cfg.optims.optimizer,
            "lr": cfg.optims.lr,
            "batch_size": cfg.trainers.batch_size,
            "num_gpu": cfg.trainers.num_gpu,
            "precision": cfg.trainers.precision,
            "dataset": cfg.dataset.rec if hasattr(cfg.dataset, 'rec') else str(cfg.dataset),
            "peft": cfg.pefts.method if hasattr(cfg.pefts, 'method') else "none",
            "num_epoch": cfg.optims.num_epoch,
        }
        mlflow.log_params(mlflow_params)

    # get model
    model = get_model(cfg.models, cfg.trainers.task)

    train_transform = model.make_train_transform()
    test_transform = model.make_test_transform()

    # get dataloader
    dataset, label_mapping = get_train_dataset(cfg.dataset, train_transform, cfg.data_augs, local_rank=cfg.trainers.local_rank)
    dataloader = fabric.setup_dataloader_from_dataset(dataset=dataset,
                                                      is_train=True,
                                                      batch_size=cfg.trainers.batch_size,
                                                      num_workers=cfg.trainers.num_workers)
    cfg.trainers.total_batch_size = cfg.trainers.batch_size * cfg.trainers.world_size
    batch_length = len(dataloader.dataset) // cfg.trainers.total_batch_size
    batch_length = batch_length if cfg.trainers.limit_num_batch <= 0 else cfg.trainers.limit_num_batch
    cfg.trainers.warmup_step = batch_length * cfg.optims.warmup_epoch
    cfg.trainers.total_step = batch_length * cfg.optims.num_epoch
    if fabric.local_rank == 0:
        visualize_dataset(dataloader, os.path.join(cfg.trainers.output_dir, 'train_data.png'))


    # get classifier
    margin_loss_fn = get_margin_loss(cfg.losses)

    extra_classes = 0
    classifier = get_classifier(cfg.classifiers,
                                margin_loss_fn=margin_loss_fn,
                                model_cfg=cfg.models,
                                num_classes=cfg.dataset.num_classes+extra_classes,
                                rank=fabric.local_rank,
                                world_size=fabric.world_size)

    # get aligner
    aligner = get_aligner(cfg.aligners)

    # apply peft if needed
    model, classifier = apply_peft(cfg.pefts, model=model, classifier=classifier, data_cfg=cfg.dataset, label_mapping=label_mapping)
    # channels_last 加速卷积
    model = model.to(memory_format=torch.channels_last)
    # torch.compile 编译加速 (在DDP包装前compile, PyTorch 2.12推荐)
    model = torch.compile(model, dynamic=False)

    # get optimizer
    optimizer = make_optimizer(cfg, model, classifier, aligner)
    lr_scheduler = make_scheduler(cfg, optimizer)

    # prepare accelerator
    if model.has_trainable_params():
        model, optimizer = fabric.setup(model, optimizer)
    else:
        model = model.to(fabric.device)
        dummy_model = torch.nn.Linear(1, 1).to(fabric.device)
        dummy_model, optimizer = fabric.setup(dummy_model, optimizer)
    if classifier is not None:
        if classifier.apply_ddp:
            classifier = fabric.setup(classifier)
        else:
            classifier = classifier.to(fabric.device)  # no ddp as it divides fc into multiple GPUs
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)
    elif aligner is not None:
        aligner = aligner.to(fabric.device)


    verify_ddp_weights_equal(model)
    if classifier is not None:
        verify_ddp_weights_equal(classifier)

    # make train pipe (after accelerator setup)
    train_pipeline = pipeline_from_config(cfg.pipelines, model, classifier, aligner, optimizer, lr_scheduler)
    train_pipeline.integrity_check(dataloader.dataset)
    
 
    # make inference pipe (after accelerator setup)
    eval_pipeline = pipeline_from_name(cfg.pipelines.eval_pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataloader.dataset.color_space)

    # External evaluation constructs evaluators in the child process.
    evaluators = []
    if not cfg.trainers.external_eval:
        for name, info in cfg.evaluations.per_epoch_evaluations.items():
            eval_data_path = os.path.join(cfg.evaluations.data_root, info.path)
            eval_type = info.evaluation_type
            eval_batch_size = info.batch_size * 4
            eval_num_workers = info.num_workers
            evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                              transform=eval_pipeline.make_test_transform(),
                                              fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers)
            evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
            evaluator.config = info
            evaluators.append(evaluator)

    # copy project files
    if fabric.local_rank == 0:
        code_dir = os.path.dirname(os.path.abspath(__file__))
        os_utils.copy_project_files(code_dir, cfg.trainers.output_dir)
        omegaconf.OmegaConf.save(cfg, os.path.join(cfg.trainers.output_dir, 'config.yaml'))
        os.makedirs(os.path.join(cfg.trainers.output_dir, 'lightning_logs'), exist_ok=True)

    # train
    step = train_pipeline.step
    n_images_seen = train_pipeline.n_images_seen
    n_epochs = cfg.optims.num_epoch - train_pipeline.start_epoch
    print(f"start at {train_pipeline.start_epoch} and training for {n_epochs} epochs")
    is_best_tracker = IsBestTracker(fabric)
    
    # 初始化早停监视器
    early_stopping = EarlyStoppingMonitor(patience=10)

    # 基于改进幅度的早停监控器
    improvement_monitors = [
        # work/tpir_at_far_1e-10: 10个epoch没有提升1%则早停 (mode='max', 越大越好)
        MetricImprovementMonitor(metric_name='work_0605_3t/tpir_at_far_1e-10', patience=10, min_improvement=0.005, mode='max', relative=True),
    ]
    # train/mean_loss: 5个epoch没有降低0.1则早停 (mode='min', 绝对值)
    loss_improvement_monitor = MetricImprovementMonitor(metric_name='train/mean_loss', patience=10, min_improvement=0.1, mode='min', relative=False)
    
    
    tic = time.time()
    epoch = train_pipeline.start_epoch
    for epoch in range(train_pipeline.start_epoch, cfg.optims.num_epoch):
        epoch_start_time = time.time()
        epoch_loss_sum = torch.zeros(1, device=fabric.device, dtype=torch.float32)
        epoch_loss_count = 0
        param_cache_backbone = {}
        param_cache_classifier = {}
        
        train_pipeline.train()
        
        set_epoch(dataloader, epoch, cfg)
        batch_length = len(dataloader) if cfg.trainers.limit_num_batch <= 0 else cfg.trainers.limit_num_batch
        pbar = tqdm(total=batch_length, disable=fabric.local_rank != 0)
        if cfg.trainers.local_rank == 0:
            print('\nRun Name', os.path.basename(cfg.trainers.output_dir))
        for batch_idx, batch in enumerate(dataloader):

            if cfg.trainers.limit_num_batch > 0 and batch_idx >= cfg.trainers.limit_num_batch:
                break

            if cfg.trainers.mock_lr_run:
                loss = 0
            else:
                is_accumulating = batch_idx % cfg.trainers.gradient_acc != 0
                with fabric.no_backward_sync(model if model.has_trainable_params() else dummy_model,
                                             enabled=is_accumulating):
                    with fabric.autocast():
                        loss = train_pipeline(batch)
                        fabric.backward(loss)
                if not is_accumulating:
                    if batch_idx % 200 == 0:
                        # ========== 分别计算 backbone 和 classifier 的 grad_norm ==========
                        grad_norm_backbone = get_norm(model)
                        grad_norm_classifier = get_norm(classifier) if classifier is not None else 0.0

                        # ========== 缓存参数（分开缓存） ==========
                        param_cache_backbone = {
                            name: p.data.clone()
                            for name, p in model.named_parameters() if p.requires_grad
                        }
                        param_cache_classifier = {}
                        if classifier is not None:
                            param_cache_classifier = {
                                name: p.data.clone()
                                for name, p in classifier.named_parameters() if p.requires_grad
                            }
                    # 源代码的梯度裁剪
                    fabric.clip_gradients(model, optimizer, max_norm=cfg.optims.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            scheduler_step(lr_scheduler, step)
            last_lr = get_last_lr(optimizer)
            
            # 记录loss (GPU上累积, 不触发sync)
            epoch_loss_sum.add_(loss.detach().float())
            epoch_loss_count += 1
            
            n_images_seen += cfg.trainers.total_batch_size
            step += 1

            if batch_idx % 200 == 0:
                # ========== 分别计算 update_ratio ==========
                update_ratio_backbone = compute_update_ratio(model, param_cache_backbone)
                update_ratio_classifier = compute_update_ratio(classifier, param_cache_classifier) if classifier is not None else 0.0

                log_dict = {}
                log_dict['epoch'] = epoch
                log_dict['step'] = step
                log_dict['n_images_seen'] = n_images_seen
                log_dict['train/loss'] = loss
                log_dict['train/lr'] = last_lr
                log_dict['trainer/global_step'] = step
                log_dict['trainer/epoch'] = epoch
                
                log_dict['train/mean_loss'] = (epoch_loss_sum / epoch_loss_count).item() if epoch_loss_count > 0 else 0.0
                log_dict['train/grad_norm_backbone']= grad_norm_backbone
                log_dict['train/grad_norm_classifier']= grad_norm_classifier
                log_dict['train/update_ratio_backbone']= update_ratio_backbone
                log_dict['train/update_ratio_classifier']= update_ratio_classifier
                
                log_dict = log_classifier(classifier, log_dict)

                fabric.log_dict(log_dict, step=step)
                # MLflow 记录训练指标 (仅 rank 0, 异步写入)
                if fabric.local_rank == 0:
                    mlflow_metrics = {}
                    for k, v in log_dict.items():
                        if isinstance(v, torch.Tensor):
                            mlflow_metrics[k] = v.detach().cpu().item()
                        elif isinstance(v, (int, float)):
                            mlflow_metrics[k] = float(v)
                    _mlflow_queue.put((mlflow_metrics, step))

            speed = cfg.trainers.batch_size / (time.time() - tic)
            speed_total = speed * fabric.world_size
            if batch_idx % 10 == 0:
                # 每 10 batch 更新 pbar, 避免每 batch 都 format tensor (.item() sync)
                loss_val = loss.item()
                pbar.set_description(f"Epoch {epoch} | Step {step} | Batch {batch_idx} | Speed {speed_total:.0f} | LR {last_lr:.5f} | Loss {loss_val:.4f}")
            pbar.update(1)
            tic = time.time()
        
        # 每个epoch保存模型
        fabric.barrier()
        save_dir = os.path.join(
            cfg.dataset.model_save_dir,
            os.path.basename(cfg.trainers.output_dir),
            'checkpoints_every_epoch',
            f'epoch:{epoch}',
        )
        train_pipeline.save_pipelines_and_configs(save_dir, fabric, train_pipeline, cfg, epoch, step, n_images_seen)
        fabric.barrier()


        # 计算epoch平均损失
        avg_epoch_loss = (epoch_loss_sum / epoch_loss_count).item() if epoch_loss_count > 0 else float('inf')


        # validation (skip when only classifier is training — model embeddings unchanged)
        if cfg.evaluations.eval_every_n_epochs > 0 and model.has_trainable_params():
            print('Evaluation Started')
            eval_start_time = time.time()
            should_evaluate = (
                epoch % cfg.evaluations.eval_every_n_epochs == 0
                or epoch == (cfg.optims.num_epoch - 1)
                or epoch + 1 in cfg.optims.lr_milestones
            )
            all_result = {}
            if should_evaluate and cfg.trainers.external_eval:
                # Release epoch-local tensors before the child loads a second model copy.
                batch = None
                loss = None
                param_cache_backbone.clear()
                param_cache_classifier.clear()
                gc.collect()
                with torch.cuda.device(fabric.device):
                    torch.cuda.empty_cache()
                fabric.barrier()
                all_result = run_external_torch_eval(
                    fabric=fabric,
                    cfg=cfg,
                    checkpoint_dir=save_dir,
                    epoch=epoch,
                )
            elif should_evaluate:
                for evaluator in evaluators:
                    print(f"Evaluating {evaluator.name}")
                    result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=step, n_images_seen=n_images_seen)
                    all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})
            eval_time = (time.time() - eval_start_time) / 60
            if fabric.local_rank == 0:
                print(f'Evaluation Time: {eval_time:.2f} mins')
            

            # Combined evaluations (合并多源评估)
            combined_config = None if cfg.trainers.external_eval else getattr(cfg.evaluations, 'combined_evaluations', None)
            if combined_config:
                evaluators_dict = {e.name: e for e in evaluators}
                combined_start = time.time()
                combined_result = run_combined_evaluations_distributed(
                    fabric, evaluators_dict, combined_config, epoch, step, n_images_seen
                )
                all_result.update(combined_result)
                if fabric.local_rank == 0:
                    print(f'合并评估完成，耗时: {(time.time() - combined_start) / 60:.2f} mins')

            if fabric.local_rank == 0:
                if all_result:
                    os.makedirs(os.path.join(cfg.trainers.output_dir, 'result'), exist_ok=True)
                    save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
                    save_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_{epoch}_{step}.csv'))
                    mean, summary_dict = summary(save_result, epoch, step, n_images_seen)
                    fabric.log_dict(summary_dict)
                    # MLflow 记录评估指标
                    mlflow_eval = {k.replace("@", "_at_"): float(v) for k, v in summary_dict.items() if isinstance(v, (int, float))}
                    _mlflow_queue.put((mlflow_eval, step))
                    summary_result = pd.DataFrame(pd.Series(summary_dict), columns=['val'])
                    summary_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_summary_{epoch}_{step}.csv'))
                else:
                    print('Skipped evaluation. So best is not updated')
                    mean = is_best_tracker.prev_best_metric
            else:
                mean = -1.0
            is_best_tracker.set_is_best(mean)
            if fabric.local_rank == 0:
                fabric.log_dict({'is_best': float(is_best_tracker.is_best())})
                print(f'Epoch {epoch} | Step {step} | Best {is_best_tracker.is_best()}')
                if all_result:
                    print(summary_result.round(2).to_markdown())

            # 检查是否需要早停（合并所有早停条件，统一广播）
            should_stop = False
            if fabric.local_rank == 0:
                # 1. loss 连续无改进
                if loss_improvement_monitor.update(avg_epoch_loss):
                    print(f"早停触发: train/mean_loss 连续{loss_improvement_monitor.patience}个epoch未降低{loss_improvement_monitor.min_improvement}")
                    should_stop = True
                # 2. summary mean 连续无提升
                if early_stopping.update(mean):
                    print(f"Early stopping triggered after {early_stopping.patience} epochs without improvement.")
                    should_stop = True
                # 3. 特定指标连续无足够改进
                if all_result:
                    for monitor in improvement_monitors:
                        if monitor.check(all_result):
                            print(f"早停触发: {monitor.metric_name} 连续{monitor.patience}个epoch无足够改进")
                            should_stop = True
            if broadcast_should_stop(fabric, should_stop):
                break

            # save model
            
            train_pipeline.save(fabric, train_pipeline, cfg, epoch, step, n_images_seen,
                                is_best=is_best_tracker.is_best())
            print('Evaluation Finished and Model Saved')

        epoch_time = (time.time() - epoch_start_time) / 60
        print(f'Epoch Time: {epoch_time:.2f} mins')

        torch.cuda.empty_cache()
    # load best model and do final eval
    is_best_path = os.path.join(cfg.trainers.output_dir, 'checkpoints', 'best')
    epoch = epoch + 1
    step = step + 1
    n_images_seen = n_images_seen + 1
    if os.path.exists(is_best_path) and cfg.trainers.skip_final_eval is False:
        fabric.barrier()
        time.sleep(fabric.local_rank * 5)  # prevent concurrent file access
        eval_pipeline.model.load_state_dict_from_path(os.path.join(is_best_path, 'model.pt'))
        print('Final Evaluation Started')

        # evaluation callbacks
        cfg.evaluations = config.load_yaml('final', directory='evaluations')
        evaluators = []
        for name, info in cfg.evaluations.per_epoch_evaluations.items():
            eval_data_path = os.path.join(cfg.evaluations.data_root, info.path)
            eval_type = info.evaluation_type
            eval_batch_size = info.batch_size
            eval_num_workers = info.num_workers
            evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                              transform=eval_pipeline.make_test_transform(),
                                              fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers)
            evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
            evaluators.append(evaluator)


        all_result = {}
        for evaluator in evaluators:
            print(f"Evaluating {evaluator.name}")
            result = evaluator.evaluate(eval_pipeline, epoch=epoch, step=step, n_images_seen=n_images_seen)
            all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

        # Combined evaluations (合并多源评估)
        combined_config = getattr(cfg.evaluations, 'combined_evaluations', None)
        if combined_config:
            evaluators_dict = {e.name: e for e in evaluators}
            combined_start = time.time()
            combined_result = run_combined_evaluations_distributed(
                fabric, evaluators_dict, combined_config, epoch, step, n_images_seen
            )
            all_result.update(combined_result)
            if fabric.local_rank == 0:
                print(f'合并评估完成，耗时: {(time.time() - combined_start) / 60:.2f} mins')

        if fabric.local_rank == 0:
            os.makedirs(os.path.join(cfg.trainers.output_dir, 'result'), exist_ok=True)
            save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
            save_result.to_csv(os.path.join(cfg.trainers.output_dir, f'result/eval_best.csv'))
            mean, summary_dict = summary(save_result, epoch, step, n_images_seen)
            summary_dict = {k.replace('summary/', 'final/'): v for k, v in summary_dict.items()}
            # round to 2 decimal places
            summary_dict = {k: np.round(v, 2) for k, v in summary_dict.items()}
            fabric.log_dict(summary_dict)
            # MLflow 记录最终评估指标
            mlflow_final = {k.replace("@", "_at_"): float(v) for k, v in summary_dict.items() if isinstance(v, (int, float, np.floating))}
            _mlflow_queue.put((mlflow_final, step))
            pd.DataFrame(pd.Series(summary_dict), columns=['val']).to_csv(
                os.path.join(cfg.trainers.output_dir, f'result/eval_summary_best.csv'))
    else:
        print('Skip final evaluation')

    # close
    if fabric.local_rank == 0:
        # 等待 MLflow 异步队列写完再关闭
        _mlflow_queue.put(None)
        _mlflow_thread.join(timeout=30)
        mlflow.end_run()
        for logger in fabric.loggers:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'finish'):
                logger.experiment.finish()
    print('done')
