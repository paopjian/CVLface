#!/bin/bash
################################################################################
# TopoFR 端到端全量训练脚本 (单阶段)
# 适用于: dataset_0605
# 模型: iResNet-101
# 损失: TopoFR + AdaFace
# 硬件: 8x GPU
################################################################################

set -e  # 遇到错误立即退出

echo "================================================================================"
echo "TopoFR 端到端全量训练"
echo "================================================================================"

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
# 791509 类 FC 头显存吃紧(实测峰值 22.0/23.5 GiB), 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 激活环境
echo "激活 conda 环境: cvlface"
source /root/anaconda3/bin/activate cvlface

# 切换到工作目录
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

# 训练参数配置
EXPERIMENT_NAME="topofr_e2e_fullmodel_$(date +%m%d_%H%M)"
OUTPUT_DIR="/data1/dataset_0605/train_output/${EXPERIMENT_NAME}"

echo "实验名称: ${EXPERIMENT_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "================================================================================"
echo "开始训练..."
echo "================================================================================"

fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=False \
    \
    models=iresnet/configs/v1_ir101.yaml \
    models.start_from=/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/adaface_ir101_webface12m/model.pt \
    models.freeze=False \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/fc.yaml \
    classifiers.sample_rate=0.40 \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/val_20260605.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.001 \
    optims.num_epoch=20 \
    optims.warmup_epoch=2 \
    "optims.lr_milestones=[10,15,18]" \
    optims.momentum=0.9 \
    optims.weight_decay=0.0005 \
    optims.lr_lambda=0.1 \
    optims.max_grad_norm=5.0 \
    optims.scheduler='step' \
    \
    pefts=configs/full.yaml

TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成！"
    echo "================================================================================"
    echo ""
    echo "输出目录: ${OUTPUT_DIR}"
    echo ""
    echo "最终模型位置:"
    echo "  - 最后一个epoch: ${OUTPUT_DIR}/checkpoints_every_epoch/epoch:*_step:*/"
    echo "  - 最佳模型: 根据验证集性能自动保存"
    echo ""
    echo "训练日志:"
    echo "  - TensorBoard: tensorboard --logdir=${OUTPUT_DIR}"
    echo "  - CSV日志: ${OUTPUT_DIR}/metrics.csv"
    echo "  - MLflow: ${OUTPUT_DIR}/mlflow.db"
    echo ""
    echo "下一步:"
    echo "  1. 查看训练日志验证收敛情况"
    echo "  2. 在验证集上评估最佳模型"
    echo "  3. 与baseline (AdaFace) 对比性能"
    echo ""
else
    echo "✗ 训练失败，退出码: ${TRAIN_EXIT_CODE}"
    echo "================================================================================"
    echo ""
    echo "故障排查:"
    echo "  1. 检查数据集路径是否正确"
    echo "  2. 检查GPU内存是否充足 (可能需要减小batch_size)"
    echo "  3. 查看日志文件: ${OUTPUT_DIR}/*.log"
    echo "  4. 如果是拓扑损失导致的OOM，可以临时设置 losses.topo_weight=0"
    echo ""
    exit $TRAIN_EXIT_CODE
fi

echo "================================================================================"
