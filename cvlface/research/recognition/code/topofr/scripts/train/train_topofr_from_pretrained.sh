#!/bin/bash
################################################################################
# TopoFR 端到端训练 - 使用 TopoFR 预训练模型
# 预训练模型: Glint360K_R100_TopoFR (IJB-C 97.60%)
# 在 dataset_0605 上继续训练
################################################################################

set -e

echo "================================================================================"
echo "TopoFR 端到端训练 (基于 TopoFR 预训练模型)"
echo "================================================================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
# 791509 类 FC 头显存吃紧(实测峰值 22.0/23.5 GiB), 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /root/anaconda3/bin/activate cvlface
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/topofr

PRETRAINED_MODEL="/root/zhaokj/CVLface/cvlface/pretrained_models/recognition/topofr100/Glint360K_R100_TopoFR_9760.pt"
EXPERIMENT_NAME="topofr_r100_from_pretrained_$(date +%m%d_%H%M)"

echo "预训练模型: ${PRETRAINED_MODEL}"
echo "模型架构: IResNet-100"
echo "实验名称: ${EXPERIMENT_NAME}"
echo ""

# 检查预训练模型是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 预训练模型不存在: ${PRETRAINED_MODEL}"
    echo "请检查路径是否正确"
    exit 1
fi

echo "✓ 预训练模型检查通过"
echo ""
echo "================================================================================"
echo "开始训练..."
echo "================================================================================"

# set -e 会让 fabric 失败时脚本直接退出, 走不到下面的 TRAIN_EXIT_CODE 判断.
# 关掉它, 让底下的错误分支真正生效.
set +e

# 关键配置说明:
#
# classifiers=configs/partial_fc.yaml (模型并行), 不要用 fc:
#   791,509 类下全量 fc 是 3.19x 的纯浪费. 实测 8卡/batch128:
#   fc 1662 vs partial_fc 4914 samples/s. 详见 TOPOFR_PERFORMANCE_REPORT.md.
#   原先这里写的 classifiers.sample_rate=0.40 是静默无效的 —— fc.py 不读这个
#   字段, 只有 partial_fc 读. 要省显存请换 configs/partial_fc_sample10.yaml.
#
# trainers.skip_final_eval=True 是必须的:
#   train_opt.py:618 在 final eval 阶段强制 config.load_yaml('final'),
#   无视 CLI 传的 evaluations=. 而 final.yaml 的 data_root
#   /data2/dataset_1201_rec 在本机不存在 -> 8 rank 全崩. 训练成果不丢
#   (best/model.pt 早已落盘), 但没必要白崩一次. 要根治就把 final.yaml 的
#   data_root 改成 /data1/dataset_0605.
#
# 不要在 CLI 传 --strategy:
#   train_opt.py:229 已经传了 DDPStrategy 对象, 重复设置会 ValueError 冲突.
#   --precision 不冲突(传的是字符串), 保留无妨.
fabric run --devices=8 --precision="bf16-mixed" \
    train_opt.py \
    trainers.prefix=${EXPERIMENT_NAME} \
    trainers.num_gpu=8 \
    trainers.batch_size=128 \
    trainers.num_workers=8 \
    trainers.precision=bf16-mixed \
    trainers.skip_final_eval=True \
    \
    models=iresnet_insightface/configs/v1_ir101.yaml \
    models.start_from=${PRETRAINED_MODEL} \
    models.freeze=False \
    \
    dataset=configs/dataset_0605_train_rec.yaml \
    dataset.model_save_dir=/data1/dataset_0605/train_output \
    \
    data_augs=configs/gridsample_v2_numpy.yaml \
    \
    classifiers=configs/partial_fc.yaml \
    \
    losses=configs/topofr_adaface.yaml \
    pipelines=configs/train_model_cls_topofr \
    \
    evaluations=configs/val_20260605.yaml \
    \
    optims=configs/step_sgd.yaml \
    optims.lr=0.0005 \
    optims.num_epoch=15 \
    optims.warmup_epoch=2 \
    "optims.lr_milestones=[8,12]" \
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
    echo "输出目录: /data1/dataset_0605/train_output/${EXPERIMENT_NAME}"
    echo ""
    echo "预训练模型信息:"
    echo "  - 来源: Glint360K"
    echo "  - 架构: IResNet-100"
    echo "  - 性能: IJB-C@1e-5 = 96.57%, IJB-C@1e-4 = 97.60%"
    echo ""
    echo "训练配置:"
    echo "  - 初始学习率: 0.0005 (比从头训练低)"
    echo "  - 总epochs: 15"
    echo "  - 使用较低学习率进行 fine-tuning"
    echo ""
else
    echo "✗ 训练失败，退出码: ${TRAIN_EXIT_CODE}"
    echo "================================================================================"
    exit $TRAIN_EXIT_CODE
fi

echo "================================================================================"
