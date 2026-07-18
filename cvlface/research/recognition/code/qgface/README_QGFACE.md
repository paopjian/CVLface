# QGFace training

This directory is based on `work_0605` and adds the quality-guided joint
training method from `/root/zhaokj/QGFace` to the existing CVLFace Fabric
training stack.

The default `base.yaml` selects:

- the IR-34 backbone used by the reference implementation;
- AdaFace classification loss;
- one original view and one degraded low-quality view of every image;
- the QGFace quality-guided contrastive loss and proxy-updated real-time queue.

The low-quality transform follows the reference `ContrastDataset`: the image is
always downsampled to 10%-50% resolution, JPEG-compressed at quality 75 and
upsampled to 112x112. Random crop uses scale 0.8-1.0, color jitter uses magnitude
0.2, and each optional transform has probability 0.2. Both views pass through
the same trainable backbone and both participate in AdaFace classification.

The paper profile uses SGD for 12 epochs with an initial learning rate of 0.2,
momentum 0.9, weight decay 5e-4, and 10x learning-rate drops after epochs 6 and
9. Quality threshold `b` is 0.2 and contrastive learning starts immediately.
The repository default dataset remains CASIA for local configuration checks;
select VGGFace2 to reproduce the paper's training dataset.

## Training

Run from this directory with the optimized entrypoint. It enables
`channels_last`, `torch.compile`, persistent workers and asynchronous MLflow
logging. Override the dataset and GPU-local batch size as needed:

```bash
cd /root/zhaokj/CVLface/cvlface/research/recognition/code/qgface
export LD_LIBRARY_PATH=/root/anaconda3/envs/cvlface/lib:$LD_LIBRARY_PATH
export DECODE_BACKEND=turbojpeg
conda run -n cvlface fabric run --devices=8 --precision=bf16-mixed \
  train_opt.py \
  dataset=configs/webface4m \
  trainers.num_gpu=8 \
  trainers.batch_size=64 \
  trainers.using_wandb=false
```

The 8-GPU `dataset_0605` command is also available as:

```bash
bash scripts/examples/run_qgface_ir34_0605.sh
```

Five IR-34/IR-101 and FC/PFC training variants are documented in
[`QGFACE_TRAINING_VARIANTS.md`](QGFACE_TRAINING_VARIANTS.md) and share this
entrypoint:

```bash
bash scripts/examples/run_qgface_variants_0605.sh <1|2|3|4|5>
```

`pipelines/configs/train_qgface_paper.yaml` sets queue size to the dataset's
identity count, as in the paper. A feature/proxy double queue consumes roughly
`2 * queue_size * embedding_size * 4` bytes. For `dataset_0605`, this would be
about 3.0 GiB per process and make every contrastive matrix 791,509 columns
wide, so the 37M training script deliberately uses the bounded 8192-entry
profile in `pipelines/configs/train_qgface.yaml`.

## Smoke test

The focused CPU test does not require a face dataset:

```bash
python test_qgface.py
```

A bounded end-to-end synthetic run can be launched with:

```bash
export DATA_ROOT=/tmp
python train_opt.py \
  models=iresnet/configs/v1_ir18 \
  dataset=configs/synthetic \
  classifiers=configs/fc \
  trainers.num_gpu=1 \
  trainers.precision=32-true \
  trainers.batch_size=2 \
  trainers.num_workers=0 \
  trainers.limit_num_batch=1 \
  trainers.using_wandb=false \
  trainers.skip_final_eval=true \
  trainers.compile_model=false \
  trainers.channels_last=false \
  dataset.num_classes=4 \
  dataset.num_image=4 \
  optims.num_epoch=1 \
  optims.warmup_epoch=0 \
  optims.lr_milestones=[] \
  pipelines.contrast_start_epoch=0 \
  pipelines.qgface.queue_size=8 \
  evaluations=configs/skip_eval
```
