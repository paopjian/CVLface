#!/bin/bash
# card8(7卡): subcenter_ohem_s2 续评 epoch 4-9 共6个 (epoch 0-3 已在 wandb run zgputcgd, resume 自动跳过; 10-17 归 card8_2)
# 用法: nohup bash run_eval_card8.sh > eval3_results/subcenter_ohem_s2_resume.log 2>&1 &
cd /root/zhaokj/CVLface_rec/cvlface/research/recognition/code/work_260922_subcenter || exit 1
export PATH=/root/miniconda3/envs/cvlface/bin:$PATH
# wandb 走本地代理 (非交互 shell 不加载 .bashrc, 必须显式带)
export https_proxy=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
# 十档 FAR, 与项目内 bench1022 系 run 同口径
export TRT_TARGET_FARS=1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12
/root/miniconda3/envs/cvlface/bin/python eval_all_trt_launcher.py \
  --num_gpu 7 \
  --eval_config_name test_20260922 \
  --ckpt_dir /data1/dataset_260918/train_output/subcenter_ohem_260922_s2_body36_10-28_0/ckpt_split_card8_ep4-9 \
  --project_name work_260922_test \
  --name subcenter_ohem_s2
rc=$?
if [ $rc -eq 0 ]; then touch eval3_results/subcenter_ohem_s2.DONE; else touch eval3_results/subcenter_ohem_s2.FAILED; fi
exit $rc
