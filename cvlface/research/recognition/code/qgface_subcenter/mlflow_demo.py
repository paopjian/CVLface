"""
MLflow Demo: 模拟人脸识别训练过程的实验追踪
运行方式: python mlflow_demo.py
查看结果: mlflow ui --host 0.0.0.0 --port 5000
"""

import mlflow
import mlflow.pytorch
import random
import math
import time
import os
import torch
import torch.nn as nn

# ============================================================
# 1. 配置 MLflow (本地 SQLite 存储, 无需服务器)
# ============================================================
# 数据存储在当前目录的 mlflow.db 文件中, artifacts 存在 mlruns/ 下
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
# 创建/选择一个实验
mlflow.set_experiment("face_recognition_training")


def simulate_training():
    """模拟一个完整的训练过程"""

    # ============================================================
    # 2. 开始一次 Run, 记录超参数
    # ============================================================
    with mlflow.start_run(run_name="ir101_adaface_demo") as run:
        # 记录超参数
        params = {
            "model": "iresnet101",
            "loss": "adaface",
            "optimizer": "SGD",
            "lr": 0.1,
            "batch_size": 256,
            "num_gpu": 8,
            "precision": "bf16-mixed",
            "dataset": "webface12m",
            "peft": "part_freeze_body.42",
            "embedding_dim": 512,
            "num_classes": 600000,
        }
        mlflow.log_params(params)

        # ============================================================
        # 3. 模拟训练循环, 记录 metrics
        # ============================================================
        num_epochs = 20
        steps_per_epoch = 500
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            # --- 训练阶段 ---
            epoch_loss = 0
            for step in range(1, steps_per_epoch + 1):
                global_step += 1
                # 模拟 loss 下降
                base_loss = 15.0 * math.exp(-0.15 * epoch) + random.gauss(0, 0.3)
                lr = 0.1 * (0.1 ** (epoch // 8))  # step decay

                # 每 50 步记录一次 step 级别 metrics
                if step % 50 == 0:
                    mlflow.log_metrics({
                        "train/loss": base_loss,
                        "train/lr": lr,
                        "train/grad_norm_backbone": random.uniform(0.5, 2.0),
                        "train/grad_norm_classifier": random.uniform(1.0, 5.0),
                    }, step=global_step)

                epoch_loss += base_loss

            avg_loss = epoch_loss / steps_per_epoch

            # --- 评估阶段 ---
            # 模拟各 benchmark 的准确率随训练提升
            progress = 1 - math.exp(-0.2 * epoch)
            eval_metrics = {
                "eval/lfw": min(0.990 + progress * 0.008 + random.gauss(0, 0.001), 0.9995),
                "eval/cfp_fp": min(0.960 + progress * 0.025 + random.gauss(0, 0.002), 0.990),
                "eval/agedb_30": min(0.955 + progress * 0.030 + random.gauss(0, 0.002), 0.985),
                "eval/cplfw": min(0.910 + progress * 0.040 + random.gauss(0, 0.003), 0.955),
                "eval/calfw": min(0.940 + progress * 0.020 + random.gauss(0, 0.002), 0.965),
                "eval/ijbb_tar_far1e-4": min(0.920 + progress * 0.040 + random.gauss(0, 0.003), 0.965),
                "eval/ijbc_tar_far1e-4": min(0.940 + progress * 0.035 + random.gauss(0, 0.003), 0.975),
                "eval/tinyface_rank1": min(0.680 + progress * 0.060 + random.gauss(0, 0.005), 0.750),
            }
            eval_metrics["epoch"] = epoch
            eval_metrics["train/epoch_avg_loss"] = avg_loss

            # 记录 epoch 级别 metrics
            mlflow.log_metrics(eval_metrics, step=global_step)

            print(f"Epoch {epoch:2d}/{num_epochs} | "
                  f"Loss: {avg_loss:.3f} | "
                  f"LFW: {eval_metrics['eval/lfw']:.4f} | "
                  f"IJB-C: {eval_metrics['eval/ijbc_tar_far1e-4']:.4f}")

        # ============================================================
        # 4. 记录最终结果和 artifacts
        # ============================================================
        # 记录最终 summary metrics
        mlflow.log_metrics({
            "final/lfw": eval_metrics["eval/lfw"],
            "final/ijbc_tar_far1e-4": eval_metrics["eval/ijbc_tar_far1e-4"],
            "final/tinyface_rank1": eval_metrics["eval/tinyface_rank1"],
        })

        # 保存一个模型 artifact (用小模型演示)
        dummy_model = nn.Linear(512, 128)
        mlflow.pytorch.log_model(dummy_model, "model")

        # 也可以记录任意文件作为 artifact
        with open("/tmp/training_notes.txt", "w") as f:
            f.write("Training completed successfully.\n")
            f.write(f"Best LFW: {eval_metrics['eval/lfw']:.4f}\n")
            f.write(f"Best IJB-C: {eval_metrics['eval/ijbc_tar_far1e-4']:.4f}\n")
        mlflow.log_artifact("/tmp/training_notes.txt")

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")


def simulate_comparison_run():
    """模拟第二次实验, 用于对比"""
    with mlflow.start_run(run_name="ir50_cosface_demo"):
        params = {
            "model": "iresnet50",
            "loss": "cosface",
            "optimizer": "SGD",
            "lr": 0.1,
            "batch_size": 512,
            "num_gpu": 8,
            "precision": "bf16-mixed",
            "dataset": "webface4m",
            "peft": "full",
            "embedding_dim": 512,
            "num_classes": 200000,
        }
        mlflow.log_params(params)

        num_epochs = 20
        global_step = 0
        for epoch in range(1, num_epochs + 1):
            global_step += 500
            progress = 1 - math.exp(-0.18 * epoch)
            # IR50 + CosFace 稍弱于 IR101 + AdaFace
            eval_metrics = {
                "eval/lfw": min(0.988 + progress * 0.007 + random.gauss(0, 0.001), 0.998),
                "eval/cfp_fp": min(0.950 + progress * 0.020 + random.gauss(0, 0.002), 0.980),
                "eval/agedb_30": min(0.948 + progress * 0.025 + random.gauss(0, 0.002), 0.978),
                "eval/ijbc_tar_far1e-4": min(0.930 + progress * 0.030 + random.gauss(0, 0.003), 0.968),
                "train/epoch_avg_loss": 16.0 * math.exp(-0.13 * epoch) + random.gauss(0, 0.3),
            }
            mlflow.log_metrics(eval_metrics, step=global_step)

        mlflow.log_metrics({
            "final/lfw": eval_metrics["eval/lfw"],
            "final/ijbc_tar_far1e-4": eval_metrics["eval/ijbc_tar_far1e-4"],
        })

        print(f"\nComparison run logged.")


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Demo - Face Recognition Training Simulation")
    print("=" * 60)

    print("\n--- Run 1: IR101 + AdaFace ---")
    simulate_training()

    print("\n--- Run 2: IR50 + CosFace (对比实验) ---")
    simulate_comparison_run()

    print("\n" + "=" * 60)
    print("完成! 查看结果:")
    print("  1. 启动 MLflow UI:")
    print(f"     mlflow ui --backend-store-uri sqlite:///{DB_PATH} --host 0.0.0.0 --port 5000")
    print("  2. 浏览器访问: http://<你的服务器IP>:5000")
    print("  3. 左侧选择 'face_recognition_training' 实验")
    print("  4. 点击某个 Run 查看详细 metrics 曲线")
    print("  5. 勾选多个 Run 点击 'Compare' 对比实验")
    print("=" * 60)
