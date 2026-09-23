import json
import os
import signal
import socket
import subprocess
import time
import traceback


_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
)


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(temporary_path, path)


def _find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


def _launch_rank_zero(cfg, checkpoint_dir, result_path, epoch):
    code_dir = os.path.dirname(os.path.abspath(__file__))
    eval_config_name = os.path.splitext(os.path.basename(cfg.evaluations.yaml_path))[0]
    timeout_minutes = int(cfg.trainers.external_eval_timeout_minutes)

    backend = getattr(cfg.trainers, "external_eval_backend", "torch")
    if backend == "trt":
        # TRT 子进程 (eval_all_trt_single): nvjpeg 解码 + TRT fp16 + v7 匹配, 无 NCCL;
        # engine 按 checkpoint 缓存 (首次 ~60s 构建, 重评同 ckpt ~2s 反序列化)
        python_bin = os.path.join(
            os.path.dirname(cfg.trainers.external_eval_fabric_bin), "python")
        eval_script = os.path.join(code_dir, "eval_all_trt_single.py")
        precision = getattr(cfg.trainers, "external_eval_precision", "fp16")
        command = [
            python_bin,
            eval_script,
            "--num_gpu", str(cfg.trainers.num_gpu),
            "--eval_config_name", eval_config_name,
            "--ckpt_path", checkpoint_dir,
            "--name", f"{os.path.basename(cfg.trainers.output_dir)}_epoch_{epoch}",
            "--precision", precision,
            "--result_path", result_path,
            "--timing",
        ]
    else:
        main_port = _find_available_port()
        command = [
            cfg.trainers.external_eval_fabric_bin,
            "run",
            f"--devices={cfg.trainers.num_gpu}",
            f"--precision={cfg.trainers.precision}",
            "--main-address=127.0.0.1",
            f"--main-port={main_port}",
            os.path.join(code_dir, "eval_all_torch_single.py"),
            "--eval_config_name", eval_config_name,
            "--pipeline_name", "default",
            "--single_ckpt_path", checkpoint_dir,
            "--name", f"{os.path.basename(cfg.trainers.output_dir)}_epoch_{epoch}",
            "--project_name", cfg.trainers.task,
            "--num_gpu", str(cfg.trainers.num_gpu),
            "--precision", cfg.trainers.precision,
            "--timeout_minutes", str(timeout_minutes),
            "--result_path", result_path,
            "--timing",
        ]

    child_env = os.environ.copy()
    for key in _DISTRIBUTED_ENV_KEYS:
        child_env.pop(key, None)
    child_env["WANDB_MODE"] = "disabled"
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["MPLCONFIGDIR"] = os.path.join("/tmp", "cvlface_external_eval_matplotlib")
    if backend == "trt":
        # nvjpeg 扩展编译/TRT 需要 conda env 的 CUDA 工具链与 libstdc++
        env_prefix = os.path.dirname(os.path.dirname(python_bin))
        child_env.setdefault("CUDA_HOME", env_prefix)
        child_env["PATH"] = (os.path.dirname(python_bin) + os.pathsep
                             + child_env.get("PATH", ""))
        child_env["LD_LIBRARY_PATH"] = (os.path.join(env_prefix, "lib") + os.pathsep
                                        + child_env.get("LD_LIBRARY_PATH", ""))

    print("External evaluation command:", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=code_dir,
        env=child_env,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired as error:
        _stop_process_group(process)
        raise RuntimeError(
            f"external evaluation timed out after {timeout_minutes} minutes"
        ) from error

    if return_code != 0:
        raise RuntimeError(f"external evaluation exited with code {return_code}")
    if not os.path.isfile(result_path):
        raise RuntimeError(f"external evaluation result not found: {result_path}")


def run_external_torch_eval(fabric, cfg, checkpoint_dir, epoch):
    """Run evaluation in a short-lived process tree and return raw metrics on rank 0."""
    result_dir = os.path.join(cfg.trainers.output_dir, "external_eval")
    result_path = os.path.join(result_dir, f"epoch_{epoch}_raw.json")
    status_path = os.path.join(result_dir, f"epoch_{epoch}_status.json")

    if fabric.local_rank == 0:
        for stale_path in (result_path, status_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)
        try:
            _launch_rank_zero(cfg, checkpoint_dir, result_path, epoch)
            _atomic_write_json(status_path, {"ok": True})
        except Exception as error:
            _atomic_write_json(
                status_path,
                {
                    "ok": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
    else:
        deadline = time.monotonic() + int(cfg.trainers.external_eval_timeout_minutes) * 60 + 120
        poll_interval = float(cfg.trainers.external_eval_poll_interval_seconds)
        while not os.path.isfile(status_path):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for external evaluation: {status_path}")
            time.sleep(poll_interval)

    fabric.barrier()
    with open(status_path, "r", encoding="utf-8") as handle:
        status = json.load(handle)
    if not status["ok"]:
        raise RuntimeError(
            "external evaluation failed:\n"
            f"{status.get('error', 'unknown error')}\n"
            f"{status.get('traceback', '')}"
        )

    if fabric.local_rank != 0:
        return {}
    with open(result_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["all_result"]
