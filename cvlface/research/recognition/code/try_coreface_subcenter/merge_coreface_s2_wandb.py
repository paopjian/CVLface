import argparse
import math

import wandb


ENTITY = "kejian-zhao-tsinghua-university"
DEFAULT_PROJECT = "try_coreface"
SOURCE_NAMES = (
    "coreface_s2_body36_0605_07-25_1",
    "coreface_s2_body36_0605_07-26_0",
)
TARGET_NAME = "coreface_s2_body36"
DROP_HISTORY_PREFIXES = ("system/", "_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--step-metric",
        choices=("epoch", "trainer/global_step"),
        default="trainer/global_step",
    )
    return parser.parse_args()


def project_path(project):
    return f"{ENTITY}/{project}"


def get_runs(api, project):
    matched = {name: [] for name in (*SOURCE_NAMES, TARGET_NAME)}
    for run in api.runs(project_path(project)):
        if run.name in matched:
            matched[run.name].append(run)

    if matched[TARGET_NAME]:
        ids = [run.id for run in matched[TARGET_NAME]]
        raise RuntimeError(f"目标 run 已存在: {TARGET_NAME}, ids={ids}")

    source_runs = []
    for name in SOURCE_NAMES:
        runs = matched[name]
        if len(runs) != 1:
            ids = [run.id for run in runs]
            raise RuntimeError(f"源 run 必须唯一: {name}, ids={ids}")
        source_runs.append(runs[0])
    return source_runs


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def clean_history_row(row):
    return {
        key: value
        for key, value in row.items()
        if not key.startswith(DROP_HISTORY_PREFIXES) and not is_missing(value)
    }


def fetch_history(run):
    return [clean_history_row(row) for row in run.scan_history(page_size=2000)]


def metric_range(rows, key):
    values = [row[key] for row in rows if key in row]
    if not values:
        raise RuntimeError(f"历史记录缺少连接指标: {key}")
    if any(left > right for left, right in zip(values, values[1:])):
        raise RuntimeError(f"run 内的 {key} 不是单调递增")
    return min(values), max(values)


def validate_boundary(first_history, second_history, step_metric):
    boundaries = {}
    keys = ("epoch", "trainer/epoch")
    if step_metric == "trainer/global_step":
        keys += ("trainer/global_step", "step", "n_images_seen")

    for key in keys:
        first_range = metric_range(first_history, key)
        second_range = metric_range(second_history, key)
        if first_range[1] >= second_range[0]:
            raise RuntimeError(
                f"{key} 未正确连接: 前段最大值={first_range[1]}, "
                f"后段最小值={second_range[0]}"
            )
        boundaries[key] = (first_range, second_range)

    expected_epoch_boundary = ((0, 10), (11, 14))
    for key in ("epoch", "trainer/epoch"):
        if boundaries[key] != expected_epoch_boundary:
            raise RuntimeError(f"{key} 范围与预期不符: {boundaries[key]}")

    global_step_is_discontinuous = (
        step_metric == "trainer/global_step"
        and boundaries[step_metric][0][1] + 1 != boundaries[step_metric][1][0]
    )
    if global_step_is_discontinuous:
        raise RuntimeError(
            "trainer/global_step 在两个 run 的连接处不连续: "
            f"{boundaries['trainer/global_step']}"
        )
    return boundaries


def merge_config(runs):
    config = {}
    for run in runs:
        config.update(
            {
                key: value
                for key, value in dict(run.config).items()
                if not str(key).startswith("_")
            }
        )
    return config


def merge_tags(runs, project):
    tags = []
    for run in runs:
        tags.extend(run.tags or [])
    tags.extend((f"copied-from:{project}", "merged"))
    return list(dict.fromkeys(tags))


def build_notes(runs):
    lines = ["Merged from the following W&B runs with continuous epoch values:"]
    lines.extend(f"- {run.name} ({run.id}): {run.url}" for run in runs)
    return "\n".join(lines)


def copy_final_summary(source_run, target_run):
    for key, value in dict(source_run.summary).items():
        if not str(key).startswith("_"):
            target_run.summary[key] = value


def main():
    args = parse_args()
    api = wandb.Api(timeout=60)
    source_runs = get_runs(api, args.project)
    histories = [fetch_history(run) for run in source_runs]
    boundaries = validate_boundary(*histories, args.step_metric)

    print(f"source ids: {[run.id for run in source_runs]}")
    print(f"history rows: {[len(history) for history in histories]}")
    for key, value in boundaries.items():
        print(f"{key}: {value[0]} -> {value[1]}")

    target_run = wandb.init(
        entity=ENTITY,
        project=args.project,
        name=TARGET_NAME,
        config=merge_config(source_runs),
        tags=merge_tags(source_runs, args.project),
        notes=build_notes(source_runs),
    )
    print(f"created: {target_run.id} | {target_run.url}")

    wandb.define_metric(args.step_metric)
    wandb.define_metric("*", step_metric=args.step_metric)

    uploaded = 0
    for history in histories:
        for row in history:
            if row:
                wandb.log(row)
                uploaded += 1
                if uploaded % 100 == 0:
                    print(f"uploaded: {uploaded}")

    copy_final_summary(source_runs[-1], target_run)
    target_run.finish()
    print(f"finished: {TARGET_NAME}, uploaded rows: {uploaded}")


if __name__ == "__main__":
    main()
