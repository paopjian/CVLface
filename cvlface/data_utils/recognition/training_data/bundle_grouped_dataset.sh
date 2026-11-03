#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "用法: $0 <源目录> <输出目录>" >&2
    exit 2
fi

readonly SOURCE_ROOT="$1"
readonly OUTPUT_ROOT="$2"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BUNDLER="${SCRIPT_DIR}/bundle_images_into_rec_v2.py"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo "源目录不存在: ${SOURCE_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${BUNDLER}" ]]; then
    echo "打包脚本不存在: ${BUNDLER}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

group_count=0
while IFS= read -r -d '' group_dir; do
    group_name="$(basename "${group_dir}")"
    save_dir="${OUTPUT_ROOT}/${group_name}"

    echo "开始打包: ${group_name}"
    conda run -n cvlface python "${BUNDLER}" \
        --source_dir "${group_dir}" \
        --save_dir "${save_dir}"
    echo "完成打包: ${save_dir}"

    group_count=$((group_count + 1))
done < <(find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ "${group_count}" -eq 0 ]]; then
    echo "源目录下没有可打包的分组文件夹: ${SOURCE_ROOT}" >&2
    exit 1
fi

echo "全部完成，共打包 ${group_count} 个分组，输出目录: ${OUTPUT_ROOT}"
