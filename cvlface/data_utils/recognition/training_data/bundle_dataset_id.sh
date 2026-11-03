#!/usr/bin/env bash

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/bundle_grouped_dataset.sh" \
    "/data2/zkj-data/dataset_2_all/dataset_id/aligned" \
    "/data2/zkj-data/dataset_2_all/rec/dataset_id"
