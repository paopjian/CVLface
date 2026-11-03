#!/usr/bin/env bash

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/bundle_grouped_dataset.sh" \
    "/data/zkj-data/dataset_J_test/all_aligned" \
    "/data2/zkj-data/dataset_2_all/rec/dataset_J_test"
