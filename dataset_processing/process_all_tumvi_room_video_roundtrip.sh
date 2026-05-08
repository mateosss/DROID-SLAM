#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TUMVI_ROOT="${TUMVI_ROOT:-/storage/group/dataset_mirrors/tumvi/512_16}"
OUT_ROOT="${OUT_ROOT:-/storage/user/hilscher/tumvi_room_encoded2}"
CONFIG="${CONFIG:-/home/stud/hilscher/basalt/data/tum/tumvi_512_config_vo.json}"
PROCESSOR="${PROCESSOR:-$SCRIPT_DIR/euroc_encoding_script.py}"
PARALLEL="${PARALLEL:-4}"

export TUMVI_ROOT OUT_ROOT CONFIG PROCESSOR

printf "%s\n" \
  dataset-room1_512_16 \
  dataset-room2_512_16 \
  dataset-room3_512_16 \
  dataset-room4_512_16 \
  dataset-room5_512_16 \
  dataset-room6_512_16 \
| xargs -I{} -P "$PARALLEL" bash -c '
    dataset="$1"
    printf "[%s] Processing %s -> %s\n" "$(date +%H:%M:%S)" "$dataset" "$OUT_ROOT/$dataset"
    python3 -u "$PROCESSOR" \
      "$TUMVI_ROOT/$dataset" \
      "$OUT_ROOT/$dataset" \
      --config "$CONFIG"
  ' _ {}
