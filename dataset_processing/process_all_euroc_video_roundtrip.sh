#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EUROC_ROOT="${EUROC_ROOT:-/storage/group/dataset_mirrors/euroc}"
OUT_ROOT="${OUT_ROOT:-/storage/user/hilscher/euroc_encoded2}"
CONFIG="${CONFIG:-/home/stud/hilscher/basalt/data/euroc/euroc_config_vo.json}"
PROCESSOR="${PROCESSOR:-$SCRIPT_DIR/euroc_encoding_script.py}"
PARALLEL="${PARALLEL:-4}"

printf "%s\n" \
  MH_02_easy \
  MH_03_medium \
  MH_04_difficult \
  MH_05_difficult \
  V1_01_easy \
  V1_02_medium \
  V1_03_difficult \
  V2_01_easy \
  V2_02_medium \
  V2_03_difficult \
| xargs -I{} -P "$PARALLEL" python3 "$PROCESSOR" \
    "$EUROC_ROOT/{}" \
    "$OUT_ROOT/{}" \
    --config "$CONFIG"
