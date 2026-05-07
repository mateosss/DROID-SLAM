#!/bin/bash
#SBATCH --job-name=droid-euroc-causal
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/home/stud/hilscher/DROID-SLAM/slurm_outputs/euroc-causal-%j.out
#SBATCH --error=/home/stud/hilscher/DROID-SLAM/slurm_outputs/euroc-causal-%j.err
#SBATCH --mail-user=slurmbot-bPpujGpQYWMTPxWBnt@vision.in.tum.de
#SBATCH --mail-type=ALL

set -euo pipefail

pwd
hostname
date
nvidia-smi

###############################################
# USER SETTINGS                               #
###############################################

REPO_DIR="/home/stud/hilscher/DROID-SLAM"
OUTPUT_BASE_DIR="${REPO_DIR}/slurm_outputs"

# Override the sequence with:
#   sbatch --export=ALL,EUROC_SEQUENCE=V1_01_easy slurm_scripts/test_euroc_causal.slurm
EUROC_ROOT="/storage/user/hilscher/euroc_encoded2"
EUROC_SEQUENCE="${EUROC_SEQUENCE:-V1_02_medium}"
DATAPATH="${EUROC_ROOT}/${EUROC_SEQUENCE}"

CONDA_ENV="droid-torch27"
WEIGHTS="${REPO_DIR}/droid.pth"

RESULT_BASE_DIR="/storage/user/hilscher/droid_slam/euroc_causal"

BUFFER=512
IMAGE_HEIGHT=320
IMAGE_WIDTH=512
STRIDE=1
TRACK_STRIDE=2   # within the strided list, feed every Nth frame to tracker; filler uses every frame. Set to 2 to mirror evaluation_scripts/test_euroc.py.

# VRAM monitoring controls.
VRAM_MONITOR_INTERVAL=2

###############################################
# OUTPUT SETUP                                #
###############################################

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENE_NAME=$(basename "${DATAPATH}")
RUN_TAG="run_${RUN_TIMESTAMP}_job${SLURM_JOB_ID}"

RUN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SCENE_NAME}/${RUN_TAG}"
mkdir -p "${RUN_OUTPUT_DIR}"

RESULT_DIR="${RESULT_BASE_DIR}/${SCENE_NAME}/${RUN_TAG}"
mkdir -p "${RESULT_DIR}"

SAVE_PATH="${RESULT_DIR}/trajectory_keyframes.txt"
SAVE_PATH_FILLED="${RESULT_DIR}/trajectory_filled.txt"
VRAM_LOG_FILE="${RUN_OUTPUT_DIR}/vram_usage.csv"

echo "[$(date)] Output directory: ${RUN_OUTPUT_DIR}"
echo "[$(date)] Result directory: ${RESULT_DIR}"

VRAM_MONITOR_PID=""

cleanup() {
  if [[ -n "${VRAM_MONITOR_PID}" ]]; then
    if kill -0 "${VRAM_MONITOR_PID}" 2>/dev/null; then
      echo "[$(date)] Stopping VRAM monitor (PID ${VRAM_MONITOR_PID})"
      kill "${VRAM_MONITOR_PID}" 2>/dev/null || true
      wait "${VRAM_MONITOR_PID}" 2>/dev/null || true
    fi
    VRAM_MONITOR_PID=""
  fi

  SLURM_OUT_FILE="${OUTPUT_BASE_DIR}/euroc-causal-${SLURM_JOB_ID}.out"
  SLURM_ERR_FILE="${OUTPUT_BASE_DIR}/euroc-causal-${SLURM_JOB_ID}.err"

  echo "[$(date)] Moving output files to ${RUN_OUTPUT_DIR}..."
  sleep 1

  if [[ -f "${SLURM_OUT_FILE}" ]]; then
    mv "${SLURM_OUT_FILE}" "${RUN_OUTPUT_DIR}/stdout.out" 2>/dev/null || \
      cp "${SLURM_OUT_FILE}" "${RUN_OUTPUT_DIR}/stdout.out" 2>/dev/null || true
  fi
  if [[ -f "${SLURM_ERR_FILE}" ]]; then
    mv "${SLURM_ERR_FILE}" "${RUN_OUTPUT_DIR}/stderr.err" 2>/dev/null || \
      cp "${SLURM_ERR_FILE}" "${RUN_OUTPUT_DIR}/stderr.err" 2>/dev/null || true
  fi

  echo "[$(date)] All logs saved to: ${RUN_OUTPUT_DIR}"
  echo "[$(date)] Trajectories saved under: ${RESULT_DIR}"
}
trap cleanup EXIT

###############################################
# ENVIRONMENT SETUP                           #
###############################################

echo "[$(date)] EuRoC sequence path: ${DATAPATH}"
echo "[$(date)] Weights: ${WEIGHTS}"
echo "[$(date)] Conda env: ${CONDA_ENV}"

if [[ ! -d "${DATAPATH}" ]]; then
  echo "ERROR: DATAPATH does not exist: ${DATAPATH}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: Weights file does not exist: ${WEIGHTS}" >&2
  exit 1
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# conda's activation hooks aren't `-u`-clean (e.g. NVCC_PREPEND_FLAGS), so
# disable nounset around the activation and re-enable it after.
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/droid_slam:${REPO_DIR}/thirdparty/lietorch:${PYTHONPATH:-}"

###############################################
# RUN DROID-SLAM CAUSAL EUROC                 #
###############################################

cd "${REPO_DIR}"

echo "timestamp,index,name,memory.used [MiB],memory.total [MiB],utilization.gpu [%]" > "${VRAM_LOG_FILE}"
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "${VRAM_LOG_FILE}" 2>/dev/null || true
    sleep "${VRAM_MONITOR_INTERVAL}"
  done
) &
VRAM_MONITOR_PID=$!
echo "[$(date)] VRAM monitor started (PID ${VRAM_MONITOR_PID}); logging to ${VRAM_LOG_FILE}"

python slurm_scripts/test_euroc_causal.py \
  --datapath "${DATAPATH}" \
  --save_path "${SAVE_PATH}" \
  --save_path_filled "${SAVE_PATH_FILLED}" \
  --weights "${WEIGHTS}" \
  --buffer "${BUFFER}" \
  --image_size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
  --stride "${STRIDE}" \
  --track_stride "${TRACK_STRIDE}" \
  --disable_vis

echo "[$(date)] Causal EuRoC run finished."
echo "[$(date)] Keyframe trajectory: ${SAVE_PATH}"
echo "[$(date)] Filled trajectory: ${SAVE_PATH_FILLED}"

nvidia-smi
