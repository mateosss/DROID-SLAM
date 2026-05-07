#!/bin/bash
# Launch one slurm job per EuRoC sequence.
# Usage:
#   ./slurm_scripts/launch_euroc_causal_all.sh                # all sequences
#   ./slurm_scripts/launch_euroc_causal_all.sh MH_01_easy V1_01_easy   # subset

set -euo pipefail

REPO_DIR="/home/stud/hilscher/DROID-SLAM"
EUROC_ROOT="/storage/user/hilscher/euroc_encoded2"
SLURM_FILE="${REPO_DIR}/slurm_scripts/test_euroc_causal.slurm"

ALL_SEQUENCES=(
  MH_01_easy
  MH_02_easy
  MH_03_medium
  MH_04_difficult
  MH_05_difficult
  V1_01_easy
  V1_02_medium
  V1_03_difficult
  V2_01_easy
  V2_02_medium
  V2_03_difficult
)

if [[ $# -gt 0 ]]; then
  SEQUENCES=("$@")
else
  SEQUENCES=("${ALL_SEQUENCES[@]}")
fi

echo "Submitting ${#SEQUENCES[@]} job(s)..."
for seq in "${SEQUENCES[@]}"; do
  if [[ ! -d "${EUROC_ROOT}/${seq}" ]]; then
    echo "  [skip] ${seq}: directory missing at ${EUROC_ROOT}/${seq}" >&2
    continue
  fi
  jobid=$(sbatch --parsable --job-name="droid-${seq}" --export=ALL,EUROC_SEQUENCE="${seq}" "${SLURM_FILE}")
  echo "  [ok]   ${seq}: job ${jobid}"
done
echo "Done. Track with: squeue -u \$USER"
