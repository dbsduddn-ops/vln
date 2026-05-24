#!/usr/bin/env bash
# Run NavGPT-2 R2R fine-tuning at 10% / 50% / 100% train data (paper Table 3, arXiv:2407.12366).
# Each run: train + test with separate outdir (...-pct10 / pct50 / pct100).
#
# Usage (from map_nav_src):
#   bash scripts/run_r2r_xl_data_ablation.sh
# Single setting only:
#   TRAIN_DATA_PCT=50 bash scripts/run_r2r_xl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

for pct in 10 50 100; do
  echo "========== [data ablation] TRAIN_DATA_PCT=${pct} =========="
  TRAIN_DATA_PCT=${pct} bash "${SCRIPT_DIR}/run_r2r_xl.sh" || exit 1
done

echo "[run_r2r_xl_data_ablation] Done: 10%, 50%, 100% runs finished."
