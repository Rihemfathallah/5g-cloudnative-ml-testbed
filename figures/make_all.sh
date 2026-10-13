#!/usr/bin/env bash
# Rebuild every report figure from results/ (one script per figure; each reads only CSVs).
#   figures/make_all.sh results/detection_main results/handover
set -euo pipefail
cd "$(dirname "$0")"
DET=${1:-../results/detection_main}; HO=${2:-../results/handover}
python3 fig_detection_summary.py "$DET" -o ../report/figures
python3 fig_detection_by_type.py "$DET" -o ../report/figures
python3 fig_detection_delay.py   "$DET" -o ../report/figures
python3 fig_attribution.py       "$DET" -o ../report/figures
python3 fig_handover_rates.py    "$HO"  -o ../report/figures
python3 fig_a3_grid.py           "$HO"  -o ../report/figures
