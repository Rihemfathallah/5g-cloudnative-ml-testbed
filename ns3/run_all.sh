#!/usr/bin/env bash
# Produce ns-3 traces for every split and speed, from the SAME trajectories as the Python producer.
#   NS3_DIR=~/ns-3-dev ./ns3/run_all.sh
set -euo pipefail
: "${NS3_DIR:?set NS3_DIR to your ns-3 tree}"
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
OUT="$ROOT/handover/traces/ns3"; mkdir -p "$OUT"
cp "$HERE/nr-rsrp-traces.cc" "$NS3_DIR/scratch/"
(cd "$NS3_DIR" && ./ns3 build scratch/nr-rsrp-traces)
tb handover export-trajectories -o "$OUT"          # writes <split>_<speed>kmh_traj.csv
for f in "$OUT"/*_traj.csv; do
  base=$(basename "$f" _traj.csv)
  (cd "$NS3_DIR" && ./ns3 run "scratch/nr-rsrp-traces --input=$f --output=$OUT/$base.csv --seed=$(( $(cksum <<<"$base" | cut -d' ' -f1) % 1000 + 1 ))")
done
echo "now set trace_source: ns3 in configs/handover.yaml and run: tb handover all"
