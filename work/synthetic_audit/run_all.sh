#!/bin/sh
# Usage: sh work/synthetic_audit/run_all.sh [seed ...]   (default seeds: 1 2 3 4 5)
# Generates data, fits on synthetic train, scores unseen synthetic test, then aggregates.
set -e
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
SEEDS=${*:-1 2 3 4 5}
mkdir -p data results
for s in $SEEDS; do
  [ -f "data/seed$s/test/targets.csv" ] || $PY sim.py --seed "$s" --out "data/seed$s"
  $PY evaluate.py --data "data/seed$s" --out "results/seed$s.json" > "results/seed$s.log" 2>&1 &
done
wait
cat results/seed*.log
$PY report.py results/seed*.json | tee results/report.txt
