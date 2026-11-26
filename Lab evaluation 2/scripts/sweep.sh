#!/usr/bin/env bash
# Runs every experiment (baseline, row-length sweep, block-size sweep, DRAM-bound sizes, attention shapes)
# Writes results/data.csv, results/gpu.txt, results/run_log.txt.  Run from anywhere after `make gpu`.
set -e
cd "$(dirname "$0")/.."
mkdir -p results
[ -x ./bench ] || { echo "build first: make gpu ARCH=sm_120"; exit 1; }
./bench results/data.csv | tee results/run_log.txt
echo "done -> results/data.csv   (python3 scripts/plot.py to make figures)"
