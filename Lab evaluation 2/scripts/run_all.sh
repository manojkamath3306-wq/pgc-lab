#!/usr/bin/env bash
# One command: environment check -> build -> run all experiments -> (optional PyTorch) -> figures -> README -> PPT -> zip
#   bash scripts/run_all.sh                  (ARCH=sm_120 by default; override:  ARCH=sm_89 bash scripts/run_all.sh)
#   INSTALL_TORCH=1 bash scripts/run_all.sh  (also installs PyTorch cu128 for the library comparison; large download)
set -e
cd "$(dirname "$0")/.."
mkdir -p results docs/screenshots presentation
echo "== environment =="; nvidia-smi 2>&1 | tee results/nvidia_smi.txt
nvcc --version 2>&1 | tee results/nvcc_version.txt
echo "== build =="; make gpu ARCH=${ARCH:-sm_120} 2>&1 | tee results/compile_log.txt
echo "== run all experiments (a few minutes) =="; bash scripts/sweep.sh
echo "== python environment =="
if python3 -m venv .venv >/dev/null 2>&1; then source .venv/bin/activate; PIP="pip install -q"
else echo "(venv missing: sudo apt install python3-venv  -- using system python instead)"; PIP="pip install -q --break-system-packages --user"; fi
$PIP matplotlib pandas numpy python-pptx pillow
[ "$INSTALL_TORCH" = "1" ] && $PIP torch --index-url https://download.pytorch.org/whl/cu128
if python3 -c "import torch" >/dev/null 2>&1; then echo "== PyTorch baseline =="; python3 scripts/bench_torch.py results/torch.csv | tee results/torch_log.txt
else echo "(PyTorch not installed -> library comparison skipped; rerun with INSTALL_TORCH=1 to include it)"; fi
echo "== figures, README, PPT, zip =="
python3 scripts/plot.py && python3 scripts/make_readme.py && python3 scripts/make_ppt.py && python3 scripts/package.py
echo; echo "DONE. Files: README.md, presentation/*.pptx, results/*.png, and the zip in the project root."
