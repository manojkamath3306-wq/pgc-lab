#!/usr/bin/env python3
"""Run neural_network three times and produce a CSV observation table."""
import csv
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "neural_network"
PATTERNS = {
    "cpu_ms": r"Average CPU compute time:\s*([\d.]+)",
    "kernel_ms": r"Average GPU kernel-only time:\s*([\d.]+)",
    "end_to_end_ms": r"Average GPU end-to-end time:\s*([\d.]+)",
    "max_difference": r"Maximum absolute output difference:\s*([\d.]+)",
    "gpu": r"GPU detected:\s*(.+)",
}

def parse_output(output):
    values = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Could not find {key!r} in program output.")
        values[key] = match.group(1).strip() if key == "gpu" else float(match.group(1))
    values["matrix"] = "PASSED" if "Matrix multiplication + bias: PASSED" in output else "FAILED"
    values["relu"] = "PASSED" if "ReLU: PASSED" in output else "FAILED"
    values["sigmoid"] = "PASSED" if "Sigmoid: PASSED" in output else "FAILED"
    values["match"] = "MATCH" if "CPU and GPU outputs: MATCH" in output else "MISMATCH"
    return values

def main():
    if not EXE.exists():
        sys.exit("Executable not found. Compile first: nvcc -O2 src/neural_network.cu -o neural_network")
    records = []
    for run in range(1, 4):
        print(f"Running experiment {run}/3...")
        completed = subprocess.run([str(EXE)], cwd=ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout)
        if completed.returncode != 0:
            sys.exit(f"Experiment exited with status {completed.returncode}.")
        row = parse_output(completed.stdout)
        row["run"] = run
        records.append(row)

    averages = {k: mean(r[k] for r in records) for k in ("cpu_ms", "kernel_ms", "end_to_end_ms")}
    averages["kernel_speedup"] = averages["cpu_ms"] / averages["kernel_ms"]
    averages["end_to_end_speedup"] = averages["cpu_ms"] / averages["end_to_end_ms"]

    out = ROOT / "results" / "tables" / "three_run_observations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "CPU compute (ms)", "GPU kernel-only (ms)", "GPU end-to-end (ms)",
                         "Matrix test", "ReLU test", "Sigmoid test", "CPU/GPU output"])
        for r in records:
            writer.writerow([r["run"], r["cpu_ms"], r["kernel_ms"], r["end_to_end_ms"],
                             r["matrix"], r["relu"], r["sigmoid"], r["match"]])
        writer.writerow(["Average", averages["cpu_ms"], averages["kernel_ms"], averages["end_to_end_ms"]])
        writer.writerow(["Kernel-only speedup", averages["kernel_speedup"]])
        writer.writerow(["End-to-end speedup", averages["end_to_end_speedup"]])

    print("\nAVERAGE OF THREE RUNS")
    print(f"CPU compute:          {averages['cpu_ms']:.6f} ms")
    print(f"GPU kernel-only:      {averages['kernel_ms']:.6f} ms")
    print(f"GPU end-to-end:       {averages['end_to_end_ms']:.6f} ms")
    print(f"Kernel-only speedup:  {averages['kernel_speedup']:.6f}x")
    print(f"End-to-end speedup:   {averages['end_to_end_speedup']:.6f}x")
    print(f"CSV saved to: {out}")

if __name__ == "__main__":
    main()
