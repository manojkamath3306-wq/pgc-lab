import subprocess
import csv
import os

# Problem sizes from 100K up to 10M
SIZES = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
THREADS = [1, 2, 4, 8]
K_ITER = 50   # 50 trigonometric operations per element
REPEATS = 3

os.makedirs("results", exist_ok=True)
csv_file = "results/sin_timings.csv"

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "N", "Threads", "K", "Kernel_ms", "Total_ms"])

    for N in SIZES:
        print(f"\n================ N = {N:,} (K = {K_ITER}) ================")

        # 1. OpenMP across various thread counts
        for T in THREADS:
            t_list = []
            for _ in range(REPEATS):
                out = subprocess.check_output(["./bin/openmp_sin", str(N), str(T), str(K_ITER)]).decode()
                # Expected output: OpenMP,N=...,Threads=...,K=...,Time_ms=...
                val = float(out.strip().split("Time_ms=")[1])
                t_list.append(val)
            avg_time = sum(t_list) / REPEATS
            print(f"  [OpenMP {T}T] Time: {avg_time:8.2f} ms")
            writer.writerow(["OpenMP", N, T, K_ITER, 0.0, avg_time])

        # 2. CUDA GPU
        k_list, tot_list = [], []
        for _ in range(REPEATS):
            out = subprocess.check_output(["./bin/cuda_sin", str(N), str(K_ITER)]).decode()
            # Expected output: CUDA_Heavy,N=...,K=...,Kernel_ms=...,Total_ms=...
            parts = out.strip().split(",")
            k_ms = float(parts[3].split("=")[1])
            tot_ms = float(parts[4].split("=")[1])
            k_list.append(k_ms)
            tot_list.append(tot_ms)
        avg_k = sum(k_list) / REPEATS
        avg_tot = sum(tot_list) / REPEATS
        print(f"  [CUDA GPU ] Kernel: {avg_k:6.2f} ms | Total (incl PCIe): {avg_tot:6.2f} ms")
        writer.writerow(["CUDA", N, 256, K_ITER, avg_k, avg_tot])

print(f"\nFinished! Results saved to {csv_file}")
