import subprocess
import csv
import os

SIZES = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]
THREADS = [1, 2, 4, 8]
BLOCK_SIZE = 256
REPEATS = 3

os.makedirs("results", exist_ok=True)
csv_file = "results/timings.csv"

with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "N", "Threads_or_Block", "Kernel_ms", "Total_ms"])

    for N in SIZES:
        print(f"=== Benchmarking N = {N} ===")
        
        # 1. Sequential
        seq_times = []
        for _ in range(REPEATS):
            out = subprocess.check_output(["./bin/sequential", str(N)]).decode()
            seq_times.append(float(out.split("Time_ms=")[1].strip()))
        avg_seq = sum(seq_times) / REPEATS
        print(f"  [Sequential] Time: {avg_seq:.3f} ms")
        writer.writerow(["Sequential", N, 1, 0.0, avg_seq])

        # 2. OpenMP
        for T in THREADS:
            omp_times = []
            for _ in range(REPEATS):
                out = subprocess.check_output(["./bin/openmp_cpu", str(N), str(T)]).decode()
                omp_times.append(float(out.split("Time_ms=")[1].strip()))
            avg_omp = sum(omp_times) / REPEATS
            print(f"  [OpenMP {T}T]   Time: {avg_omp:.3f} ms")
            writer.writerow(["OpenMP", N, T, 0.0, avg_omp])

        # 3. CUDA
        cuda_k, cuda_t = [], []
        for _ in range(REPEATS):
            out = subprocess.check_output(["./bin/cuda_gpu", str(N), str(BLOCK_SIZE)]).decode()
            parts = out.strip().split(",")
            cuda_k.append(float(parts[3].split("=")[1]))
            cuda_t.append(float(parts[4].split("=")[1]))
        avg_k = sum(cuda_k) / REPEATS
        avg_t = sum(cuda_t) / REPEATS
        print(f"  [CUDA GPU]   Kernel: {avg_k:.3f} ms | Total: {avg_t:.3f} ms")
        writer.writerow(["CUDA", N, BLOCK_SIZE, avg_k, avg_t])

print(f"\nAll benchmark runs completed! Output recorded to {csv_file}")
