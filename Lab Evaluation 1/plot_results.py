import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("graphs", exist_ok=True)
df = pd.read_csv("results/timings.csv")

# 1. Execution Time Plot (Log-Log)
plt.figure(figsize=(9, 5))
seq_df = df[df["Model"] == "Sequential"]
omp_df = df[(df["Model"] == "OpenMP") & (df["Threads_or_Block"] == 8)]
cuda_df = df[df["Model"] == "CUDA"]

plt.plot(seq_df["N"], seq_df["Total_ms"], 'o-', label="CPU Sequential")
plt.plot(omp_df["N"], omp_df["Total_ms"], 's-', label="OpenMP (8 Threads)")
plt.plot(cuda_df["N"], cuda_df["Kernel_ms"], '^-', label="CUDA (Kernel Compute)")
plt.plot(cuda_df["N"], cuda_df["Total_ms"], 'd--', label="CUDA (Total incl. PCIe)")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Vector Size (N)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time Comparison (Sequential vs OpenMP vs CUDA)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("graphs/execution_time_comparison.png", dpi=300)
plt.close()

# 2. Speedup Plot
plt.figure(figsize=(9, 5))
seq_times = seq_df.set_index("N")["Total_ms"]

omp_speedup = seq_times / omp_df.set_index("N")["Total_ms"]
cuda_kernel_speedup = seq_times / cuda_df.set_index("N")["Kernel_ms"]
cuda_total_speedup = seq_times / cuda_df.set_index("N")["Total_ms"]

plt.plot(seq_df["N"], omp_speedup, 's-', label="OpenMP Speedup (8T)")
plt.plot(seq_df["N"], cuda_kernel_speedup, '^-', label="CUDA Kernel Speedup")
plt.plot(seq_df["N"], cuda_total_speedup, 'd--', label="CUDA Total Speedup")

plt.axhline(y=1.0, color='r', linestyle=':', label="Baseline (1x)")
plt.xscale('log')
plt.xlabel("Vector Size (N)")
plt.ylabel("Speedup Factor (vs Sequential)")
plt.title("Speedup vs Problem Size (Crossover Analysis)")
plt.legend()
plt.grid(True)
plt.savefig("graphs/speedup_curves.png", dpi=300)
plt.close()

print("Graphs successfully created in 'graphs/' folder!")
