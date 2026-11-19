import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("graphs", exist_ok=True)
df = pd.read_csv("results/sin_timings.csv")

# -------------------------------------------------------------
# GRAPH 1: Execution Time vs N (Log-Log Line Plot)
# -------------------------------------------------------------
plt.figure(figsize=(9, 5))

omp_1t = df[(df["Model"] == "OpenMP") & (df["Threads"] == 1)]
omp_4t = df[(df["Model"] == "OpenMP") & (df["Threads"] == 4)]
omp_8t = df[(df["Model"] == "OpenMP") & (df["Threads"] == 8)]
cuda_df = df[df["Model"] == "CUDA"]

plt.plot(omp_1t["N"], omp_1t["Total_ms"], 'o-', label="CPU OpenMP (1 Thread / Sequential)")
plt.plot(omp_4t["N"], omp_4t["Total_ms"], 's-', label="CPU OpenMP (4 Threads)")
plt.plot(omp_8t["N"], omp_8t["Total_ms"], 'd-', label="CPU OpenMP (8 Threads)")
plt.plot(cuda_df["N"], cuda_df["Total_ms"], '^-', label="CUDA GPU (Total Time incl. PCIe)", linewidth=2)
plt.plot(cuda_df["N"], cuda_df["Kernel_ms"], '*--', label="CUDA GPU (Kernel Compute Only)", linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Array Size (N)", fontsize=11, fontweight="bold")
plt.ylabel("Execution Time in ms (Log Scale)", fontsize=11, fontweight="bold")
plt.title("Compute-Intensive Sine Evaluation: CPU vs GPU Scaling (K=50)", fontsize=12, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/sin_execution_time_comparison.png", dpi=300)
plt.close()
print("Generated: graphs/sin_execution_time_comparison.png")

# -------------------------------------------------------------
# GRAPH 2: Speedup Factor Relative to CPU OpenMP (8 Threads)
# -------------------------------------------------------------
plt.figure(figsize=(9, 5))

omp_8t_times = omp_8t.set_index("N")["Total_ms"]
cuda_tot_speedup = omp_8t_times / cuda_df.set_index("N")["Total_ms"]
cuda_krn_speedup = omp_8t_times / cuda_df.set_index("N")["Kernel_ms"]

plt.plot(cuda_df["N"], cuda_tot_speedup, '^-', color="green", label="CUDA Total Speedup (vs 8-Thread CPU)")
plt.plot(cuda_df["N"], cuda_krn_speedup, '*--', color="purple", label="CUDA Kernel Speedup (vs 8-Thread CPU)")
plt.axhline(y=1.0, color='red', linestyle=':', label="Parity / 1x (OpenMP 8T Baseline)")

plt.xscale('log')
plt.xlabel("Array Size (N)", fontsize=11, fontweight="bold")
plt.ylabel("Speedup Multiplier (x times faster)", fontsize=11, fontweight="bold")
plt.title("CUDA Speedup over 8-Thread OpenMP CPU", fontsize=12, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/sin_speedup_curves.png", dpi=300)
plt.close()
print("Generated: graphs/sin_speedup_curves.png")

# -------------------------------------------------------------
# GRAPH 3: Direct Bar Chart Comparison (at 1M, 5M, 10M)
# -------------------------------------------------------------
sample_sizes = [1_000_000, 5_000_000, 10_000_000]
filtered = df[df["N"].isin(sample_sizes)]

labels = [f"{int(n/1e6)}M" for n in sample_sizes]
x = np.arange(len(labels))
width = 0.25

t_omp4 = filtered[(filtered["Model"] == "OpenMP") & (filtered["Threads"] == 4)]["Total_ms"].tolist()
t_omp8 = filtered[(filtered["Model"] == "OpenMP") & (filtered["Threads"] == 8)]["Total_ms"].tolist()
t_cuda = filtered[filtered["Model"] == "CUDA"]["Total_ms"].tolist()

plt.figure(figsize=(9, 5))
plt.bar(x - width, t_omp4, width, label="OpenMP (4 Threads)", color="#e74c3c")
plt.bar(x, t_omp8, width, label="OpenMP (8 Threads)", color="#f39c12")
plt.bar(x + width, t_cuda, width, label="CUDA GPU (Total Time)", color="#27ae60")

plt.xlabel("Array Size (N)", fontsize=11, fontweight="bold")
plt.ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
plt.title("Runtime Comparison: OpenMP CPU vs CUDA GPU (K=50)", fontsize=12, fontweight="bold")
plt.xticks(x, labels)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/sin_bar_chart.png", dpi=300)
plt.close()
print("Generated: graphs/sin_bar_chart.png")
