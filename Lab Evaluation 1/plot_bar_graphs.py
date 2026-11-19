import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("graphs", exist_ok=True)
df = pd.read_csv("results/timings.csv")

# Filter representative dataset sizes for clean bar visualization
sample_sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000]
filtered_df = df[df["N"].isin(sample_sizes)].copy()

labels = [f"{int(n/1e6)}M" for n in sample_sizes]
x = np.arange(len(labels))
width = 0.22

# Extract times
seq_times = filtered_df[filtered_df["Model"] == "Sequential"]["Total_ms"].tolist()
omp_4t = filtered_df[(filtered_df["Model"] == "OpenMP") & (filtered_df["Threads_or_Block"] == 4)]["Total_ms"].tolist()
omp_8t = filtered_df[(filtered_df["Model"] == "OpenMP") & (filtered_df["Threads_or_Block"] == 8)]["Total_ms"].tolist()
cuda_times = filtered_df[filtered_df["Model"] == "CUDA"]["Total_ms"].tolist()

# -------------------------------------------------------------
# GRAPH 1: Grouped Bar Chart (Execution Time Comparison)
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, seq_times, width, label="Sequential CPU", color="#4C72B0")
plt.bar(x - 0.5*width, omp_4t, width, label="OpenMP (4 Threads)", color="#55A868")
plt.bar(x + 0.5*width, omp_8t, width, label="OpenMP (8 Threads)", color="#C44E52")
plt.bar(x + 1.5*width, cuda_times, width, label="CUDA (Total)", color="#8172B3")

plt.xlabel("Vector Size (N in Millions)", fontsize=11, fontweight="bold")
plt.ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
plt.title("Execution Time by Model and Data Size", fontsize=13, fontweight="bold")
plt.xticks(x, labels)
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

bar_chart_path = "graphs/execution_time_bar_chart.png"
plt.savefig(bar_chart_path, dpi=300)
plt.close()
print(f"Generated: {bar_chart_path}")

# -------------------------------------------------------------
# GRAPH 2: Stacked Bar Chart (CUDA PCIe vs Compute Kernel)
# -------------------------------------------------------------
cuda_df = filtered_df[filtered_df["Model"] == "CUDA"]
kernel_ms = cuda_df["Kernel_ms"].tolist()
transfer_ms = (cuda_df["Total_ms"] - cuda_df["Kernel_ms"]).tolist()

plt.figure(figsize=(8, 6))

plt.bar(labels, kernel_ms, width=0.45, label="Kernel Compute Time", color="#2ca02c")
plt.bar(labels, transfer_ms, width=0.45, bottom=kernel_ms, label="PCIe Transfer Overhead (H2D + D2H)", color="#ff7f0e")

plt.xlabel("Vector Size (N in Millions)", fontsize=11, fontweight="bold")
plt.ylabel("Time (ms)", fontsize=11, fontweight="bold")
plt.title("CUDA Time Breakdown: Compute vs PCIe Transfer", fontsize=13, fontweight="bold")
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

stacked_path = "graphs/cuda_breakdown_bar_chart.png"
plt.savefig(stacked_path, dpi=300)
plt.close()
print(f"Generated: {stacked_path}")
