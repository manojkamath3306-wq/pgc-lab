#!/usr/bin/env python3
"""results/data.csv (+torch.csv) -> results/*.png and results/summary.md"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = load(); GPU = gpuinfo().get("GPU", "GPU"); S = stats(df)
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(RES, name)); plt.close(fig)

def p_speedup():
    fig, ax = plt.subplots(figsize=(7, 4.2)); sep = series(df, "colsweep", "separate", R=4096)
    for v in VARS[1:]:
        d = series(df, "colsweep", v, R=4096)
        if len(d): ax.plot(d.index, (sep / d).dropna(), "o-", color=COL[v], label=LAB[v])
    ax.axhline(1, color="k", lw=.8); ax.set_xscale("log", base=2); ax.set_xlabel("row length (columns), 4096 rows"); ax.set_ylabel("speedup over 3 separate kernels")
    ax.set_title(f"Speedup vs row length - {GPU}", fontsize=10); ax.legend(fontsize=8); save(fig, "speedup_vs_cols.png")
def p_time():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in VARS:
        d = series(df, "colsweep", v, R=4096)
        if len(d): ax.plot(d.index, d.values, "o-", color=COL[v], label=LAB[v])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("row length (columns), 4096 rows"); ax.set_ylabel("time (ms)")
    ax.set_title("Kernel time vs row length", fontsize=10); ax.legend(fontsize=8); save(fig, "time_vs_cols.png")
def p_bw():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in VARS:
        d = series(df, "colsweep", v, "gbps", R=4096)
        if len(d): ax.plot(d.index, d.values, "o-", color=COL[v], label=LAB[v])
    ax.axhline(PEAK, color="k", ls="--", lw=1); ax.text(130, PEAK * 1.02, f"spec DRAM peak {PEAK:.0f} GB/s", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_xlabel("row length (columns), 4096 rows"); ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Effective bandwidth (ideal bytes / time); above the dashed line = L2-cache hits", fontsize=9); ax.legend(fontsize=8); save(fig, "bandwidth_vs_cols.png")
def p_heat():
    b = df[df.exp == "blocksweep"]
    if not len(b): return
    cs = sorted(b.cols.unique()); bl = sorted(b.block.unique()); fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, kind in zip(axs, ["time", "speedup"]):
        M = np.full((len(cs), len(bl)), np.nan)
        for i, c in enumerate(cs):
            for j, k in enumerate(bl):
                f = ms(df, "blocksweep", "fused_smem", 4096, c, k); s = ms(df, "blocksweep", "separate", 4096, c, k); M[i, j] = f if kind == "time" else s / f
        im = ax.imshow(M, cmap="viridis_r" if kind == "time" else "viridis", aspect="auto")
        ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl); ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs); ax.grid(False)
        ax.set_xlabel("block size (threads)"); ax.set_ylabel("columns (4096 rows)")
        best = np.nanargmin(M, axis=1) if kind == "time" else np.nanargmax(M, axis=1)
        for i in range(len(cs)):
            for j in range(len(bl)):
                if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.3f}" if kind == "time" else f"{M[i,j]:.2f}x", ha="center", va="center", fontsize=8, color="w", fontweight="bold" if j == best[i] else "normal")
        ax.set_title("fused time (ms), bold = best" if kind == "time" else "fused speedup vs separate (same block size)", fontsize=9); fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, "blocksize_heatmap.png")
def p_dram():
    d = df[df.exp == "dram"]
    if not len(d): return
    sz = sizes(df, "dram"); vs = [v for v in VARS if len(d[d.variant == v])]; fig, ax = plt.subplots(figsize=(7, 4.2)); w = .8 / len(vs)
    for k, v in enumerate(vs):
        ys = []
        for r, c in sz:
            q = d[(d.variant == v) & (d.rows == r) & (d.cols == c)]; ys.append(float(q.gbps.iloc[0]) if len(q) else 0)
        ax.bar(np.arange(len(sz)) + k * w, ys, w, color=COL[v], label=LAB[v])
    ax.axhline(PEAK, color="k", ls="--", lw=1); ax.set_xticks(np.arange(len(sz)) + .4 - w / 2); ax.set_xticklabels([f"{r}x{c}" for r, c in sz])
    ax.set_ylabel("effective bandwidth (GB/s)"); ax.set_title(f"DRAM-bound sizes (268 MB per matrix); dashed = spec peak {PEAK:.0f} GB/s", fontsize=9); ax.legend(fontsize=7); save(fig, "bandwidth_dram.png")
def p_break():
    sz = sizes(df, "baseline"); fig, ax = plt.subplots(figsize=(7, 4.2)); x = np.arange(len(sz)); bot = np.zeros(len(sz))
    for v, c, l in [("sep_max", "#7AA2F7", "max kernel"), ("sep_sum", "#F2B544", "sum-exp kernel"), ("sep_norm", "#C3A6FF", "normalise kernel")]:
        y = np.array([ms(df, "baseline", v, r, cc) for r, cc in sz]); ax.bar(x - .2, y, .38, bottom=bot, color=c, label=l); bot += y
    ax.bar(x + .2, [ms(df, "baseline", "fused_smem", r, c) for r, c in sz], .38, color=COL["fused_smem"], label="fused kernel")
    ax.set_xticks(x); ax.set_xticklabels([f"{r}x{c}" for r, c in sz], fontsize=8); ax.set_ylabel("time (ms)"); ax.set_title("Where the separate version spends its time vs fused", fontsize=10); ax.legend(fontsize=8); save(fig, "breakdown.png")
def p_attn():
    a = df[df.exp == "attention"]
    if not len(a): return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); sep = series(df, "attention", "separate")
    for v in VARS:
        d = series(df, "attention", v)
        if len(d):
            axs[0].plot(d.index, d.values, "o-", color=COL[v], label=LAB[v])
            if v != "separate": axs[1].plot(d.index, (sep / d).dropna(), "o-", color=COL[v], label=LAB[v])
    for a_ in axs: a_.set_xscale("log", base=2); a_.set_xlabel("sequence length (16 heads: rows = 16 x seq, cols = seq)")
    axs[0].set_yscale("log"); axs[0].set_ylabel("time (ms)"); axs[1].set_ylabel("speedup over separate"); axs[1].axhline(1, color="k", lw=.8)
    axs[0].set_title("Attention-score softmax: time", fontsize=10); axs[1].set_title("Attention-score softmax: speedup", fontsize=10); axs[1].legend(fontsize=7); save(fig, "attention.png")
for f in (p_speedup, p_time, p_bw, p_heat, p_dram, p_break, p_attn):
    try: f()
    except Exception as e: print("plot skipped:", f.__name__, e)

# ---------------- summary.md ----------------
def md(rows, head):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in rows)
out = f"# Summary of results\n\nGPU: {GPU}. Times in ms (average of 30 runs after 3 warm-up runs, CUDA events).\n\n## Baseline sizes\n\n"
rows = []
for (r, c), sp in zip(S["base_sizes"], S["base_speed"]):
    e = df[(df.exp == "baseline") & (df.variant == "fused_smem") & (df.rows == r) & (df.cols == c)].err.iloc[0]; nb = r * c * 4
    rows.append([f"{r}x{c}", fmt(ms(df, "baseline", "sep_max", r, c)), fmt(ms(df, "baseline", "sep_sum", r, c)), fmt(ms(df, "baseline", "sep_norm", r, c)), fmt(ms(df, "baseline", "separate", r, c)), fmt(ms(df, "baseline", "fused_smem", r, c)), f"{sp:.2f}x", f"{4*nb/1e6:.1f}", f"{2*nb/1e6:.1f}", f"{e:.1e}"])
out += md(rows, ["Rows x Cols", "max", "sum", "norm", "Separate", "Fused", "Speedup", "Bytes sep (MB)", "Bytes fused (MB)", "Max err"])
out += "\n## Row-length sweep (4096 rows, block 256)\n\n"
rows = [[c] + [fmt(ms(df, "colsweep", v, 4096, c)) for v in VARS] for c in S["cols"]]
out += md(rows, ["Cols"] + [LAB[v] for v in VARS])
out += "\n## Block-size sweep (fused kernel)\n\n"
rows = []
for c, bsz in S["best_block"].items():
    d = df[(df.exp == "blocksweep") & (df.variant == "fused_smem") & (df.cols == c)].sort_values("ms")
    rows.append([c, int(d.block.iloc[0]), fmt(d.ms.iloc[0]), int(d.block.iloc[-1]), fmt(d.ms.iloc[-1])])
out += md(rows, ["Cols", "Best block", "Best ms", "Worst block", "Worst ms"])
out += "\n## expf vs __expf (fused kernel)\n\n"; rows = []
for c in S["cols"]:
    a = df[(df.exp == "colsweep") & (df.variant == "fused_smem") & (df.cols == c)]; b = df[(df.exp == "colsweep") & (df.variant == "fused_fast") & (df.cols == c)]
    if len(a) and len(b): rows.append([c, fmt(a.ms.iloc[0]), f"{a.err.iloc[0]:.1e}", fmt(b.ms.iloc[0]), f"{b.err.iloc[0]:.1e}", f"{a.ms.iloc[0]/b.ms.iloc[0]:.2f}x"])
out += md(rows, ["Cols", "expf ms", "expf err", "__expf ms", "__expf err", "gain"])
open(os.path.join(RES, "summary.md"), "w").write(out); print("figures + summary.md written to results/")
