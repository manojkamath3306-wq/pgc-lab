"""Shared loading + helpers for plot.py / make_readme.py / make_ppt.py"""
import os, re, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PEAK = float(os.environ.get("PEAK_GBPS", "448"))          # RTX 5060 Ti spec peak DRAM bandwidth (GB/s); edit for other GPUs
COL = {"separate": "#8696AB", "fused_smem": "#35D0BA", "online": "#F2B544", "warp_row": "#7AA2F7", "fused_fast": "#F07178", "torch": "#C3A6FF"}
LAB = {"separate": "3 separate kernels", "fused_smem": "Fused (shared memory)", "online": "Online softmax", "warp_row": "Warp-per-row",
       "fused_fast": "Fused + __expf", "torch": "PyTorch softmax"}
VARS = ["separate", "fused_smem", "online", "warp_row", "fused_fast", "torch"]

def load():
    df = pd.read_csv(os.path.join(RES, "data.csv"))
    t = os.path.join(RES, "torch.csv")
    if os.path.exists(t): df = pd.concat([df, pd.read_csv(t)], ignore_index=True)
    df.loc[df.err < 0, "err"] = np.nan
    return df
def gpuinfo():
    d = {}
    for l in open(os.path.join(RES, "gpu.txt")).read().splitlines():
        if ": " in l: k, v = l.split(": ", 1); d[k] = v
    return d
def read(name):
    p = os.path.join(RES, name); return open(p, errors="replace").read() if os.path.exists(p) else ""
def cuda_release():
    m = re.search(r"release ([\d.]+)", read("nvcc_version.txt")); return m.group(1) if m else "?"
def driver_ver():
    m = re.search(r"Driver Version:\s*([\d.]+)", read("nvidia_smi.txt")); return m.group(1) if m else "?"
def ms(df, ex, var, R, C, bs=None):
    d = df[(df.exp == ex) & (df.variant == var) & (df.rows == R) & (df.cols == C)]
    if bs is not None: d = d[d.block == bs]
    return float(d.ms.iloc[0]) if len(d) else np.nan
def series(df, ex, var, key="ms", R=None):
    d = df[(df.exp == ex) & (df.variant == var)]
    if R is not None: d = d[d.rows == R]
    return d.set_index("cols")[key]
def sizes(df, ex):
    return sorted(df[df.exp == ex].groupby(["rows", "cols"]).size().index, key=lambda t: t[0] * t[1])
def fmt(x, n=3): return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"

def stats(df):
    """All numbers quoted in README/PPT come from here."""
    S = {}
    base = sizes(df, "baseline"); S["base_sizes"] = base
    S["base_speed"] = [ms(df, "baseline", "separate", r, c) / ms(df, "baseline", "fused_smem", r, c) for r, c in base]
    S["maxerr"] = float(df.err.max()); S["all_match"] = bool((df.dropna(subset=["err"]).err < 1e-5).all())
    S["n_checked"] = int(df.err.notna().sum())
    sep = series(df, "colsweep", "separate", R=4096); fus = series(df, "colsweep", "fused_smem", R=4096)
    S["col_speed"] = (sep / fus).dropna()
    S["cols"] = sorted(df[df.exp == "colsweep"].cols.unique())
    best = {}
    for c in S["cols"]:
        cand = {v: ms(df, "colsweep", v, 4096, c) for v in VARS}; cand = {k: x for k, x in cand.items() if not np.isnan(x)}
        if cand: best[c] = min(cand, key=cand.get)
    S["best_variant"] = best
    bsw = df[(df.exp == "blocksweep") & (df.variant == "fused_smem")]
    S["best_block"] = {int(c): int(bsw[bsw.cols == c].sort_values("ms").block.iloc[0]) for c in sorted(bsw.cols.unique())}
    dr = df[(df.exp == "dram") & (df.variant == "fused_smem")]; S["dram_fused"] = dr
    S["small_bw_max"] = float(df[(df.exp == "colsweep")].gbps.max()) if len(df[df.exp == "colsweep"]) else 0
    return S

def observations(df, S):
    """Plain-language findings computed from the data (explanations are hedged; they are hypotheses unless profiled)."""
    o = []; sp = S["base_speed"]
    o.append(f"Fusing the three kernels into one gave {min(sp):.2f}x to {max(sp):.2f}x on the baseline sizes. Ideal global-memory traffic falls from 4 accesses per element to 2, so about 2x is the expected ceiling for a memory-bound layer.")
    cs = S["col_speed"]
    if len(cs): o.append(f"Across row lengths {int(cs.index.min())} to {int(cs.index.max())} the fused speedup ranges from {cs.min():.2f}x (at {int(cs.idxmin())} columns) to {cs.max():.2f}x (at {int(cs.idxmax())} columns).")
    if S["best_variant"]: o.append("Fastest variant per row length: " + ", ".join(f"{c} cols: {LAB[v]}" for c, v in S["best_variant"].items()) + ". Differences under about 3% (for example fused vs fused + __expf) are within run-to-run noise.")
    wr = series(df, "colsweep", "warp_row", R=4096); fu = series(df, "colsweep", "fused_smem", R=4096)
    if len(wr): o.append("Warp-per-row (registers, no __syncthreads) against the shared-memory fused kernel on short rows: " + ", ".join(f"{int(c)} cols {fu[c]/wr[c]:.2f}x" for c in wr.index if c in fu.index) + " (above 1 means warp-per-row is faster).")
    on = series(df, "colsweep", "online", R=4096); se = series(df, "colsweep", "separate", R=4096)
    if len(on):
        rr = (on / fu).dropna(); win = [int(c) for c in rr.index if rr[c] < 0.97]; tie = [int(c) for c in rr.index if 0.97 <= rr[c] <= 1.03]
        o.append(f"Online softmax (one pass for max and sum, 3 nominal accesses per element) is {(se/on).min():.2f}x to {(se/on).max():.2f}x faster than separate kernels. Against the shared-memory fused kernel its time is {rr.min():.2f}x to {rr.max():.2f}x (below 1 means online is faster): it wins at " + (", ".join(map(str, win)) + " columns" if win else "no row length") + (f", ties (within 3%) at {', '.join(map(str, tie))}" if tie else "") + ".")
    if S["best_block"]: o.append("Best block size of the fused kernel: " + ", ".join(f"{c} cols -> {b} threads" for c, b in S["best_block"].items()) + ".")
    dr = S["dram_fused"]
    if len(dr): o.append(f"On DRAM-bound sizes (268 MB per matrix) the fused kernel reaches {dr.gbps.min():.0f} to {dr.gbps.max():.0f} GB/s effective bandwidth, up to {dr.gbps.max()/PEAK*100:.0f}% of the {PEAK:.0f} GB/s spec peak.")
    dd = df[df.exp == "dram"]
    if len(dd) and dd.gbps.max() > PEAK: o.append(f"Some DRAM-sized runs report more than the {PEAK:.0f} GB/s peak (up to {dd.gbps.max():.0f} GB/s, {dd.loc[dd.gbps.idxmax(),'variant']}). Effective bandwidth uses nominal accesses per element; a variant that reads the same row twice gets the second read from cache, so nominal bytes overstate its real DRAM traffic. Compare variants by time, and use the fused kernel's figure as the DRAM estimate.")
    if S["small_bw_max"] > PEAK: o.append(f"At small sizes the effective bandwidth rises to {S['small_bw_max']:.0f} GB/s, above the DRAM peak. The matrix fits in the GPU's L2 cache and the timing loop re-reads it from there, so these numbers show cache speed, not DRAM speed.")
    if len(series(df, "colsweep", "torch", R=4096)):
        tt = series(df, "colsweep", "torch", R=4096); o.append("Best own kernel against PyTorch softmax: " + ", ".join(f"{int(c)} cols {tt[c]/min(x for x in [ms(df,'colsweep',v,4096,c) for v in VARS[1:5]] if x==x):.2f}x" for c in tt.index) + " (above 1 means ours is faster).")
    a = df[(df.exp == "colsweep") & (df.variant == "fused_fast")]; b = df[(df.exp == "colsweep") & (df.variant == "fused_smem")]
    if len(a) and len(b):
        g = (b.set_index("cols").ms / a.set_index("cols").ms).dropna(); o.append(f"Replacing expf with __expf changed the fused time by {g.min():.2f}x to {g.max():.2f}x, with a worst error of {a.err.max():.1e} against {b.err.max():.1e} for expf." + (f" The median change is {g.median():.2f}x, i.e. no consistent gain, which fits a memory-bound layer: a cheaper exponential does not help when the kernel waits on memory." if abs(g.median() - 1) < 0.05 else ""))
    o.append(f"Correctness: {S['n_checked']} kernel runs were compared with a double-precision CPU softmax on 64 rows each; the worst maximum absolute error is {S['maxerr']:.1e} ({'all MATCH' if S['all_match'] else 'SOME MISMATCH, see data.csv'}).")
    return o
