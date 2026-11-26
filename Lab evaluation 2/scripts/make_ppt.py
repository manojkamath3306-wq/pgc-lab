#!/usr/bin/env python3
"""results/ -> presentation/CUDA_Softmax_Kernel_Fusion.pptx  (same look as the GPU Retrieval Pipeline deck)"""
import os, sys, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

df = load(); g = gpuinfo(); S = stats(df); O = observations(df, S); GPU = g.get("GPU", "GPU")
C = lambda h: RGBColor.from_string(h)
DARK, CARDD, TEAL, BLUE, PURP, RED, AMB, MUTE, CARDL = "0F1722", "182638", "35D0BA", "7AA2F7", "C3A6FF", "F07178", "F2B544", "8696AB", "E7ECF2"
P = Presentation(); P.slide_width = Inches(13.333); P.slide_height = Inches(7.5); BLANK = P.slide_layouts[6]
TITLE = "CUDA Softmax and Kernel Fusion"; LAB2 = "Parallel & Distributed Computing Laboratory"; N = [0]

def new(dark):
    s = P.slides.add_slide(BLANK); s.background.fill.solid(); s.background.fill.fore_color.rgb = C(DARK if dark else "FFFFFF"); N[0] += 1; s._dark = dark; return s
def tx(s, x, y, w, h, t, size=16, color=None, bold=False, font="Calibri", align=None, anchor=None, spacing=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    if anchor: tf.vertical_anchor = anchor
    lines = t if isinstance(t, list) else [t]
    for i, l in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); r = p.add_run(); r.text = l; f = r.font; f.size = Pt(size); f.bold = bold; f.name = font
        f.color.rgb = C(color or ("FFFFFF" if s._dark else "0F1722"));
        if align: p.alignment = align
        if spacing: p.space_after = Pt(spacing)
    return tb
def head(s, eyebrow, title):
    tx(s, .7, .45, 11.9, .3, eyebrow.upper(), 12, TEAL, True); tx(s, .7, .85, 11.9, .95, title, 32, bold=True)
    tx(s, .7, 7.05, 8, .3, f"{TITLE}  ·  {LAB2}", 9, MUTE); tx(s, 11.9, 7.05, .8, .3, str(N[0]), 9, MUTE, align=PP_ALIGN.RIGHT)
def lead(s, t, y=1.85, size=18, h=.9): tx(s, .7, y, 11.93, h, t, size)
def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); b.fill.solid(); b.fill.fore_color.rgb = C(fill)
    if line: b.line.color.rgb = C(line); b.line.width = Pt(1.25)
    else: b.line.fill.background()
    try: b.adjustments[0] = .06
    except Exception: pass
    b.shadow.inherit = False; return b
def card(s, x, y, w, h, title, body):
    box(s, x, y, w, h, CARDD if s._dark else CARDL); tx(s, x + .2, y + .15, w - .4, .4, title, 16, bold=True); tx(s, x + .2, y + .65, w - .4, h - .8, body, 13, MUTE if s._dark else "3B4656")
def stat(s, x, y, w, h, big, cap, color=TEAL):
    box(s, x, y, w, h, CARDD if s._dark else CARDL); tx(s, x + .25, y + .2, w - .5, .7, big, 30, color, True); tx(s, x + .25, y + 1.0, w - .5, h - 1.1, cap, 13, MUTE if s._dark else "3B4656")
def stages(s, items, hi=(), y=2.05, h=1.55):
    n = len(items); gap = .35; w = (11.93 - gap * (n - 1)) / n
    for i, (t, b) in enumerate(items):
        x = .7 + i * (w + gap); box(s, x, y, w, h, CARDD, AMB if i in hi else TEAL); tx(s, x + .12, y + .12, w - .24, .35, t, 14, bold=True, color="FFFFFF"); tx(s, x + .12, y + .55, w - .24, h - .6, b, 11, MUTE)
        if i < n - 1:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w + .04), Inches(y + h / 2 - .14), Inches(gap - .08), Inches(.28)); a.fill.solid(); a.fill.fore_color.rgb = C(AMB); a.line.fill.background()
def code(s, x, y, w, h, lines):
    box(s, x, y, w, h, CARDD); tx(s, x + .25, y + .2, w - .5, h - .3, lines, 13, "DDE6F0", font="Courier New")
def table(s, x, y, w, rows, headr, colw=None, size=12, rh=.37):
    t = s.shapes.add_table(len(rows) + 1, len(headr), Inches(x), Inches(y), Inches(w), Inches(rh * (len(rows) + 1))).table
    if colw:
        for j, cw in enumerate(colw): t.columns[j].width = Inches(w * cw / sum(colw))
    for j, h_ in enumerate(headr):
        c = t.cell(0, j); c.text = str(h_); c.fill.solid(); c.fill.fore_color.rgb = C(CARDD)
        p = c.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(size); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = C("FFFFFF")
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i + 1, j); c.text = str(v); c.fill.solid(); c.fill.fore_color.rgb = C("FFFFFF" if i % 2 == 0 else CARDL)
            p = c.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(size); p.runs[0].font.color.rgb = C("0F1722")
    return t
def chart(s, kind, cats, ser, x, y, w, h, colors, stacked=False, ytitle=None, numfmt='0.0'):
    cd = CategoryChartData(); cd.categories = [str(c) for c in cats]
    for k, v in ser.items(): cd.add_series(k, [None if (x_ is None or x_ != x_) else float(x_) for x_ in v])
    ct = {"line": XL_CHART_TYPE.LINE_MARKERS, "bar": XL_CHART_TYPE.COLUMN_STACKED if stacked else XL_CHART_TYPE.COLUMN_CLUSTERED, "hbar": XL_CHART_TYPE.BAR_STACKED}[kind]
    gf = s.shapes.add_chart(ct, Inches(x), Inches(y), Inches(w), Inches(h), cd); ch = gf.chart; fc = C("FFFFFF" if s._dark else "0F1722")
    ch.font.size = Pt(11); ch.font.color.rgb = fc; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.color.rgb = fc
    ch.value_axis.has_major_gridlines = True; ch.value_axis.major_gridlines.format.line.color.rgb = C("2A3A52" if s._dark else "D5DBE5")
    ch.value_axis.tick_labels.font.color.rgb = fc; ch.category_axis.tick_labels.font.color.rgb = fc; ch.value_axis.tick_labels.number_format = numfmt; ch.value_axis.tick_labels.number_format_is_linked = False
    if ytitle: ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = ytitle; ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11); ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.color.rgb = fc
    for sr, col in zip(ch.plots[0].series, colors):
        if kind == "line": sr.format.line.color.rgb = C(col); sr.format.line.width = Pt(2.25); sr.marker.format.fill.solid(); sr.marker.format.fill.fore_color.rgb = C(col); sr.marker.format.line.color.rgb = C(col); sr.smooth = False
        else: sr.format.fill.solid(); sr.format.fill.fore_color.rgb = C(col)
    return ch
def pic(s, name, x, y, w, h):
    from PIL import Image
    fn = os.path.join(RES, name)
    if not os.path.exists(fn): return
    iw, ih = Image.open(fn).size; sc = min(w / iw, h / ih); s.shapes.add_picture(fn, Inches(x), Inches(y), Inches(iw * sc), Inches(ih * sc))
def have(ex, var=None): d = df[df.exp == ex]; return len(d if var is None else d[d.variant == var]) > 0

sp = S["base_speed"]; sz = S["base_sizes"]; cols = S["cols"]
# 1 title
s = new(True); tx(s, .9, 1.5, 11.5, .4, "PARALLEL & DISTRIBUTED COMPUTING LABORATORY", 14, TEAL, True); tx(s, .9, 2.1, 11.5, 1.2, "Softmax and Kernel Fusion", 54, bold=True)
tx(s, .9, 3.25, 11.5, .7, "row-wise softmax in CUDA: separate kernels, fusion and beyond", 28, TEAL)
tx(s, .9, 4.5, 11.5, .5, "Team: Mudit Nayak · Kartik · Manoj · Vinayak", 22); tx(s, .9, 5.05, 11.5, .4, f"KLE Technological University, Hubballi  ·  CUDA  ·  {GPU}  ·  WSL2", 16, MUTE); tx(s, .9, 5.5, 11.5, .4, "Evaluation Theme 10  ·  [Add guide name and date]", 16, MUTE)
# 2 problem
s = new(False); head(s, "The problem", "Softmax is everywhere, and it is memory-bound")
lead(s, "For every row x: softmax(x)_i = exp(x_i − max) / Σ_j exp(x_j − max). Each row needs two reductions (max, sum) and one element-wise pass.")
for i, (t, b) in enumerate([("Classifiers", "Turns the final scores into class probabilities."), ("Attention", "One softmax per row of the QKᵀ score matrix in every transformer layer."), ("LLM inference", "Runs for every head, every layer, every token: small kernels, huge memory traffic.")]): card(s, .7 + i * 4.05, 3.0, 3.85, 1.75, t, b)
lead(s, "Very little arithmetic per byte, so time is set by how many times the matrix crosses global memory. That is why fusing kernels matters.", 5.2)
# 3 why parallel
s = new(False); head(s, "Why parallel", "A big matrix, many independent rows")
big = df[(df.exp == "dram")]; bigR, bigC = (int(big.rows.iloc[0]), int(big.cols.iloc[0])) if len(big) else (16384, 4096)
stat(s, .7, 2.1, 3.85, 2.1, f"{bigR*bigC*4/1e6:.0f} MB", f"one {bigR} × {bigC} float32 matrix, read and written by every softmax call")
stat(s, 4.75, 2.1, 3.85, 2.1, f"{bigR}", "independent rows: one CUDA block per row, no communication between rows", BLUE)
stat(s, 8.8, 2.1, 3.85, 2.1, "2 reductions", "per row (max, sum) that threads inside a block must cooperate on", AMB)
lead(s, "Parallelism at two levels: across rows (blocks) and across the elements of a row (threads, warp shuffles, shared memory).", 4.7)
# 4 design
s = new(True); head(s, "Design", "Separate kernels versus one fused kernel")
stages(s, [("Read x", "global memory"), ("max", "block reduction"), ("Read x again", "global memory"), ("Σ exp(x−m)", "block reduction"), ("Read x, write y", "normalise")], hi=(1, 3))
lead(s, "Separate: 3 launches, 4 global accesses per element. Fused: 1 launch, row cached in shared memory, 2 accesses per element (1 read, 1 write).", 4.0, 17)
tx(s, .7, 5.1, 11.9, 1.2, ["Extra variants measured: online softmax (running max + rescaled sum, 3 accesses), warp-per-row (registers, no __syncthreads), fused with __expf, and PyTorch softmax as a library baseline."], 15, MUTE)
# 5 setup
s = new(False); head(s, "Experimental setup", "Hardware, software and method")
table(s, .7, 1.9, 11.9, [["GPU", f"{GPU}, {g.get('MemGB','?')} GB, {g.get('SMs','?')} SMs, compute capability {g.get('CC','?')}, L2 {g.get('L2 MB','?')} MB"], ["Software", f"CUDA {cuda_release()} (nvcc), driver {driver_ver()}, Ubuntu on WSL2"],
   ["Data", "Synthetic uniform random float32 values in [−10, 10]"], ["Experiments", "baseline sizes · row-length sweep 128–8192 · block-size sweep 64–1024 · DRAM-bound 268 MB · attention shapes"],
   ["Timing", "CUDA events; 3 warm-up runs, average of 30 runs"], ["Verification", "double-precision CPU softmax on 64 rows; output zeroed before each kernel"]], ["Item", "Detail"], [1.4, 9], 13, .5)
lead(s, "Correctness is checked against the CPU on every run before any speed is reported.", 5.6, 16)
# 6 concept
s = new(False); head(s, "Implementation · concept", "Numerical stability and the block reduction")
tx(s, .7, 1.8, 6, 2.6, ["Without the shift, exp(x) overflows to inf for x > ~88 in float32.", "Subtracting the row max m makes every exponent ≤ 0 and leaves the result unchanged.", "Max and sum are reductions: warp shuffle inside each warp, then shared memory + __syncthreads across warps."], 16, spacing=10)
code(s, 7.0, 1.9, 5.6, 2.4, ["float m = warpMax(...);", "  // __shfl_down_sync(mask, v, o)", "v = expf(x - m);", "s += v;   // block-wide sum", "y = v * (1.f / s);"])
lead(s, "[Add your own worked example for a 4-element row with and without the max subtraction]", 4.9, 15)
# 7 separate
s = new(True); head(s, "Implementation · separate kernels", "Three kernels, one block per row")
tb_rows = [[f"{r}×{c}", fmt(ms(df, "baseline", "sep_max", r, c)), fmt(ms(df, "baseline", "sep_sum", r, c)), fmt(ms(df, "baseline", "sep_norm", r, c)), fmt(ms(df, "baseline", "separate", r, c))] for r, c in sz]
t = table(s, .7, 1.9, 6.9, tb_rows, ["Size", "max", "sum-exp", "normalise", "total (ms)"], [1.5, 1, 1.2, 1.2, 1.3], 12)
code(s, 7.9, 1.9, 4.7, 2.2, ["k_max      <<<rows, 256>>>", "k_sumexp   <<<rows, 256>>>", "k_norm     <<<rows, 256>>>", "// 4 global accesses / element"])
lead(s, "Per-kernel times are measured standalone; the normalise kernel is the only one that reads and writes the whole matrix.", 4.5, 16)
# 8 fused
s = new(False); head(s, "Implementation · fused kernel", "One block per row, row cached in shared memory")
for i, (t_, b_) in enumerate([("1 · Load once", "Each thread loads a strided slice of the row into shared memory and keeps a running max."), ("2 · Reduce twice", "The same block reduction gives the max, then the sum of exp(x−max), without touching global memory."), ("3 · Write once", "Normalise from shared memory and write y: 2 global accesses per element instead of 4.")]): card(s, .7 + i * 4.05, 1.95, 3.85, 2.2, t_, b_)
lead(s, "Limit: the row must fit in shared memory (columns × 4 bytes ≤ 48 KB, i.e. up to 12,288 columns). Launch: k_fused<<<rows, 256, cols*4>>>.", 4.55, 16)
# 9 variants
s = new(False); head(s, "Implementation · other variants", "Online softmax, warp-per-row and fast math")
for i, (t_, b_) in enumerate([("Online softmax", "Keeps (max, sum) together and rescales the sum when a bigger max appears, so one pass gives both. Then one write pass: 3 accesses."), ("Warp-per-row", "One warp owns a row (≤ 1024 columns); 32 values per lane stay in registers; no shared memory, no __syncthreads."), ("Fast math", "Replaces expf with __expf. Faster instruction, slightly less accurate; the error is measured against the CPU.")]): card(s, .7 + i * 4.05, 1.95, 3.85, 2.5, t_, b_)
# 10 correctness
s = new(True); head(s, "Results · correctness", "Every kernel matches the CPU")
stat(s, .7, 2.0, 3.85, 2.2, f"{S['n_checked']}", "kernel runs verified against the CPU reference (every variant, every experiment)")
stat(s, 4.75, 2.0, 3.85, 2.2, f"{S['maxerr']:.1e}", "largest absolute difference versus the double-precision CPU result", BLUE)
stat(s, 8.8, 2.0, 3.85, 2.2, "MATCH" if S["all_match"] else "CHECK", "status at threshold 1e-5" if S["all_match"] else "some runs exceeded 1e-5: see data.csv", AMB if S["all_match"] else RED)
lead(s, "All variants compute the same function, so the speedups that follow compare equal work.", 4.7)
# 11 baseline table
s = new(False); head(s, "Results · baseline", "Separate versus fused")
rows = [[f"{r}×{c}", fmt(ms(df, "baseline", "separate", r, c)), fmt(ms(df, "baseline", "fused_smem", r, c)), f"{x:.2f}×", f"{4*r*c*4/1e6:.1f}", f"{2*r*c*4/1e6:.1f}"] for (r, c), x in zip(sz, sp)]
table(s, .7, 1.9, 11.9, rows, ["Matrix", "Separate (ms)", "Fused (ms)", "Speedup", "Bytes sep (MB)", "Bytes fused (MB)"], None, 13, .48)
lead(s, O[0], 5.2, 15, 1.2)
# 12 breakdown
s = new(True); head(s, "Results · where the time goes", "Time per kernel, separate versus fused")
chart(s, "bar", [f"{r}×{c}" for r, c in sz], {"max": [ms(df, "baseline", "sep_max", r, c) for r, c in sz], "sum-exp": [ms(df, "baseline", "sep_sum", r, c) for r, c in sz], "normalise": [ms(df, "baseline", "sep_norm", r, c) for r, c in sz]}, .7, 1.7, 5.9, 4.4, [BLUE, AMB, PURP], True, "ms", '0.00')
chart(s, "bar", [f"{r}×{c}" for r, c in sz], {"fused": [ms(df, "baseline", "fused_smem", r, c) for r, c in sz]}, 6.8, 1.7, 5.9, 4.4, [TEAL], False, "ms", '0.00')
lead(s, "Left: the three separate kernels stacked (standalone times). Right: the single fused kernel at the same sizes.", 6.2, 14, .6)
# 13 speedup vs cols
if have("colsweep"):
    s = new(False); head(s, "Results · row length", "Speedup over separate kernels vs row length")
    ser = {}; col_ = []
    for v in VARS[1:]:
        d = series(df, "colsweep", v, R=4096); se = series(df, "colsweep", "separate", R=4096)
        if len(d): ser[LAB[v]] = [se[c] / d[c] if c in d.index else None for c in cols]; col_.append(COL[v].lstrip("#"))
    chart(s, "line", cols, ser, .7, 1.7, 11.9, 4.3, col_, False, "speedup (×)", '0.0'); lead(s, O[1] + " " + O[2], 6.05, 13, 1)
# 14 bandwidth
if have("colsweep"):
    s = new(True); head(s, "Results · bandwidth", f"Effective bandwidth (spec peak {PEAK:.0f} GB/s)")
    ser = {}; col_ = []
    for v in VARS:
        d = series(df, "colsweep", v, "gbps", R=4096)
        if len(d): ser[LAB[v]] = [d[c] if c in d.index else None for c in cols]; col_.append(COL[v].lstrip("#"))
    ser[f"spec peak {PEAK:.0f}"] = [PEAK] * len(cols); col_.append("FFFFFF"); chart(s, "line", cols, ser, .7, 1.7, 11.9, 4.2, col_, False, "GB/s", '0')
    lead(s, next((x for x in O if "L2" in x), "Effective bandwidth = ideal bytes ÷ time."), 6.0, 13, 1)
# 15 dram
if have("dram"):
    s = new(False); head(s, "Results · DRAM-bound sizes", "268 MB matrices: true memory-bandwidth limit")
    dsz = sizes(df, "dram"); ser = {}; col_ = []
    for v in VARS:
        d = df[(df.exp == "dram") & (df.variant == v)]
        if len(d): ser[LAB[v]] = [float(d[(d.rows == r) & (d.cols == c)].gbps.iloc[0]) if len(d[(d.rows == r) & (d.cols == c)]) else None for r, c in dsz]; col_.append(COL[v].lstrip("#"))
    chart(s, "bar", [f"{r}×{c}" for r, c in dsz], ser, .7, 1.7, 11.9, 4.3, col_, False, "GB/s", '0'); lead(s, next((x for x in O if "DRAM-bound" in x), ""), 6.05, 13, 1)
# 16 block sizes
if have("blocksweep"):
    s = new(False); head(s, "Results · block size", "Best block size depends on row length")
    pic(s, "blocksize_heatmap.png", .7, 1.7, 11.9, 4.2); lead(s, next((x for x in O if x.startswith("Best block")), ""), 6.1, 14, .8)
# 17 attention
if have("attention"):
    s = new(True); head(s, "Results · attention shapes", "Softmax over attention scores (16 heads)")
    asz = sorted(df[df.exp == "attention"].cols.unique()); ser = {}; col_ = []; se = series(df, "attention", "separate")
    for v in VARS[1:]:
        d = series(df, "attention", v)
        if len(d): ser[LAB[v]] = [se[c] / d[c] if c in d.index else None for c in asz]; col_.append(COL[v].lstrip("#"))
    chart(s, "line", asz, ser, .7, 1.7, 11.9, 4.3, col_, False, "speedup over separate (×)", '0.0'); lead(s, "Rows = 16 heads × sequence length, columns = sequence length. The same kernels, applied to the shapes real attention layers produce.", 6.1, 13, .8)
# 18 library + fast math
has_t = have("colsweep", "torch")
s = new(False); head(s, "Results · library and fast math" if has_t else "Results · fast math", "Against PyTorch, and the cost of __expf" if has_t else "Does a faster exponential help? expf vs __expf")
if has_t:
    rows = [[c, fmt(ms(df, "colsweep", "torch", 4096, c)), fmt(min(x for x in [ms(df, "colsweep", v, 4096, c) for v in VARS[1:5]] if x == x)), best] for c, best in [(c, LAB[S["best_variant"][c]]) for c in cols if c in S["best_variant"]]]
    table(s, .7, 1.9, 11.9, rows, ["Columns", "PyTorch (ms)", "Best of ours (ms)", "Fastest variant"], None, 12, .42)
else:
    rows = []
    for c in cols:
        a_ = df[(df.exp == "colsweep") & (df.variant == "fused_smem") & (df.cols == c)]; b_ = df[(df.exp == "colsweep") & (df.variant == "fused_fast") & (df.cols == c)]
        if len(a_) and len(b_): rows.append([c, fmt(a_.ms.iloc[0], 4), f"{a_.err.iloc[0]:.1e}", fmt(b_.ms.iloc[0], 4), f"{b_.err.iloc[0]:.1e}", f"{a_.ms.iloc[0]/b_.ms.iloc[0]:.2f}×"])
    table(s, .7, 1.9, 11.9, rows, ["Columns (4096 rows)", "expf (ms)", "expf max error", "__expf (ms)", "__expf max error", "gain"], None, 12, .42)
lead(s, next((x for x in O if "__expf" in x), ""), 5.6, 14, 1.2)
# 19 analysis
s = new(True); head(s, "Analysis", "How close to the hardware limit are we?")
dr = S["dram_fused"]; best_gbps = float(dr.gbps.max()) if len(dr) else 0
stat(s, .7, 2.0, 3.85, 2.3, f"{max(sp):.2f}×", "best baseline speedup of fused over separate (ideal ceiling ≈ 2× from traffic)")
stat(s, 4.75, 2.0, 3.85, 2.3, f"{best_gbps:.0f} GB/s", f"best fused bandwidth on DRAM-bound sizes ({best_gbps/PEAK*100:.0f}% of {PEAK:.0f} GB/s spec)", BLUE)
stat(s, 8.8, 2.0, 3.85, 2.3, f"{S['small_bw_max']:.0f} GB/s", "peak 'bandwidth' at small sizes: above DRAM speed, so it is L2-cache hits", AMB)
lead(s, "These are derived from measured times and ideal byte counts (estimates, not hardware counters). Trends such as the drop on long rows are hypotheses until profiled with Nsight Compute.", 4.7, 15, 1.3)
# 20 limitations
s = new(False); head(s, "Limitations", "What these results do and do not show")
tx(s, .7, 1.9, 11.9, 4.8, ["One GPU on one machine; synthetic random data, not real model activations.", "Each size is timed as a loop on the same data, so small matrices are served from L2; the 268 MB experiment avoids this.", "Separate-kernel component times are standalone and may not add exactly to the sequence time.", "Only 64 rows per matrix are verified against the CPU.", "The fused kernel needs the row in shared memory (≤ 12,288 columns).", "Run-to-run variance exists; read speedups as ranges."], 16, spacing=10)
# 21 future
s = new(False); head(s, "Future work", "Where the next speedup is")
for i, (t_, b_) in enumerate([("Fuse into attention", "Compute scores and softmax in one kernel so the score matrix is never written (FlashAttention-style)."), ("Longer rows", "Fallback when a row exceeds shared memory: block cooperation per row or a two-pass online scheme."), ("Profile", "Nsight Compute: DRAM throughput, achieved occupancy, register use, to confirm the explanations.")]): card(s, .7 + i * 4.05, 1.95, 3.85, 2.4, t_, b_)
# 22 conclusion
s = new(True); head(s, "Conclusion", "What we built and measured")
stat(s, .7, 2.0, 3.85, 2.3, f"{min(sp):.2f}–{max(sp):.2f}×", "fused over separate kernels on the baseline sizes")
stat(s, 4.75, 2.0, 3.85, 2.3, f"{S['col_speed'].min():.2f}–{S['col_speed'].max():.2f}×" if len(S["col_speed"]) else "-", "fused speedup across row lengths 128–8192", BLUE)
stat(s, 8.8, 2.0, 3.85, 2.3, "MATCH" if S["all_match"] else "CHECK", f"all {S['n_checked']} verified runs vs CPU (max error {S['maxerr']:.1e})", AMB)
lead(s, "Fewer memory trips make a memory-bound layer faster; the best variant depends on row length; every result is verified against the CPU.", 4.7, 16)
# screenshots
shots = sorted(glob.glob(os.path.join(ROOT, "docs/screenshots/*_labelled.png"))) or sorted(glob.glob(os.path.join(ROOT, "docs/screenshots/*.png")))
from PIL import Image
import json
LEG = json.load(open(os.path.join(ROOT, "docs/screenshots/legends.json"))) if os.path.exists(os.path.join(ROOT, "docs/screenshots/legends.json")) else {}
TT = {"01_benchmark_run_baseline_and_colsweep": "Benchmark run: baseline and row-length sweep", "02_benchmark_run_blocksweep_and_dram": "Benchmark run: block-size sweep and DRAM-bound sizes",
      "03_benchmark_run_dram_attention_done": "Benchmark run: DRAM, attention shapes, finish", "04_report_build": "Figures, README, PPT and zip generated"}
LC = [RED, AMB, TEAL, PURP]
for f in shots[:4]:
    key = os.path.basename(f).replace("_labelled.png", ""); boxed = f.replace("_labelled.png", "_boxed.png"); img = boxed if os.path.exists(boxed) else f
    s = new(True); head(s, "Evidence", TT.get(key, key)); iw, ih = Image.open(img).size; leg = LEG.get(key, [])
    if ih / iw > .7:      # tall terminal capture: image left, legend right
        sc = min(6.4 / iw, 5.3 / ih); s.shapes.add_picture(img, Inches(.7), Inches(1.65), Inches(iw * sc), Inches(ih * sc)); y = 1.7
        for i, t_ in enumerate(leg):
            box(s, 7.45, y + .02, .4, .4, LC[i % 4], shape=MSO_SHAPE.OVAL); tx(s, 7.45, y + .02, .4, .4, str(i + 1), 13, DARK, True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            tx(s, 8.0, y - .02, 4.7, 1.3, t_, 13); y += 1.25 if len(t_) > 120 else 0.95
    else:                 # wide capture: image on top, legend below
        sc = min(11.9 / iw, 2.6 / ih); s.shapes.add_picture(img, Inches(.7), Inches(1.8), Inches(iw * sc), Inches(ih * sc)); y = 1.8 + ih * sc + .5
        for i, t_ in enumerate(leg):
            box(s, .7, y + .02, .4, .4, LC[i % 4], shape=MSO_SHAPE.OVAL); tx(s, .7, y + .02, .4, .4, str(i + 1), 13, DARK, True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE); tx(s, 1.3, y, 11.3, .6, t_, 14); y += .65
# thanks
s = new(True); tx(s, .9, 2.4, 11.5, 1.2, "Thank you", 54, bold=True); tx(s, .9, 3.6, 11.5, .6, "Questions?", 28, TEAL); tx(s, .9, 4.5, 11.5, .5, "Mudit · Kartik · Manoj · Vinayak", 22)
tx(s, .9, 5.1, 11.5, .5, "Code, plan and benchmark scripts: the cuda_softmax_fusion repository.", 16, MUTE)
out = os.path.join(ROOT, "presentation", "CUDA_Softmax_Kernel_Fusion.pptx"); os.makedirs(os.path.dirname(out), exist_ok=True); P.save(out); print("PPT written:", out, "-", len(P.slides), "slides")
