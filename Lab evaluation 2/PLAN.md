# Softmax and Kernel Fusion in CUDA

Parallel Computing lab evaluation, Theme 10.

## 1. Problem
Row-wise softmax over a matrix (rows x cols): out[i] = exp(x[i] - max) / sum(exp(x[j] - max)).
It is the output of every classifier and the core of attention. It does little arithmetic per byte, so it is
memory-bound and the main optimisation is **kernel fusion**: fewer trips through global memory.

## 2. Variants
| Variant | Accesses / element | Idea |
|---|---|---|
| Separate (max, sum-exp, normalise) | 4 | 3 kernels, one block per row |
| Fused (shared memory) | 2 | one kernel, row cached in shared memory |
| Online softmax | 3 | running max + rescaled sum in one pass |
| Warp-per-row | 2 | one warp per row, values in registers (cols <= 1024) |
| Fused + __expf | 2 | fast-math exponential, accuracy trade-off |
| PyTorch softmax | - | library baseline (optional) |

## 3. Experiments
1. Baseline sizes (5 matrices)  2. Row-length sweep 128..8192 (4096 rows)  3. Block-size sweep 64..1024
4. DRAM-bound sizes (268 MB matrices)  5. Attention shapes (16 heads x seq 128..2048)

## 4. Method
CUDA events, 3 warm-up runs, average of 30 runs. Output zeroed before each kernel and checked against a
double-precision CPU softmax (64 rows) before any speed is reported.

## 5. Milestones
1. Kernels + CPU reference check  2. Sweeps and CSV output  3. Figures, README, slides  4. Run on RTX 5060 Ti (WSL2), add screenshots  5. Publish repo
