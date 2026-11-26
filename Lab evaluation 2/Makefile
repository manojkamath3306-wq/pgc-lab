NVCC    ?= nvcc
# GPU compute capability: RTX 50xx = sm_120, RTX 40xx = sm_89, RTX 30xx = sm_86, T4 = sm_75, A100 = sm_80
ARCH    ?= native
NVFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: gpu
gpu: bench
bench: src/bench.cu src/kernels.cuh
	$(NVCC) $(NVFLAGS) src/bench.cu -o $@

clean:
	rm -f bench
