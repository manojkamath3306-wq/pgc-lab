// kernels.cuh - all softmax kernels (separate, fused, online, warp-per-row, fast-math) and the timing helper
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA error: %s (line %d)\n",cudaGetErrorString(e_),__LINE__);exit(1);}}while(0)
#define CHK_ROWS 64          // rows verified against the CPU reference
#define RUNS 30              // timed runs (after 3 warm-up runs)

template<bool FAST> __device__ __forceinline__ float EXPF(float x){ return FAST ? __expf(x) : expf(x); }
__device__ __forceinline__ float warpMax(float v){ for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o)); return v; }
__device__ __forceinline__ float warpSum(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); return v; }
__device__ float blockMax(float v){            // result broadcast to every thread
  __shared__ float s[32]; __shared__ float r;
  int l=threadIdx.x&31, w=threadIdx.x>>5;
  v=warpMax(v); if(l==0) s[w]=v; __syncthreads();
  if(w==0){ v=(l<(int)(blockDim.x>>5))?s[l]:-INFINITY; v=warpMax(v); if(l==0) r=v; }
  __syncthreads(); float out=r; __syncthreads(); return out;
}
__device__ float blockSum(float v){
  __shared__ float s[32]; __shared__ float r;
  int l=threadIdx.x&31, w=threadIdx.x>>5;
  v=warpSum(v); if(l==0) s[w]=v; __syncthreads();
  if(w==0){ v=(l<(int)(blockDim.x>>5))?s[l]:0.f; v=warpSum(v); if(l==0) r=v; }
  __syncthreads(); float out=r; __syncthreads(); return out;
}

// ---------- V1: three separate kernels (one block per row) ----------
__global__ void k_max(const float* x,float* m,int cols){
  const float* row=x+(size_t)blockIdx.x*cols; float v=-INFINITY;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x) v=fmaxf(v,row[i]);
  v=blockMax(v); if(threadIdx.x==0) m[blockIdx.x]=v;
}
__global__ void k_sumexp(const float* x,const float* m,float* s,int cols){
  const float* row=x+(size_t)blockIdx.x*cols; float mx=m[blockIdx.x], v=0.f;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x) v+=expf(row[i]-mx);
  v=blockSum(v); if(threadIdx.x==0) s[blockIdx.x]=v;
}
__global__ void k_norm(const float* x,const float* m,const float* s,float* y,int cols){
  size_t base=(size_t)blockIdx.x*cols; float mx=m[blockIdx.x], inv=1.f/s[blockIdx.x];
  for(int i=threadIdx.x;i<cols;i+=blockDim.x) y[base+i]=expf(x[base+i]-mx)*inv;
}
// ---------- V2: fused, row cached in shared memory (1 read + 1 write) ----------
template<bool FAST> __global__ void k_fused(const float* x,float* y,int cols){
  extern __shared__ float row[];
  size_t base=(size_t)blockIdx.x*cols; float mx=-INFINITY;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x){ float v=x[base+i]; row[i]=v; mx=fmaxf(mx,v); }
  mx=blockMax(mx);
  float sum=0.f;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x){ float e=EXPF<FAST>(row[i]-mx); row[i]=e; sum+=e; }
  sum=blockSum(sum); float inv=1.f/sum;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x) y[base+i]=row[i]*inv;
}
// ---------- V3: online softmax (running max + rescaled sum in ONE pass, then write) ----------
__global__ void k_online(const float* x,float* y,int cols){
  size_t base=(size_t)blockIdx.x*cols; float m=-INFINITY, d=0.f;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x){
    float v=x[base+i];
    if(v>m){ d=d*expf(m-v)+1.f; m=v; } else d+=expf(v-m);
  }
  float M=blockMax(m);
  float dd=(m==-INFINITY)?0.f:d*expf(m-M);
  float D=blockSum(dd); float inv=1.f/D;
  for(int i=threadIdx.x;i<cols;i+=blockDim.x) y[base+i]=expf(x[base+i]-M)*inv;
}
// ---------- V4: one warp per row, row kept in registers (cols<=1024, multiple of 32) ----------
__global__ void k_warprow(const float* x,float* y,int R,int cols){
  int warp=threadIdx.x>>5, lane=threadIdx.x&31;
  int r=blockIdx.x*(blockDim.x>>5)+warp; if(r>=R) return;
  const float* xr=x+(size_t)r*cols; float* yr=y+(size_t)r*cols;
  float v[32]; int n=cols>>5; float mx=-INFINITY;
  #pragma unroll
  for(int k=0;k<32;k++){ v[k]=0.f; if(k<n){ v[k]=xr[lane+32*k]; mx=fmaxf(mx,v[k]); } }
  mx=__shfl_sync(0xffffffff,warpMax(mx),0);
  float s=0.f;
  #pragma unroll
  for(int k=0;k<32;k++){ if(k<n){ v[k]=expf(v[k]-mx); s+=v[k]; } }
  s=__shfl_sync(0xffffffff,warpSum(s),0); float inv=1.f/s;
  #pragma unroll
  for(int k=0;k<32;k++){ if(k<n) yr[lane+32*k]=v[k]*inv; }
}

template<class F> static float timeIt(F f){
  for(int i=0;i<3;i++) f();                         // warm-up
  CK(cudaDeviceSynchronize());
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); for(int i=0;i<RUNS;i++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
  CK(cudaGetLastError()); float ms; cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b); return ms/RUNS;   // average of RUNS runs
}

