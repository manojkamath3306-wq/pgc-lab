// bench.cu - benchmark driver: runs every experiment, verifies against a CPU reference, writes results/data.csv
// Build: make gpu ARCH=sm_120      Run (from project root): ./bench results/data.csv
#include "kernels.cuh"

static FILE* g_csv;
static void emit(const char* ex,const char* var,int R,int C,int bs,float ms,double acc,float err,bool show){
  double gbps = acc>0 ? acc*(double)R*C*4.0/(ms*1e-3)/1e9 : 0.0;
  fprintf(g_csv,"%s,%s,%d,%d,%d,%.5f,%.2f,%.3e\n",ex,var,R,C,bs,ms,gbps,err); fflush(g_csv);
  if(show) printf("%-10s %-11s %6dx%-6d bs=%-4d %9.4f ms %8.1f GB/s  err=%.1e %s\n",ex,var,R,C,bs,ms,gbps,err,(err<1e-5f)?"MATCH":"MISMATCH");
  fflush(stdout);
}

enum { M_SEP=1, M_FUSED=2, M_ONLINE=4, M_WARP=8, M_FAST=16 };

static void runCase(const char* ex,int R,int C,std::vector<int> bsl,unsigned mask){
  size_t N=(size_t)R*C; int CH=std::min(R,CHK_ROWS);
  std::vector<float> hx(N), hy((size_t)CH*C), hr((size_t)CH*C);
  unsigned s=12345u; for(size_t i=0;i<N;i++){ s=s*1664525u+1013904223u; hx[i]=((s>>8)&0xFFFF)/65535.f*20.f-10.f; }
  for(int r=0;r<CH;r++){ double mx=-1e30,sm=0; for(int c=0;c<C;c++) mx=fmax(mx,hx[(size_t)r*C+c]);
    for(int c=0;c<C;c++) sm+=exp(hx[(size_t)r*C+c]-mx);
    for(int c=0;c<C;c++) hr[(size_t)r*C+c]=(float)(exp(hx[(size_t)r*C+c]-mx)/sm); }
  float *dx,*dy,*dm,*ds; CK(cudaMalloc(&dx,N*4)); CK(cudaMalloc(&dy,N*4)); CK(cudaMalloc(&dm,R*4)); CK(cudaMalloc(&ds,R*4));
  CK(cudaMemcpy(dx,hx.data(),N*4,cudaMemcpyHostToDevice));
  auto check=[&]()->float{ CK(cudaMemcpy(hy.data(),dy,(size_t)CH*C*4,cudaMemcpyDeviceToHost));
    float d=0; for(size_t i=0;i<hy.size();i++) d=fmaxf(d,fabsf(hy[i]-hr[i])); if(d!=d) d=1e9f; return d; };
  for(int bs:bsl){
    if(mask&M_SEP){
      auto seq=[&]{ k_max<<<R,bs>>>(dx,dm,C); k_sumexp<<<R,bs>>>(dx,dm,ds,C); k_norm<<<R,bs>>>(dx,dm,ds,dy,C); };
      CK(cudaMemset(dy,0,N*4)); float t=timeIt(seq); float e=check();
      emit(ex,"separate",R,C,bs,t,4,e,true);
      emit(ex,"sep_max",R,C,bs,timeIt([&]{k_max<<<R,bs>>>(dx,dm,C);}),1,-1,false);
      emit(ex,"sep_sum",R,C,bs,timeIt([&]{k_sumexp<<<R,bs>>>(dx,dm,ds,C);}),1,-1,false);
      emit(ex,"sep_norm",R,C,bs,timeIt([&]{k_norm<<<R,bs>>>(dx,dm,ds,dy,C);}),2,-1,false);
    }
    if((mask&M_FUSED) && (size_t)C*4<=48*1024){
      CK(cudaMemset(dy,0,N*4)); float t=timeIt([&]{ k_fused<false><<<R,bs,C*4>>>(dx,dy,C); });
      emit(ex,"fused_smem",R,C,bs,t,2,check(),true);
    }
    if(mask&M_ONLINE){
      CK(cudaMemset(dy,0,N*4)); float t=timeIt([&]{ k_online<<<R,bs>>>(dx,dy,C); });
      emit(ex,"online",R,C,bs,t,3,check(),true);
    }
    if((mask&M_WARP) && C<=1024 && C%32==0){
      int wb=128; CK(cudaMemset(dy,0,N*4)); float t=timeIt([&]{ k_warprow<<<(R+3)/4,wb>>>(dx,dy,R,C); });
      emit(ex,"warp_row",R,C,wb,t,2,check(),true);
    }
    if((mask&M_FAST) && (size_t)C*4<=48*1024){
      CK(cudaMemset(dy,0,N*4)); float t=timeIt([&]{ k_fused<true><<<R,bs,C*4>>>(dx,dy,C); });
      emit(ex,"fused_fast",R,C,bs,t,2,check(),true);
    }
  }
  cudaFree(dx);cudaFree(dy);cudaFree(dm);cudaFree(ds);
}

int main(int argc,char** argv){
  const char* path = argc>1 ? argv[1] : "results/data.csv";
  g_csv=fopen(path,"w"); if(!g_csv){ printf("cannot open %s\n",path); return 1; }
  fprintf(g_csv,"exp,variant,rows,cols,block,ms,gbps,err\n");
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int l2=0,sms=0,rt=0,drv=0; cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0); cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
  cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
  printf("GPU: %s\nSMs: %d | L2 cache: %.1f MB | compute capability %d.%d | global mem %.1f GB | CUDA runtime %d driver %d\n",
         p.name,sms,l2/1048576.0,p.major,p.minor,p.totalGlobalMem/1e9,rt,drv);
  printf("Timing: CUDA events, 3 warm-up runs, average of %d runs; check = max abs error vs CPU (double) on %d rows\n",RUNS,CHK_ROWS);
  { FILE* g=fopen("results/gpu.txt","w"); if(g){ fprintf(g,"GPU: %s\nSMs: %d\nL2 MB: %.1f\nCC: %d.%d\nMemGB: %.1f\nRuntime: %d\nDriver: %d\n",
      p.name,sms,l2/1048576.0,p.major,p.minor,p.totalGlobalMem/1e9,rt,drv); fclose(g);} }
  unsigned all=M_SEP|M_FUSED|M_ONLINE|M_WARP|M_FAST;
  // 0) baseline sizes (same as the first lab run)
  int base[][2]={{4096,1024},{4096,2048},{8192,2048},{4096,4096},{2048,8192}};
  for(auto& b:base) runCase("baseline",b[0],b[1],{256},M_SEP|M_FUSED);
  // A) row-length sweep, 4096 rows
  for(int C:{128,256,512,1024,2048,4096,8192}) runCase("colsweep",4096,C,{256},all);
  // B) block-size sweep
  for(int C:{1024,2048,4096,8192}) runCase("blocksweep",4096,C,{64,128,256,512,1024},M_SEP|M_FUSED|M_ONLINE);
  // C) DRAM-bound sizes (data far larger than L2)
  runCase("dram",16384,4096,{256},all); runCase("dram",32768,2048,{256},all); runCase("dram",65536,1024,{256},all);
  // D) attention-style: 16 heads x seq rows, seq columns
  for(int sq:{128,256,512,1024,2048}) runCase("attention",16*sq,sq,{256},all);
  fclose(g_csv); printf("Done. CSV written to %s\n",path); return 0;
}
