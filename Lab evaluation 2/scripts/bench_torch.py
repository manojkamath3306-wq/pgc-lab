# PyTorch library baseline: torch.softmax(dim=-1) timed with CUDA events on the same cases.
import sys, torch
out = sys.argv[1] if len(sys.argv)>1 else "results/torch.csv"
assert torch.cuda.is_available(), "CUDA not available in this PyTorch build"
cases=[("baseline",r,c) for r,c in [(4096,1024),(4096,2048),(8192,2048),(4096,4096),(2048,8192)]]
cases+=[("colsweep",4096,c) for c in [128,256,512,1024,2048,4096,8192]]
cases+=[("dram",16384,4096),("dram",32768,2048),("dram",65536,1024)]
cases+=[("attention",16*s,s) for s in [128,256,512,1024,2048]]
RUNS=30
with open(out,"w") as f:
    f.write("exp,variant,rows,cols,block,ms,gbps,err\n")
    for ex,R,C in cases:
        g=torch.Generator(device="cuda").manual_seed(1)
        x=torch.rand(R,C,device="cuda",generator=g)*20-10
        for _ in range(3): y=torch.softmax(x,dim=-1)
        torch.cuda.synchronize()
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(RUNS): y=torch.softmax(x,dim=-1)
        b.record(); torch.cuda.synchronize(); ms=a.elapsed_time(b)/RUNS
        ref=torch.softmax(x[:64].double(),dim=-1); err=(y[:64].double()-ref).abs().max().item()
        gbps=2*R*C*4/(ms*1e-3)/1e9
        f.write(f"{ex},torch,{R},{C},0,{ms:.5f},{gbps:.2f},{err:.3e}\n")
        print(f"{ex:10s} torch {R}x{C}: {ms:.4f} ms {gbps:.1f} GB/s err={err:.1e}")
