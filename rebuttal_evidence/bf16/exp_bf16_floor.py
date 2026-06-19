"""BF16 accuracy vs the BF16 quantization floor (rebuttal §3).
Fair reference: a BF16 nonlinear unit's output cannot beat BF16 round-to-nearest.
We measure XDA-BF16's mean relative error (LUT entries + output rounded to BF16)
against that floor, per function, at the 512-entry budget. Result: XDA-BF16 is
within 1.5-3.2x of the floor across all 9 functions -> near BF16-optimal.
Reproduce: PYTHONPATH=<research_root> python exp_bf16_floor.py
"""
import torch, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for exp_bf16_preliminary; XDA.eda.nli_eda resolved via PYTHONPATH=<research_root>
from XDA.eda.nli_eda import get_function, get_domain
from exp_bf16_preliminary import get_bf16_exponent_bins, optimize_bf16
dev='cuda' if torch.cuda.is_available() else 'cpu'; TAU=2.0**-14
def b16(t): return t.to(torch.bfloat16).to(torch.float32)
def bin_err(func,b0,b1,K,mant=7):
    T=max(mant-K,0); n=min(max(2**K*32,256),4096)
    x=torch.linspace(b0,b1,n,device=dev,dtype=torch.float32); y=func(x)
    den=torch.clamp(y.abs(),min=TAU)
    floor=((b16(y)-y).abs()/den)                       # BF16 quantization floor
    cps=torch.linspace(b0,b1,2**K+1,device=dev); yc=b16(func(cps))   # LUT entries in BF16
    idx=torch.clamp(torch.searchsorted(cps,x)-1,0,2**K-1)
    x0=cps[idx];x1=cps[idx+1];y0=yc[idx];y1=yc[idx+1]; dx=x1-x0
    t=torch.where(dx>0,(x-x0)/dx,torch.zeros_like(dx)); s=float(1<<T); tq=torch.floor(t*s)/s
    yp=b16(y0+tq*(y1-y0))                               # output in BF16
    xda=((yp-y).abs()/den)
    return floor.sum().item(),xda.sum().item(),n
FUNCS=['silu','gelu','exp','sigmoid','tanh','hardswish','mish','rsqrt','reciprocal']
print(f"{'func':>11} | {'BF16 floor':>10} | {'XDA-BF16':>10} | ratio(XDA/floor)")
for f in FUNCS:
    func=get_function(f); bins,ka,_=optimize_bf16(f,max_lut=512,device=dev)
    fs=xs=ns=0
    for (b0,b1,sgn,e),k in zip(bins,ka):
        a,b,n=bin_err(func,b0,b1,k); fs+=a; xs+=b; ns+=n
    print(f"{f:>11} | {fs/ns:>10.5f} | {xs/ns:>10.5f} | {xs/fs:>6.2f}x")
