import torch
import torch.nn.functional as F
import inference

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

EPS=1e-6

def reference_rmsnorm(x,weight,eps=EPS):
    return F.rms_norm(x,(x.size(-1),),weight=weight,eps=eps)

def benchmark(fn,x,weight,eps=EPS,warmup=50,iters=200):
    for _ in range(warmup):
        fn(x,weight,eps)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x,weight,eps)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

def test_case(shape,dtype):
    hidden=shape[-1]
    x=torch.randn(shape,device="cuda",dtype=dtype)
    weight=torch.randn(hidden,device="cuda",dtype=dtype)
    y1=inference.RMSNorm(x,weight,EPS)
    y2=inference.RMSNorm_v2(x,weight,EPS)
    ref=reference_rmsnorm(x,weight,EPS)
    max_error_v1=(y1.float()-ref.float()).abs().max().item()
    mean_error_v1=(y1.float()-ref.float()).abs().mean().item()
    max_error_v2=(y2.float()-ref.float()).abs().max().item()
    mean_error_v2=(y2.float()-ref.float()).abs().mean().item()
    max_diff=(y1.float()-y2.float()).abs().max().item()
    mean_diff=(y1.float()-y2.float()).abs().mean().item()
    t1=benchmark(inference.RMSNorm,x,weight,EPS)
    t2=benchmark(inference.RMSNorm_v2,x,weight,EPS)
    speedup=t1/t2
    rows=x.numel()/hidden
    bytes_per_row=hidden*x.element_size()*3
    bandwidth=rows*bytes_per_row/(t2*1e-3)/1e9
    print(
        f"shape={tuple(shape)} "
        f"dtype={str(dtype).replace('torch.','')} "
        f"V1={t1:.4f} ms "
        f"V2={t2:.4f} ms "
        f"speedup={speedup:.2f}x "
        f"V1_max={max_error_v1:.6e} "
        f"V1_mean={mean_error_v1:.6e} "
        f"V2_max={max_error_v2:.6e} "
        f"V2_mean={mean_error_v2:.6e} "
        f"V1_V2_max={max_diff:.6e} "
        f"V1_V2_mean={mean_diff:.6e} "
        f"V2_BW={bandwidth:.2f} GB/s"
    )

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("RMSNorm V1 vs V2")
    print("hidden=4096, block=256")
    print(f"eps={EPS}")
    print()
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for rows in [1,8,32,128,512]:
            test_case((rows,4096),dtype)

if __name__ == "__main__":
    main()