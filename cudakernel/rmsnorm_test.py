import torch
import inference
from torch.nn import functional as F
def reference_rmsnorm(x, weight=None, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
def benchmark(fn,warmup=50,iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/iters

def test_case(shape,dtype,eps=1e-6):
    x=torch.randn(shape,device="cuda",dtype=dtype)
    weight=torch.randn(shape[-1],device="cuda",dtype=dtype)
    y_ref=reference_rmsnorm(x,weight,eps)
    y=inference.RMSNorm(x,weight,eps)
    max_error=(y.float()-y_ref.float()).abs().max().item()
    mean_error=(y.float()-y_ref.float()).abs().mean().item()
    custom_ms=benchmark(lambda: inference.RMSNorm(x,weight,eps))
    torch_ms=benchmark(lambda: reference_rmsnorm(x,weight,eps))
    speedup=torch_ms/custom_ms
    print(f"shape={str(shape):18s} dtype={str(dtype):12s} "
          f"max_error={max_error:.6e} mean_error={mean_error:.6e} "
          f"custom={custom_ms:.4f} ms torch={torch_ms:.4f} ms "
          f"speedup={speedup:.2f}x")

def main():
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cases=[
        ((1,4096),torch.float32),
        ((8,4096),torch.float32),
        ((32,4096),torch.float32),
        ((1,4096),torch.float16),
        ((8,4096),torch.float16),
        ((32,4096),torch.float16),
        ((1,4096),torch.bfloat16),
        ((8,4096),torch.bfloat16),
        ((32,4096),torch.bfloat16),
        ((1,128,4096),torch.bfloat16),
        ((4,128,4096),torch.bfloat16),
        ((1,512,4096),torch.bfloat16),
    ]
    print("="*110)
    print("RMSNorm correctness + end-to-end benchmark")
    print("="*110)
    for shape,dtype in cases:
        test_case(shape,dtype)
    print("="*110)

if __name__=="__main__":
    main()