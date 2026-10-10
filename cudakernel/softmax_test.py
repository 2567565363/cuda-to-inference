import torch
import inference

def benchmark(fn,x,warmup=20,iters=100):
    for _ in range(warmup):
        y=fn(x)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y=fn(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)/iters

def test(shape,dtype):
    x=torch.randn(shape,device="cuda",dtype=dtype)
    y_ref=torch.softmax(x,dim=-1)
    y=inference.softmax(x)
    max_error=(y-y_ref).abs().max().item()
    mean_error=(y-y_ref).abs().mean().item()
    sum_error=(y.sum(dim=-1)-1.0).abs().max().item()
    torch_ms=benchmark(lambda x:torch.softmax(x,dim=-1),x)
    cuda_ms=benchmark(inference.softmax,x)
    elements=x.numel()
    bytes_moved=2*elements*x.element_size()
    torch_bw=bytes_moved/(torch_ms*1e-3)/1e9
    cuda_bw=bytes_moved/(cuda_ms*1e-3)/1e9
    print(
        f"shape={tuple(shape)} "
        f"dtype={str(dtype).replace('torch.','')} "
        f"torch={torch_ms:.4f} ms "
        f"cuda={cuda_ms:.4f} ms "
        f"speedup={torch_ms/cuda_ms:.2f}x "
        f"torch_bw={torch_bw:.2f} GB/s "
        f"cuda_bw={cuda_bw:.2f} GB/s "
        f"max_err={max_error:.3e} "
        f"mean_err={mean_error:.3e} "
        f"sum_err={sum_error:.3e}"
    )

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Softmax: custom CUDA vs PyTorch")
    print()
    lengths=[128,256,512,1024,2048,4096,8192]
    rows=[1,8,32,128,512]
    dtypes=[torch.float32,torch.float16,torch.bfloat16]
    for dtype in dtypes:
        print(f"===== dtype={dtype} =====")
        for length in lengths:
            for row in rows:
                test((row,length),dtype)
        print()

if __name__=="__main__":
    main()