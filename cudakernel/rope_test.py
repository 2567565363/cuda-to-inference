import torch
import inference

def rotate_half(x):
    x1=x[..., :x.shape[-1]//2]
    x2=x[..., x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

def rope_reference(x,cos,sin):
    cos=cos.unsqueeze(1)
    sin=sin.unsqueeze(1)
    return x*cos+rotate_half(x)*sin

def make_cos_sin(batch,seq_len,head_dim,dtype,device):
    base=1000000.0
    inv_freq=1.0/(base**(torch.arange(0,head_dim,2,device=device,dtype=torch.float32)/head_dim))
    position_ids=torch.arange(seq_len,device=device,dtype=torch.float32).unsqueeze(0).expand(batch,-1)
    freqs=torch.einsum("bs,d->bsd",position_ids,inv_freq)
    emb=torch.cat((freqs,freqs),dim=-1)
    cos=emb.cos().to(dtype)
    sin=emb.sin().to(dtype)
    return cos,sin

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

def test_rope(batch,heads,seq_len,head_dim,dtype):
    device="cuda"
    x=torch.randn(batch,heads,seq_len,head_dim,device=device,dtype=dtype)
    cos,sin=make_cos_sin(batch,seq_len,head_dim,dtype,device)
    ref=rope_reference(x,cos,sin)
    out=inference.rope(x,cos,sin)
    diff=(out-ref).abs()
    ref_time=benchmark(lambda:rope_reference(x,cos,sin))
    cuda_time=benchmark(lambda:inference.rope(x,cos,sin))
    bytes_per_element=torch.tensor([],dtype=dtype).element_size()
    elements=x.numel()
    total_bytes=elements*bytes_per_element*3
    ref_bw=total_bytes/(ref_time*1e-3)/1e9
    cuda_bw=total_bytes/(cuda_time*1e-3)/1e9
    print(f"B={batch}, H={heads}, S={seq_len}, D={head_dim}, dtype={dtype}")
    print(f"max_abs_error={diff.max().item():.8e}")
    print(f"mean_abs_error={diff.mean().item():.8e}")
    print(f"PyTorch: {ref_time:.4f} ms, {ref_bw:.2f} GB/s")
    print(f"CUDA:    {cuda_time:.4f} ms, {cuda_bw:.2f} GB/s")
    print(f"speedup: {ref_time/cuda_time:.2f}x")
    print("PASS" if torch.allclose(out,ref,rtol=1e-2,atol=1e-2) else "FAIL")
    print()

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for batch in [1,2]:
            for heads in [32,8]:
                for seq_len in [1,7,64,128,512,2048]:
                    test_rope(batch,heads,seq_len,128,dtype)

if __name__=="__main__":
    main()