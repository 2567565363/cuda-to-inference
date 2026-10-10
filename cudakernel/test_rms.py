
import torch
import inference
from torch.nn import functional as F

def reference_rmsnorm_fp64(x,weight,eps=1e-6):
    x64=x.double()
    w64=weight.double()
    rms=torch.rsqrt(torch.mean(x64*x64,dim=-1,keepdim=True)+eps)
    return x64*rms*w64

def test_case(shape,eps=1e-6):
    x=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    weight=torch.randn(shape[-1],device="cuda",dtype=torch.bfloat16)
    y_cuda=inference.RMSNorm(x,weight,eps)
    y_torch=F.rms_norm(x,(x.shape[-1],),weight=weight,eps=eps)
    y_ref=reference_rmsnorm_fp64(x,weight,eps)
    cuda_error=(y_cuda.double()-y_ref).abs()
    torch_error=(y_torch.double()-y_ref).abs()
    cuda_torch_error=(y_cuda.double()-y_torch.double()).abs()
    cuda_rel_error=cuda_error/(y_ref.abs()+1e-12)
    torch_rel_error=torch_error/(y_ref.abs()+1e-12)
    cuda_torch_rel_error=cuda_torch_error/(y_ref.abs()+1e-12)
    print(f"shape={str(shape):18s}")
    print(f"  CUDA  vs FP64  max={cuda_error.max().item():.8e} mean={cuda_error.mean().item():.8e} rel_max={cuda_rel_error.max().item():.8e} rel_mean={cuda_rel_error.mean().item():.8e}")
    print(f"  Torch vs FP64  max={torch_error.max().item():.8e} mean={torch_error.mean().item():.8e} rel_max={torch_rel_error.max().item():.8e} rel_mean={torch_rel_error.mean().item():.8e}")
    print(f"  CUDA  vs Torch  max={cuda_torch_error.max().item():.8e} mean={cuda_torch_error.mean().item():.8e} rel_max={cuda_torch_rel_error.max().item():.8e} rel_mean={cuda_torch_rel_error.mean().item():.8e}")
    print()

def main():
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    cases=[
        (1,4096),
        (8,4096),
        (32,4096),
        (1,128,4096),
        (4,128,4096),
        (1,512,4096),
    ]
    print("="*120)
    print("BF16 RMSNorm accuracy: CUDA vs PyTorch vs FP64 reference")
    print("="*120)
    for shape in cases:
        test_case(shape)
    print("="*120)

if __name__=="__main__":
    main()

