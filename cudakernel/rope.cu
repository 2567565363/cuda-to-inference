#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* a,scalar_t* out,const scalar_t* cos,const scalar_t* sin,int64_t heads,int64_t seq_len,int64_t hidden_size)
{
    //a[B,H,L,D]    cos/sin[B,L,D]
    int tid=threadIdx.x;
    int64_t bid=blockIdx.x;
    int64_t split=hidden_size/2;
    int64_t start=bid*hidden_size;
    int64_t batch=bid/(heads*seq_len);//计算有多少个batch
    int64_t pos=bid%seq_len;
    int64_t pos_sin=(batch*seq_len+pos)*hidden_size;
    for(int64_t i=tid;i<split;i+=blockDim.x)
    {
        int64_t a_idx=start+i;
        int64_t cs_idx=pos_sin+i;
        out[a_idx]=a[a_idx]*cos[cs_idx]-a[a_idx+split]*sin[cs_idx];
        out[a_idx+split]=a[a_idx+split]*cos[cs_idx]+a[a_idx]*sin[cs_idx];
    }
}

void rope_cuda(torch::Tensor a,torch::Tensor out,torch::Tensor cos,torch::Tensor sin)
{
    int64_t hidden_size=a.size(-1);
    int64_t seq_len=a.size(-2);
    int64_t heads=a.size(-3);
    int blocksize=64;
    int64_t gridsize=a.numel()/a.size(-1);
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "rope_kernel",
        [&]{
            rope_kernel<scalar_t><<<gridsize,blocksize>>>(
                a.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                cos.data_ptr<scalar_t>(),
                sin.data_ptr<scalar_t>(),
                heads,
                seq_len,
                hidden_size
            );
        }
    );
}