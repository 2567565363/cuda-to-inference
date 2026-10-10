#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include "csrc/reduction.cuh"
#include <cfloat>

template<typename scalar_t>
__global__ void softmax_kernel(const scalar_t* in,scalar_t* out,int length)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int64_t start=static_cast<int64_t>(bid)*length;
    int64_t end=start+length;
    float local_max=-FLT_MAX;
    for(int64_t i=start+tid;i<end;i+=blockDim.x)
    {
        float x=static_cast<float>(in[i]);//转换为float进行计算
        local_max=max(local_max,x);
    }
    local_max=block_reduce_max(local_max);
    __shared__ float row_max;
    if(tid==0)
        row_max=local_max;
    __syncthreads();
    float local_sum=0.0f;
    for(int64_t i=start+tid;i<end;i+=blockDim.x)
    {
        float x=static_cast<float>(in[i]);
        local_sum+=expf(x-row_max);
    }
    local_sum=block_reduce_sum(local_sum);
    __shared__ float row_sum;
    if(tid==0)
        row_sum=local_sum;
    __syncthreads();
    for(int64_t i=start+tid;i<end;i+=blockDim.x)
    {
        float x=static_cast<float>(in[i]);
        float y=expf(x-row_max)/row_sum;
        out[i]=static_cast<scalar_t>(y);
    }
}




void softmax_cuda(torch::Tensor a,torch::Tensor b)
{
    int length=a.size(-1);//获取最后一个维度的长度
    int64_t blocksize=256;
    int64_t gridsize=a.numel()/a.size(-1);

    
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "softmax_kernel",
        [&]{
            softmax_kernel<scalar_t><<<gridsize,blocksize>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            length
        );
        }
    );

}