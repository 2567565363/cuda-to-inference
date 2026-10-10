#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include "csrc/reduction.cuh"

template<typename scalar_t>
__global__ void RMSNorm_kernel(const scalar_t* a,scalar_t* b,const scalar_t* weight,float eps,int hidden)
{
    int row=blockIdx.x;
    int tid=threadIdx.x;
    int start=row*hidden;
    float sum=0.f;
    for(int j=tid;j<hidden;j+=blockDim.x)
    {
        float x=static_cast<float>(a[start+j]);
        sum+=x*x;
    }
    sum=block_reduce_sum(sum);
    __shared__ float rms;
    if(tid==0)
        rms=rsqrtf(sum/static_cast<float>(hidden)+eps);
    __syncthreads();
    for(int j=tid;j<hidden;j+=blockDim.x)
    {
        float x=static_cast<float>(a[start+j]);
        float w=static_cast<float>(weight[j]);
        b[start+j]=static_cast<scalar_t>(x*w*rms);
    }
}

template<typename scalar_t>
__global__ void RMSNorm_kernel_v2(
    const scalar_t* a,
    scalar_t* b,
    const scalar_t* weight,
    float eps,
    int hidden)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int start = row * hidden;

    float x_cache[16];
    float sum = 0.0f;

    for(int k=0,j=tid;j<hidden;k++,j+=blockDim.x)
    {
        float x = static_cast<float>(a[start+j]);
        x_cache[k] = x;
        sum += x * x;
    }

    sum = block_reduce_sum(sum);

    __shared__ float rms;

    if(tid == 0)
        rms = rsqrtf(sum / static_cast<float>(hidden) + eps);

    __syncthreads();

    for(int k=0,j=tid;j<hidden;k++,j+=blockDim.x)
    {
        float x = x_cache[k];
        float w = static_cast<float>(weight[j]);
        b[start+j] = static_cast<scalar_t>(x * w * rms);
    }
}

void RMSNorm_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor weight,float eps)
{
    int hidden=a.size(-1);
    int64_t blocksize=256;
    int64_t gridsize=a.numel()/a.size(-1);
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "RMSNorm_kernel",
        [&]{
            RMSNorm_kernel<scalar_t><<<gridsize,blocksize>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                eps,
                hidden);
        }
    );
}

//V2 引入了额外的 cache/local access，但没有获得足够的 global memory traffic reduction 收益。
void RMSNorm_cuda_v2(torch::Tensor a,torch::Tensor b,torch::Tensor weight,float eps)
{
    int hidden=a.size(-1);
    int64_t blocksize=256;
    int64_t gridsize=a.numel()/a.size(-1);
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "RMSNorm_kernel",
        [&]{
            RMSNorm_kernel_v2<scalar_t><<<gridsize,blocksize>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                eps,
                hidden);
        }
    );
}