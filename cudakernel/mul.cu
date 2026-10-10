#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
template <typename scalar_t>
__global__ void mul_kernel(const scalar_t* a,const scalar_t* b,scalar_t *c,int64_t n)
{
    int64_t index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<n)
        c[index]=a[index]*b[index];
}

void mul_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c)
{
    const int64_t length=a.numel();
    int blocksize=256;
    int gridsize=(length+blocksize-1)/blocksize;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "mul_kernel",
        [&]{
            mul_kernel<scalar_t><<<gridsize,blocksize>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                length    
            );
        }
    );
}
