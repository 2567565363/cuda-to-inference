
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
constexpr int MAX_DIMS = 64;
struct BroadcastMeta
{
    int ndim;
    int64_t out_sizes[MAX_DIMS];
    int64_t a_strides[MAX_DIMS];
    int64_t b_strides[MAX_DIMS];
};
template <typename scalar_t>
__global__ void add_kernel(const scalar_t* a,const scalar_t* b,scalar_t* c,int64_t n,BroadcastMeta meta)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n)
        return;
    int64_t remaining = index;
    int64_t offset_a = 0;
    int64_t offset_b = 0;
    for (int d = meta.ndim - 1; d >= 0; --d)
    {
        int64_t coord = remaining % meta.out_sizes[d];
        remaining /= meta.out_sizes[d];
        offset_a += coord * meta.a_strides[d];
        offset_b += coord * meta.b_strides[d];
    }
    c[index] = a[offset_a] + b[offset_b];
}
void add_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor c)
{
    const int64_t n = c.numel();
    if (n == 0)
        return;
    BroadcastMeta meta{};
    meta.ndim = static_cast<int>(c.dim());
    for (int d = 0; d < meta.ndim; ++d)
    {
        meta.out_sizes[d] = c.size(d);
        int a_dim = d - (meta.ndim - static_cast<int>(a.dim()));
        int b_dim = d - (meta.ndim - static_cast<int>(b.dim()));
        if (a_dim >= 0 && a.size(a_dim) != 1)
            meta.a_strides[d] = a.stride(a_dim);
        if (b_dim >= 0 && b.size(b_dim) != 1)
            meta.b_strides[d] = b.stride(b_dim);
    }
    int blocksize = 256;
    int gridsize = static_cast<int>((n + blocksize - 1) / blocksize);
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,at::ScalarType::BFloat16,
        a.scalar_type(),
        "add_kernel",
        [&]()
        {
            add_kernel<scalar_t><<<gridsize,blocksize,0,at::cuda::getCurrentCUDAStream(a.get_device())>>>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                c.data_ptr<scalar_t>(),
                n,
                meta
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
