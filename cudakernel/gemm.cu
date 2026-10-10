#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
__global__ void gemm_kernel(const float* a, const float* b, float* c, int M, int N, int K)
{
    constexpr int TILE = 32;
    constexpr int THREAD_TILE = 4;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int batch = blockIdx.z;
    int tid = tidy * blockDim.x + tidx;
    const float* a_batch = a + (size_t)batch * M * K;
    const float* b_batch = b + (size_t)batch * K * N;
    float* c_batch = c + (size_t)batch * M * N;
    __shared__ float s_a[TILE][TILE];
    __shared__ float s_b[TILE][TILE];
    float local_sum[THREAD_TILE][THREAD_TILE] = {0.0f};
    int start_tidx = bidx * TILE + tidx * THREAD_TILE;
    int start_tidy = bidy * TILE + tidy * THREAD_TILE;
    int loops = (K + TILE - 1) / TILE;
    for (int i = 0; i < loops; i++)
    {
        for (int idx = tid; idx < TILE * TILE; idx += blockDim.x * blockDim.y)
        {
            int row = idx / TILE;
            int col = idx % TILE;
            int global_a_row = bidy * TILE + row;
            int global_a_col = i * TILE + col;
            int global_b_row = i * TILE + row;
            int global_b_col = bidx * TILE + col;
            s_a[row][col] = (global_a_row < M && global_a_col < K) ? a_batch[global_a_row * K + global_a_col] : 0.0f;
            s_b[row][col] = (global_b_row < K && global_b_col < N) ? b_batch[global_b_row * N + global_b_col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];
            #pragma unroll
            for (int m = 0; m < THREAD_TILE; m++)
                a_reg[m] = s_a[tidy * THREAD_TILE + m][k];
            #pragma unroll
            for (int n = 0; n < THREAD_TILE; n++)
                b_reg[n] = s_b[k][tidx * THREAD_TILE + n];
            #pragma unroll
            for (int m = 0; m < THREAD_TILE; m++)
            {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE; n++)
                    local_sum[m][n] += a_reg[m] * b_reg[n];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < THREAD_TILE; m++)
    {
        #pragma unroll
        for (int n = 0; n < THREAD_TILE; n++)
        {
            int row = start_tidy + m;
            int col = start_tidx + n;
            if (row < M && col < N)
                c_batch[row * N + col] = local_sum[m][n];
        }
    }
}

void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    int batch = a.size(0);
    int M = a.size(1);
    int K = a.size(2);
    int N = b.size(2);
    constexpr int TILE = 32;
    constexpr int THREAD_TILE = 4;
    dim3 block(TILE / THREAD_TILE, TILE / THREAD_TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    gemm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream(a.get_device())>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}