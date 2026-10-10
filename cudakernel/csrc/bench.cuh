//测速框架
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <numeric>
#define CHECK_CUDA_ERROR(call)                                                                          \
do{                                                                                                     \
    cudaError_t err = call;                                                                             \
    if (err != cudaSuccess)                                                                             \
    {                                                                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                                             \
    }                                                                                                   \
}while(0)

template <typename F>
std::vector<float> benchmark(F&& fn,int warmup=300,int iters=5000)
{
    // warmup: 跑到连续 3 次中位数差距 < 1%，确保温度稳定
    cudaEvent_t ws, we;
    CHECK_CUDA_ERROR(cudaEventCreate(&ws));
    CHECK_CUDA_ERROR(cudaEventCreate(&we));
    float last_median = 0;
    int stable_count = 0;
    for (int round = 0; round < warmup; ++round) {
        std::vector<float> wt(10);
        for (int i = 0; i < 10; ++i) {
            CHECK_CUDA_ERROR(cudaEventRecord(ws));
            fn();
            CHECK_CUDA_ERROR(cudaEventRecord(we));
            CHECK_CUDA_ERROR(cudaEventSynchronize(we));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&wt[i], ws, we));
        }
        std::sort(wt.begin(), wt.end());
        float ms = wt[5];
        if (round > 0 && std::abs(ms - last_median) / last_median < 0.01f) {
            stable_count++;
            if (stable_count >= 3) break;
        } else {
            stable_count = 0;
        }
        last_median = ms;
    }
    CHECK_CUDA_ERROR(cudaEventDestroy(ws));
    CHECK_CUDA_ERROR(cudaEventDestroy(we));

    // benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    std::vector<float> times(iters);   
    for(int i=0;i<iters;i++)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        fn();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        float milliseconds_item = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_item, start, stop));
        times[i] = milliseconds_item;
    }
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    std::sort(times.begin(), times.end());

    std::vector<float> results(3);
    results[0] = times[iters / 2];        //middle
    results[1] = std::accumulate(times.begin(), times.end(), 0.0f) / iters;    //avg
    float stddev = 0.0f;
    float mean = results[1];
    float sq_diff_sum = 0.0f;
    for (int i = 0; i < iters; i++)
    {
        sq_diff_sum += (times[i] - results[1]) * (times[i] - results[1]);
    }
    stddev = std::sqrt(sq_diff_sum / iters);
    results[2]= stddev;   //stddev
    return results;
}