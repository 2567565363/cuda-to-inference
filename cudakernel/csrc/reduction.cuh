#include <cuda_runtime.h>
#include <cuda.h>
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}


// block 级归约: thread 0 持有整个 block 的和
// 通用: 任意 blockDim.x ≤ 1024 (不要求 2 的幂 / 32 的倍数):
//   - 满 warp: 标准 shuffle (mask 0xffffffff)
//   - 尾 partial warp (B 非 32 倍数): shuffle 读不存在的 lane 是 UB →
//     每 lane 先把值暂存 shared, 该 warp 的 lane 0 串行加 (≤31 次 FADD, 开销可忽略)
//   - 第二段: nw = ceil(B/32) 个部分和, lane ≥ nw 补 0 (旧代码 B>>5 在这里会漏读);
//     nw > 1 ⟺ B ≥ 33 ⟺ warp 0 是满 warp, 最后一步 shuffle 一定合法
//   - B 是 32 的倍数: if (B & 31) 整块统一走 else, 与旧版零差异 → B=256 基准无回退
__device__ __forceinline__ float block_reduce_sum(float v) {
    const int B    = blockDim.x;
    const int wid  = threadIdx.x >> 5;            // warp 编号 (/32)
    const int lane = threadIdx.x & 31;            // lane 编号 (%32)
    const int nw   = (B + 31) >> 5;               // warp 数
    static __shared__ float smem[64];             // [0,32) 各 warp 和; [32,64) 尾 warp 暂存

    if (B & 31) {                                 // 存在尾 warp (B 全 block 统一 → 无 barrier 分歧)
        const int M = min(32, B - 32 * wid);      // 本 warp 实际 lane 数
        if (M < 32)
            smem[32 + lane] = v;                  // partial warp: 逐 lane 暂存
        else
            v = warp_reduce_sum(v);
        __syncthreads();
        if (M < 32 && lane == 0) {
            float s = 0.0f;
            for (int i = 0; i < M; ++i) s += smem[32 + i];
            v = s;                                // 单线程串行加, ≤31 次
        }
    } else {
        v = warp_reduce_sum(v);                   // 全满 warp: 原路径
    }

    // 每个 warp 的 lane 0 把部分和写到共享内存
    if (lane == 0)
        smem[wid] = v;
    __syncthreads();

    // 第二段: warp 0 读各 warp 部分和 (lane ≥ nw 补 0), 再 warp 归约
    v = (lane < nw) ? smem[lane] : 0.0f;
    if (wid == 0 && nw > 1)
        v = warp_reduce_sum(v);

    return v;  // thread 0 持有最终结果
}

__device__ __forceinline__ float block_reduce_max(float v) {
    // 同 block_reduce_sum, 用 warp_reduce_max, 中性元换 -1e30f
    const int B    = blockDim.x;
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw   = (B + 31) >> 5;
    static __shared__ float smem[64];

    if (B & 31) {
        const int M = min(32, B - 32 * wid);
        if (M < 32)
            smem[32 + lane] = v;
        else
            v = warp_reduce_max(v);
        __syncthreads();
        if (M < 32 && lane == 0) {
            float s = -1e30f;
            for (int i = 0; i < M; ++i) s = max(s, smem[32 + i]);
            v = s;
        }
    } else {
        v = warp_reduce_max(v);
    }

    if (lane == 0)
        smem[wid] = v;
    __syncthreads();

    v = (lane < nw) ? smem[lane] : -1e30f;        // max 用极小值兜底
    if (wid == 0 && nw > 1)
        v = warp_reduce_max(v);

    return v;  // thread 0 持有最终结果
}