#include "dp4a_gemm.cuh"
#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <cuda_fp16.h>

#define QK_K 256
#define QK8_0 32

struct block_q8_0_dp4a {
    half d;
    int8_t qs[32];
};

struct block_q4_K_dp4a {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct block_q3_K_dp4a {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

struct block_q5_K_dp4a {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct block_q6_K_dp4a {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half d;
};

struct block_q4_0_dp4a {
    half d;
    uint8_t qs[16];
};

struct block_q5_0_dp4a {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct block_q4_1_dp4a {
    half2 dm;
    uint8_t qs[16];
};

struct block_q5_1_dp4a {
    half2 dm;
    uint8_t qh[4];
    uint8_t qs[16];
};

static bool g_dp4a_enabled = false;
static bool g_dp4a_env_checked = false;

void ggml_cuda_dp4a_set_enabled(bool enabled) {
    g_dp4a_enabled = enabled;
    g_dp4a_env_checked = true;
}

bool ggml_cuda_dp4a_is_enabled() {
    if (!g_dp4a_env_checked) {
        const char* env = std::getenv("SD_CUDA_DP4A");
        if (env && std::atoi(env) != 0) {
            g_dp4a_enabled = true;
        }
        g_dp4a_env_checked = true;
    }
    return g_dp4a_enabled;
}

bool ggml_cuda_dp4a_can_handle(int cc, const struct ggml_tensor* src0, const struct ggml_tensor* src1, const struct ggml_tensor* dst) {
    if (!ggml_cuda_dp4a_is_enabled()) return false;
    if (cc != GGML_CUDA_CC_DP4A) return false;
    if (!src0 || !src1 || !dst) return false;
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q3_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q5_0 &&
        src0->type != GGML_TYPE_Q4_1 && src0->type != GGML_TYPE_Q5_1 &&
        src0->type != GGML_TYPE_Q8_0) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    if (M < 16 || N < 16 || K < 256 || (K % 256) != 0) return false;
    return true;
}

__device__ __forceinline__ void get_scale_min_k4_dp4a(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---- One-time weight repack kernels: GGML quant -> per-row INT8 packed [N, K/4] + scale_B[N] ----

// 1 warp (32 threads) per row of B (col in 0..N-1)
__global__ void kernel_repack_q4k_to_i8(
    const block_q4_K_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (col >= N) return;

    const int nsuper = K / 256;
    const block_q4_K_dp4a* row_blk = B_blk + (size_t)col * nsuper;

    // Pass 1: find absmax across the entire row of K weights
    // Each thread handles 8 weights per superblock? Better: iterate superblocks, within superblock:
    // il = lane / 8 (0..3), ir = lane % 8 (0..7) -> 4 low + 4 high weights!
    float local_max = 0.0f;
    const int il = lane >> 3;
    const int ir = lane & 7;
    const int is = 2 * il;

    for (int s = 0; s < nsuper; ++s) {
        const block_q4_K_dp4a* blk = &row_blk[s];
        float dall = __low2half(blk->dm);
        float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        get_scale_min_k4_dp4a(is + 0, blk->scales, sc, m);
        float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4_dp4a(is + 1, blk->scales, sc, m);
        float d2 = dall * sc, m2 = dmin * m;

        const uint8_t* q = blk->qs + 32 * il + 4 * ir;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            float v0 = d1 * (float)(q[l] & 0xF) - m1;
            float v1 = d2 * (float)(q[l] >> 4)  - m2;
            local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    // Pass 2: quantize and pack 4 consecutive weights into int32_t
    // Note: 4 low weights at (4*ir .. 4*ir+3) are consecutive in K!
    // 4 high weights at (4*ir .. 4*ir+3) + 32 are consecutive in K!
    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int s = 0; s < nsuper; ++s) {
        const block_q4_K_dp4a* blk = &row_blk[s];
        float dall = __low2half(blk->dm);
        float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        get_scale_min_k4_dp4a(is + 0, blk->scales, sc, m);
        float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4_dp4a(is + 1, blk->scales, sc, m);
        float d2 = dall * sc, m2 = dmin * m;

        const uint8_t* q = blk->qs + 32 * il + 4 * ir;
        uint32_t p0 = 0, p1 = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int8_t i0 = (int8_t)__float2int_rn((d1 * (float)(q[l] & 0xF) - m1) * inv_scale);
            int8_t i1 = (int8_t)__float2int_rn((d2 * (float)(q[l] >> 4)  - m2) * inv_scale);
            p0 |= ((uint32_t)(uint8_t)i0) << (l * 8);
            p1 |= ((uint32_t)(uint8_t)i1) << (l * 8);
        }
        int base_vec = s * 64 + il * 16 + ir;
        out_row[base_vec + 0] = (int32_t)p0;
        out_row[base_vec + 8] = (int32_t)p1;
    }
}

// 1 warp (32 threads) per row of B for Q3_K (64 threads -> 2 warps, or 32 threads x 8 weights)
__global__ void kernel_repack_q3k_to_i8(
    const block_q3_K_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (col >= N) return;

    const int nsuper = K / 256;
    const block_q3_K_dp4a* row_blk = B_blk + (size_t)col * nsuper;

    // Each lane handles 2 groups of 4 consecutive weights: n=0 and n=1?
    // In dequantize_q3_K (64 threads): r=tid/4, t=r/2, is0=r%2, l0=16*is0+4*(tid%4), n=t/4, j=t-4*n
    // With 32 threads, each thread runs tid0 = lane and tid1 = lane + 32!
    float local_max = 0.0f;
    for (int s = 0; s < nsuper; ++s) {
        const block_q3_K_dp4a* blk = &row_blk[s];
        float d_all = __half2float(blk->d);
        #pragma unroll
        for (int half_t = 0; half_t < 2; ++half_t) {
            int tid = lane + half_t * 32;
            int r = tid / 4, t = r / 2, is0 = r % 2;
            int l0 = 16 * is0 + 4 * (tid % 4);
            int n = t / 4, j = t - 4 * n;
            uint8_t m = 1 << (4 * n + j);
            int is = 8 * n + 2 * j + is0;
            int shift = 2 * j;
            int8_t us = is <  4 ? (blk->scales[is-0] & 0xF) | (((blk->scales[is+8] >> 0) & 3) << 4) :
                        is <  8 ? (blk->scales[is-0] & 0xF) | (((blk->scales[is+4] >> 2) & 3) << 4) :
                        is < 12 ? (blk->scales[is-8] >>  4) | (((blk->scales[is+0] >> 4) & 3) << 4) :
                                  (blk->scales[is-8] >>  4) | (((blk->scales[is-4] >> 6) & 3) << 4);
            float dl = d_all * (us - 32);
            const uint8_t* q = blk->qs + 32 * n;
            const uint8_t* hm = blk->hmask;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int idx = l0 + l;
                float v = dl * (float)((int8_t)((q[idx] >> shift) & 3) - ((hm[idx] & m) ? 0 : 4));
                local_max = fmaxf(local_max, fabsf(v));
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int s = 0; s < nsuper; ++s) {
        const block_q3_K_dp4a* blk = &row_blk[s];
        float d_all = __half2float(blk->d);
        #pragma unroll
        for (int half_t = 0; half_t < 2; ++half_t) {
            int tid = lane + half_t * 32;
            int r = tid / 4, t = r / 2, is0 = r % 2;
            int l0 = 16 * is0 + 4 * (tid % 4);
            int n = t / 4, j = t - 4 * n;
            uint8_t m = 1 << (4 * n + j);
            int is = 8 * n + 2 * j + is0;
            int shift = 2 * j;
            int8_t us = is <  4 ? (blk->scales[is-0] & 0xF) | (((blk->scales[is+8] >> 0) & 3) << 4) :
                        is <  8 ? (blk->scales[is-0] & 0xF) | (((blk->scales[is+4] >> 2) & 3) << 4) :
                        is < 12 ? (blk->scales[is-8] >>  4) | (((blk->scales[is+0] >> 4) & 3) << 4) :
                                  (blk->scales[is-8] >>  4) | (((blk->scales[is-4] >> 6) & 3) << 4);
            float dl = d_all * (us - 32);
            const uint8_t* q = blk->qs + 32 * n;
            const uint8_t* hm = blk->hmask;
            uint32_t p = 0;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int idx = l0 + l;
                float v = dl * (float)((int8_t)((q[idx] >> shift) & 3) - ((hm[idx] & m) ? 0 : 4));
                int8_t iv = (int8_t)__float2int_rn(v * inv_scale);
                p |= ((uint32_t)(uint8_t)iv) << (l * 8);
            }
            int out_k = s * 256 + 128 * n + 32 * j + l0;
            out_row[out_k >> 2] = (int32_t)p;
        }
    }
}

__global__ void kernel_repack_q8_0_to_i8(
    const block_q8_0_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (col >= N) return;

    const int nblk = K / 32;
    const block_q8_0_dp4a* row_blk = B_blk + (size_t)col * nblk;

    float local_max = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        local_max = fmaxf(local_max, fabsf(d * (float)row_blk[b].qs[lane]));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        int32_t qv = __float2int_rn((d * (float)row_blk[b].qs[lane]) * inv_scale);
        int32_t q0 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 0);
        int32_t q1 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 1);
        int32_t q2 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 2);
        int32_t q3 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 3);
        if ((lane & 3) == 0) {
            uint32_t p = ((uint32_t)(uint8_t)(int8_t)q0)
                       | ((uint32_t)(uint8_t)(int8_t)q1 << 8)
                       | ((uint32_t)(uint8_t)(int8_t)q2 << 16)
                       | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
            out_row[b * 8 + (lane >> 2)] = (int32_t)p;
        }
    }
}

__device__ __forceinline__ int32_t dp4a_warp_pack4_i8(int32_t qv, int lane) {
    int32_t q0 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 0);
    int32_t q1 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 1);
    int32_t q2 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 2);
    int32_t q3 = __shfl_sync(0xffffffff, qv, (lane & ~3) + 3);
    uint32_t p = ((uint32_t)(uint8_t)(int8_t)q0)
               | ((uint32_t)(uint8_t)(int8_t)q1 << 8)
               | ((uint32_t)(uint8_t)(int8_t)q2 << 16)
               | ((uint32_t)(uint8_t)(int8_t)q3 << 24);
    return (int32_t)p;
}

__global__ void kernel_repack_q5k_to_i8(
    const block_q5_K_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (col >= N) return;

    const int nsuper = K / 256;
    const block_q5_K_dp4a* row_blk = B_blk + (size_t)col * nsuper;

    float local_max = 0.0f;
    const int il = lane >> 3;
    const int ir = lane & 7;
    const int is = 2 * il;
    const uint8_t hm0 = (uint8_t)(1 << (2 * il));
    const uint8_t hm1 = (uint8_t)(hm0 << 1);

    for (int s = 0; s < nsuper; ++s) {
        const block_q5_K_dp4a* blk = &row_blk[s];
        float dall = __low2half(blk->dm);
        float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        get_scale_min_k4_dp4a(is + 0, blk->scales, sc, m);
        float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4_dp4a(is + 1, blk->scales, sc, m);
        float d2 = dall * sc, m2 = dmin * m;

        const uint8_t* ql = blk->qs + 32 * il + 4 * ir;
        const uint8_t* qh = blk->qh + 4 * ir;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            float v0 = d1 * (float)((ql[l] & 0xF) + ((qh[l] & hm0) ? 16 : 0)) - m1;
            float v1 = d2 * (float)((ql[l] >> 4)  + ((qh[l] & hm1) ? 16 : 0)) - m2;
            local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int s = 0; s < nsuper; ++s) {
        const block_q5_K_dp4a* blk = &row_blk[s];
        float dall = __low2half(blk->dm);
        float dmin = __high2half(blk->dm);
        uint8_t sc, m;
        get_scale_min_k4_dp4a(is + 0, blk->scales, sc, m);
        float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4_dp4a(is + 1, blk->scales, sc, m);
        float d2 = dall * sc, m2 = dmin * m;

        const uint8_t* ql = blk->qs + 32 * il + 4 * ir;
        const uint8_t* qh = blk->qh + 4 * ir;
        uint32_t p0 = 0, p1 = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int8_t i0 = (int8_t)__float2int_rn((d1 * (float)((ql[l] & 0xF) + ((qh[l] & hm0) ? 16 : 0)) - m1) * inv_scale);
            int8_t i1 = (int8_t)__float2int_rn((d2 * (float)((ql[l] >> 4)  + ((qh[l] & hm1) ? 16 : 0)) - m2) * inv_scale);
            p0 |= ((uint32_t)(uint8_t)i0) << (l * 8);
            p1 |= ((uint32_t)(uint8_t)i1) << (l * 8);
        }
        int base_vec = s * 64 + il * 16 + ir;
        out_row[base_vec + 0] = (int32_t)p0;
        out_row[base_vec + 8] = (int32_t)p1;
    }
}

__global__ void kernel_repack_q6k_to_i8(
    const block_q6_K_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x; // 0..31
    if (col >= N) return;

    const int nsuper = K / 256;
    const block_q6_K_dp4a* row_blk = B_blk + (size_t)col * nsuper;

    float local_max = 0.0f;
    for (int s = 0; s < nsuper; ++s) {
        const block_q6_K_dp4a* blk = &row_blk[s];
        float d = __half2float(blk->d);
        #pragma unroll
        for (int ip = 0; ip < 2; ++ip) {
            const int is = 8 * ip + (lane >> 4);
            const uint8_t* ql = blk->ql + 64 * ip + lane;
            const uint8_t  qh = blk->qh[32 * ip + lane];
            const int8_t*  sc = blk->scales + is;

            float v0 = d * (float)sc[0] * (float)((int8_t)((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
            float v1 = d * (float)sc[2] * (float)((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            float v2 = d * (float)sc[4] * (float)((int8_t)((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            float v3 = d * (float)sc[6] * (float)((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
            local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int s = 0; s < nsuper; ++s) {
        const block_q6_K_dp4a* blk = &row_blk[s];
        float d = __half2float(blk->d);
        #pragma unroll
        for (int ip = 0; ip < 2; ++ip) {
            const int is = 8 * ip + (lane >> 4);
            const uint8_t* ql = blk->ql + 64 * ip + lane;
            const uint8_t  qh = blk->qh[32 * ip + lane];
            const int8_t*  sc = blk->scales + is;

            int32_t q0 = __float2int_rn((d * (float)sc[0] * (float)((int8_t)((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32)) * inv_scale);
            int32_t q1 = __float2int_rn((d * (float)sc[2] * (float)((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32)) * inv_scale);
            int32_t q2 = __float2int_rn((d * (float)sc[4] * (float)((int8_t)((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32)) * inv_scale);
            int32_t q3 = __float2int_rn((d * (float)sc[6] * (float)((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32)) * inv_scale);

            int32_t p0 = dp4a_warp_pack4_i8(q0, lane);
            int32_t p1 = dp4a_warp_pack4_i8(q1, lane);
            int32_t p2 = dp4a_warp_pack4_i8(q2, lane);
            int32_t p3 = dp4a_warp_pack4_i8(q3, lane);
            if ((lane & 3) == 0) {
                int base = s * 64 + ip * 32 + (lane >> 2);
                out_row[base +  0] = p0;
                out_row[base +  8] = p1;
                out_row[base + 16] = p2;
                out_row[base + 24] = p3;
            }
        }
    }
}

__global__ void kernel_repack_q4_0_to_i8(
    const block_q4_0_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int nblk = K / 32;
    const block_q4_0_dp4a* row_blk = B_blk + (size_t)col * nblk;
    const int idx = lane & 15;

    float local_max = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16) ? (byte & 0xF) : (byte >> 4);
        local_max = fmaxf(local_max, fabsf(d * (float)(q - 8)));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16) ? (byte & 0xF) : (byte >> 4);
        int32_t qv = __float2int_rn((d * (float)(q - 8)) * inv_scale);
        int32_t p = dp4a_warp_pack4_i8(qv, lane);
        if ((lane & 3) == 0) {
            out_row[b * 8 + (lane >> 2)] = p;
        }
    }
}

__global__ void kernel_repack_q5_0_to_i8(
    const block_q5_0_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int nblk = K / 32;
    const block_q5_0_dp4a* row_blk = B_blk + (size_t)col * nblk;
    const int idx = lane & 15;

    float local_max = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        uint32_t qh = (uint32_t)row_blk[b].qh[0]
                    | ((uint32_t)row_blk[b].qh[1] << 8)
                    | ((uint32_t)row_blk[b].qh[2] << 16)
                    | ((uint32_t)row_blk[b].qh[3] << 24);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16)
            ? ((byte & 0xF) | (((qh >> idx) << 4) & 0x10))
            : ((byte >> 4)  | ((qh >> (idx + 12)) & 0x10));
        local_max = fmaxf(local_max, fabsf(d * (float)(q - 16)));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int b = 0; b < nblk; ++b) {
        float d = __half2float(row_blk[b].d);
        uint32_t qh = (uint32_t)row_blk[b].qh[0]
                    | ((uint32_t)row_blk[b].qh[1] << 8)
                    | ((uint32_t)row_blk[b].qh[2] << 16)
                    | ((uint32_t)row_blk[b].qh[3] << 24);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16)
            ? ((byte & 0xF) | (((qh >> idx) << 4) & 0x10))
            : ((byte >> 4)  | ((qh >> (idx + 12)) & 0x10));
        int32_t qv = __float2int_rn((d * (float)(q - 16)) * inv_scale);
        int32_t p = dp4a_warp_pack4_i8(qv, lane);
        if ((lane & 3) == 0) {
            out_row[b * 8 + (lane >> 2)] = p;
        }
    }
}

__global__ void kernel_repack_q4_1_to_i8(
    const block_q4_1_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int nblk = K / 32;
    const block_q4_1_dp4a* row_blk = B_blk + (size_t)col * nblk;
    const int idx = lane & 15;

    float local_max = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        float d = __low2half(row_blk[b].dm);
        float m = __high2half(row_blk[b].dm);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16) ? (byte & 0xF) : (byte >> 4);
        local_max = fmaxf(local_max, fabsf(d * (float)q + m));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int b = 0; b < nblk; ++b) {
        float d = __low2half(row_blk[b].dm);
        float m = __high2half(row_blk[b].dm);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16) ? (byte & 0xF) : (byte >> 4);
        int32_t qv = __float2int_rn((d * (float)q + m) * inv_scale);
        int32_t p = dp4a_warp_pack4_i8(qv, lane);
        if ((lane & 3) == 0) {
            out_row[b * 8 + (lane >> 2)] = p;
        }
    }
}

__global__ void kernel_repack_q5_1_to_i8(
    const block_q5_1_dp4a* __restrict__ B_blk,
    int32_t* __restrict__ B_q,
    float* __restrict__ scale_B,
    int N, int K)
{
    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (col >= N) return;

    const int nblk = K / 32;
    const block_q5_1_dp4a* row_blk = B_blk + (size_t)col * nblk;
    const int idx = lane & 15;

    float local_max = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        float d = __low2half(row_blk[b].dm);
        float m = __high2half(row_blk[b].dm);
        uint32_t qh = (uint32_t)row_blk[b].qh[0]
                    | ((uint32_t)row_blk[b].qh[1] << 8)
                    | ((uint32_t)row_blk[b].qh[2] << 16)
                    | ((uint32_t)row_blk[b].qh[3] << 24);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16)
            ? ((byte & 0xF) | (((qh >> idx) << 4) & 0x10))
            : ((byte >> 4)  | ((qh >> (idx + 12)) & 0x10));
        local_max = fmaxf(local_max, fabsf(d * (float)q + m));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    float scale = (local_max == 0.0f) ? 1.0f : (local_max / 127.0f);
    float inv_scale = (local_max == 0.0f) ? 0.0f : (127.0f / local_max);
    if (lane == 0) scale_B[col] = scale;

    int32_t* out_row = B_q + (size_t)col * (K / 4);
    for (int b = 0; b < nblk; ++b) {
        float d = __low2half(row_blk[b].dm);
        float m = __high2half(row_blk[b].dm);
        uint32_t qh = (uint32_t)row_blk[b].qh[0]
                    | ((uint32_t)row_blk[b].qh[1] << 8)
                    | ((uint32_t)row_blk[b].qh[2] << 16)
                    | ((uint32_t)row_blk[b].qh[3] << 24);
        uint8_t byte = row_blk[b].qs[idx];
        int q = (lane < 16)
            ? ((byte & 0xF) | (((qh >> idx) << 4) & 0x10))
            : ((byte >> 4)  | ((qh >> (idx + 12)) & 0x10));
        int32_t qv = __float2int_rn((d * (float)q + m) * inv_scale);
        int32_t p = dp4a_warp_pack4_i8(qv, lane);
        if ((lane & 3) == 0) {
            out_row[b * 8 + (lane >> 2)] = p;
        }
    }
}

// ---- Fast float4 vectorized per-row activation quantizer ----
__global__ void kernel_quant_row_f32_to_i8_vec4(
    const float* __restrict__ A,
    int32_t* __restrict__ A_q,
    float* __restrict__ scale_A,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x; // 256 threads
    const int K4 = K / 4;
    const float4* row_f4 = reinterpret_cast<const float4*>(A + (size_t)row * K);

    float local_max = 0.0f;
    for (int i = tid; i < K4; i += blockDim.x) {
        float4 v = row_f4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));
    }

    __shared__ float s_max[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) s_max[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float mx = (lane < 8) ? s_max[lane] : 0.0f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if (lane == 0) {
            float sc = (mx == 0.0f) ? 1.0f : (mx / 127.0f);
            scale_A[row] = sc;
            s_max[0] = (mx == 0.0f) ? 0.0f : (127.0f / mx);
        }
    }
    __syncthreads();

    const float inv = s_max[0];
    int32_t* out_row = A_q + (size_t)row * K4;
    for (int i = tid; i < K4; i += blockDim.x) {
        float4 v = row_f4[i];
        int8_t b0 = (int8_t)__float2int_rn(v.x * inv);
        int8_t b1 = (int8_t)__float2int_rn(v.y * inv);
        int8_t b2 = (int8_t)__float2int_rn(v.z * inv);
        int8_t b3 = (int8_t)__float2int_rn(v.w * inv);
        uint32_t p = ((uint32_t)(uint8_t)b0)
                   | ((uint32_t)(uint8_t)b1 << 8)
                   | ((uint32_t)(uint8_t)b2 << 16)
                   | ((uint32_t)(uint8_t)b3 << 24);
        out_row[i] = (int32_t)p;
    }
}

// ---- High-throughput 128x128 W8A8 DP4A GEMM (LDS.128 + register prefetch) ----
__global__ void __launch_bounds__(256, 2)
kernel_w8a8_dp4a_128x128(
    const int32_t* __restrict__ A_q,
    const float*   __restrict__ scale_A,
    const int32_t* __restrict__ B_q,
    const float*   __restrict__ scale_B,
    float*         __restrict__ C,
    int M, int N, int K)
{
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row_base = blockIdx.y * 128;
    const int col_base = blockIdx.x * 128;

    __shared__ int32_t s_A[16][128 + 4];
    __shared__ int32_t s_B[16][128 + 4];

    int32_t acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0;

    const int K_vec = K / 4;
    const int r_load = tid >> 2;
    const int k4 = (tid & 3) * 4;

    int4 va0 = (row_base + r_load < M)      ? *reinterpret_cast<const int4*>(&A_q[(size_t)(row_base + r_load) * K_vec + k4]) : make_int4(0,0,0,0);
    int4 va1 = (row_base + r_load + 64 < M) ? *reinterpret_cast<const int4*>(&A_q[(size_t)(row_base + r_load + 64) * K_vec + k4]) : make_int4(0,0,0,0);
    int4 vb0 = (col_base + r_load < N)      ? *reinterpret_cast<const int4*>(&B_q[(size_t)(col_base + r_load) * K_vec + k4]) : make_int4(0,0,0,0);
    int4 vb1 = (col_base + r_load + 64 < N) ? *reinterpret_cast<const int4*>(&B_q[(size_t)(col_base + r_load + 64) * K_vec + k4]) : make_int4(0,0,0,0);

    for (int kv = 0; kv < K_vec; kv += 16) {
        s_A[k4+0][r_load] = va0.x; s_A[k4+1][r_load] = va0.y; s_A[k4+2][r_load] = va0.z; s_A[k4+3][r_load] = va0.w;
        s_A[k4+0][r_load+64] = va1.x; s_A[k4+1][r_load+64] = va1.y; s_A[k4+2][r_load+64] = va1.z; s_A[k4+3][r_load+64] = va1.w;
        s_B[k4+0][r_load] = vb0.x; s_B[k4+1][r_load] = vb0.y; s_B[k4+2][r_load] = vb0.z; s_B[k4+3][r_load] = vb0.w;
        s_B[k4+0][r_load+64] = vb1.x; s_B[k4+1][r_load+64] = vb1.y; s_B[k4+2][r_load+64] = vb1.z; s_B[k4+3][r_load+64] = vb1.w;
        __syncthreads();

        if (kv + 16 < K_vec) {
            int nkv = kv + 16;
            va0 = (row_base + r_load < M)      ? *reinterpret_cast<const int4*>(&A_q[(size_t)(row_base + r_load) * K_vec + nkv + k4]) : make_int4(0,0,0,0);
            va1 = (row_base + r_load + 64 < M) ? *reinterpret_cast<const int4*>(&A_q[(size_t)(row_base + r_load + 64) * K_vec + nkv + k4]) : make_int4(0,0,0,0);
            vb0 = (col_base + r_load < N)      ? *reinterpret_cast<const int4*>(&B_q[(size_t)(col_base + r_load) * K_vec + nkv + k4]) : make_int4(0,0,0,0);
            vb1 = (col_base + r_load + 64 < N) ? *reinterpret_cast<const int4*>(&B_q[(size_t)(col_base + r_load + 64) * K_vec + nkv + k4]) : make_int4(0,0,0,0);
        }

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            int4 a0 = *reinterpret_cast<const int4*>(&s_A[k][ty * 4]);
            int4 a1 = *reinterpret_cast<const int4*>(&s_A[k][ty * 4 + 64]);
            int4 b0 = *reinterpret_cast<const int4*>(&s_B[k][tx * 4]);
            int4 b1 = *reinterpret_cast<const int4*>(&s_B[k][tx * 4 + 64]);
            int32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            int32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    float ad[8], bd[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r0 = row_base + ty * 4 + i, r1 = r0 + 64;
        ad[i]   = (r0 < M) ? scale_A[r0] : 0.0f;
        ad[i+4] = (r1 < M) ? scale_A[r1] : 0.0f;
        int c0 = col_base + tx * 4 + i, c1 = c0 + 64;
        bd[i]   = (c0 < N) ? scale_B[c0] : 0.0f;
        bd[i+4] = (c1 < N) ? scale_B[c1] : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row_base + (i < 4 ? ty * 4 + i : ty * 4 + 64 + (i - 4));
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = col_base + (j < 4 ? tx * 4 + j : tx * 4 + 64 + (j - 4));
                if (c < N) C[(size_t)r * N + c] = (float)acc[i][j] * (ad[i] * bd[j]);
            }
        }
    }
}

// ---- Weight cache & activation workspace ----
struct CachedWeightI8 {
    int32_t* B_q = nullptr;
    float* scale_B = nullptr;
    int64_t N = 0;
    int64_t K = 0;
    int type = 0;
};

#define DP4A_MAX_DEVICES 16

struct Dp4aDeviceState {
    std::unordered_map<const void*, CachedWeightI8> weight_cache;
    bool   cache_disabled = false;
    size_t cache_bytes_used = 0;

    int32_t* sp_Bq = nullptr;
    float*   sp_scaleB = nullptr;
    size_t   sp_Bq_cap = 0;
    size_t   sp_scaleB_cap = 0;

    int32_t* ws_Aq = nullptr;
    float*   ws_scaleA = nullptr;
    size_t   ws_Aq_cap = 0;
    size_t   ws_scaleA_cap = 0;
};

static std::mutex g_dp4a_mutex;
static Dp4aDeviceState g_state[DP4A_MAX_DEVICES];

static bool   g_cache_limit_checked = false;
static size_t g_cache_limit_bytes = SIZE_MAX;

static void dp4a_clear_device_state_locked(int device) {
    if (device < 0 || device >= DP4A_MAX_DEVICES) return;
    Dp4aDeviceState& st = g_state[device];
    const bool has_work = !st.weight_cache.empty() || st.cache_disabled || st.cache_bytes_used > 0 ||
                          st.sp_Bq != nullptr || st.sp_scaleB != nullptr ||
                          st.ws_Aq != nullptr || st.ws_scaleA != nullptr;
    if (!has_work) return;

    int prev_dev = 0;
    cudaError_t prev_err = cudaGetDevice(&prev_dev);
    if (prev_err != cudaSuccess || prev_dev != device) {
        cudaSetDevice(device);
    }

    for (auto& kv : st.weight_cache) {
        if (kv.second.B_q) cudaFree(kv.second.B_q);
        if (kv.second.scale_B) cudaFree(kv.second.scale_B);
    }
    st.weight_cache.clear();
    st.cache_disabled = false;
    st.cache_bytes_used = 0;

    if (st.sp_Bq)     { cudaFree(st.sp_Bq);     st.sp_Bq = nullptr; }
    if (st.sp_scaleB) { cudaFree(st.sp_scaleB); st.sp_scaleB = nullptr; }
    st.sp_Bq_cap = 0;
    st.sp_scaleB_cap = 0;

    if (st.ws_Aq)     { cudaFree(st.ws_Aq);     st.ws_Aq = nullptr; }
    if (st.ws_scaleA) { cudaFree(st.ws_scaleA); st.ws_scaleA = nullptr; }
    st.ws_Aq_cap = 0;
    st.ws_scaleA_cap = 0;

    if (prev_err == cudaSuccess && prev_dev != device) {
        cudaSetDevice(prev_dev);
    }
}

void ggml_cuda_dp4a_clear_device_cache(int device) {
    std::lock_guard<std::mutex> lock(g_dp4a_mutex);
    dp4a_clear_device_state_locked(device);
}

void ggml_cuda_dp4a_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_dp4a_mutex);
    for (int d = 0; d < DP4A_MAX_DEVICES; ++d) {
        dp4a_clear_device_state_locked(d);
    }
}

static size_t dp4a_get_cache_limit_bytes() {
    if (!g_cache_limit_checked) {
        const char* sp_env = std::getenv("SD_CUDA_DP4A_SCRATCHPAD");
        if (sp_env && std::atoi(sp_env) != 0) {
            g_cache_limit_bytes = 0;
        } else {
            const char* mb_env = std::getenv("SD_CUDA_DP4A_CACHE_MB");
            if (mb_env) {
                long long mb = std::atoll(mb_env);
                g_cache_limit_bytes = (mb <= 0) ? 0ULL : ((size_t)mb << 20);
            }
        }
        g_cache_limit_checked = true;
    }
    return g_cache_limit_bytes;
}

static void dp4a_launch_repack(const ggml_tensor* src0, int32_t* B_q, float* scale_B, int N, int K, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((unsigned int)((N + 7) / 8));
    if (src0->type == GGML_TYPE_Q4_K) {
        kernel_repack_q4k_to_i8<<<grid, block, 0, stream>>>(
            (const block_q4_K_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q3_K) {
        kernel_repack_q3k_to_i8<<<grid, block, 0, stream>>>(
            (const block_q3_K_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q5_K) {
        kernel_repack_q5k_to_i8<<<grid, block, 0, stream>>>(
            (const block_q5_K_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q6_K) {
        kernel_repack_q6k_to_i8<<<grid, block, 0, stream>>>(
            (const block_q6_K_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q4_0) {
        kernel_repack_q4_0_to_i8<<<grid, block, 0, stream>>>(
            (const block_q4_0_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q5_0) {
        kernel_repack_q5_0_to_i8<<<grid, block, 0, stream>>>(
            (const block_q5_0_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q4_1) {
        kernel_repack_q4_1_to_i8<<<grid, block, 0, stream>>>(
            (const block_q4_1_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q5_1) {
        kernel_repack_q5_1_to_i8<<<grid, block, 0, stream>>>(
            (const block_q5_1_dp4a*)src0->data, B_q, scale_B, N, K);
    } else if (src0->type == GGML_TYPE_Q8_0) {
        kernel_repack_q8_0_to_i8<<<grid, block, 0, stream>>>(
            (const block_q8_0_dp4a*)src0->data, B_q, scale_B, N, K);
    }
}

bool ggml_cuda_dp4a_mul_mat(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst, cudaStream_t stream) {
    const int64_t M = src1->ne[1];
    const int64_t N = src0->ne[1];
    const int64_t K = src0->ne[0];

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= DP4A_MAX_DEVICES) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_dp4a_mutex);
    Dp4aDeviceState& st = g_state[dev];

    const size_t need_Bq = (size_t)N * (K / 4) * sizeof(int32_t);
    const size_t need_sB = (size_t)N * sizeof(float);
    const size_t need_bytes = need_Bq + need_sB;

    int32_t* B_q_ptr = nullptr;
    float*   scale_B_ptr = nullptr;

    // Level 1: Persistent W8 Weight Cache
    auto it = st.weight_cache.find(src0->data);
    if (it != st.weight_cache.end() && it->second.N == N && it->second.K == K && it->second.type == (int)src0->type) {
        B_q_ptr = it->second.B_q;
        scale_B_ptr = it->second.scale_B;
    } else {
        const size_t cache_limit = dp4a_get_cache_limit_bytes();
        bool can_cache = !st.cache_disabled && (cache_limit > 0) &&
                         (st.cache_bytes_used + need_bytes <= cache_limit);

        if (can_cache) {
            size_t free_mem = 0, total_mem = 0;
            // Reserve 1536 MiB headroom so execution_engine stays in 1 segment (avoids multi-segment PCIe offload) and VAE fits safely
            if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess || free_mem < need_bytes + (1536ULL << 20)) {
                st.cache_disabled = true;
                can_cache = false;
            }
        }

        if (can_cache) {
            if (it != st.weight_cache.end()) {
                st.cache_bytes_used -= (size_t)it->second.N * (it->second.K / 4) * sizeof(int32_t) + (size_t)it->second.N * sizeof(float);
                if (it->second.B_q) cudaFree(it->second.B_q);
                if (it->second.scale_B) cudaFree(it->second.scale_B);
                st.weight_cache.erase(it);
            }

            CachedWeightI8 entry;
            entry.N = N;
            entry.K = K;
            entry.type = (int)src0->type;
            if (cudaMalloc(&entry.B_q, need_Bq) == cudaSuccess &&
                cudaMalloc(&entry.scale_B, need_sB) == cudaSuccess) {
                dp4a_launch_repack(src0, entry.B_q, entry.scale_B, (int)N, (int)K, stream);
                st.weight_cache[src0->data] = entry;
                st.cache_bytes_used += need_bytes;
                B_q_ptr = entry.B_q;
                scale_B_ptr = entry.scale_B;
            } else {
                if (entry.B_q) cudaFree(entry.B_q);
                if (entry.scale_B) cudaFree(entry.scale_B);
                st.cache_disabled = true;
            }
        }

        // Level 2: On-the-fly Single-Layer Reusable Scratchpad Buffer (never fall back to slow mmq unless OOM)
        if (!B_q_ptr) {
            if (need_Bq > st.sp_Bq_cap) {
                if (st.sp_Bq) { cudaFree(st.sp_Bq); st.sp_Bq = nullptr; st.sp_Bq_cap = 0; }
                if (cudaMalloc(&st.sp_Bq, need_Bq) != cudaSuccess) { st.sp_Bq = nullptr; st.sp_Bq_cap = 0; return false; }
                st.sp_Bq_cap = need_Bq;
            }
            if (need_sB > st.sp_scaleB_cap) {
                if (st.sp_scaleB) { cudaFree(st.sp_scaleB); st.sp_scaleB = nullptr; st.sp_scaleB_cap = 0; }
                if (cudaMalloc(&st.sp_scaleB, need_sB) != cudaSuccess) { st.sp_scaleB = nullptr; st.sp_scaleB_cap = 0; return false; }
                st.sp_scaleB_cap = need_sB;
            }
            dp4a_launch_repack(src0, st.sp_Bq, st.sp_scaleB, (int)N, (int)K, stream);
            B_q_ptr = st.sp_Bq;
            scale_B_ptr = st.sp_scaleB;
        }
    }

    // 2. Ensure activation workspace
    size_t need_Aq = (size_t)M * (K / 4) * sizeof(int32_t);
    size_t need_sA = (size_t)M * sizeof(float);
    if (need_Aq > st.ws_Aq_cap) {
        if (st.ws_Aq) cudaFree(st.ws_Aq);
        if (cudaMalloc(&st.ws_Aq, need_Aq) != cudaSuccess) { st.ws_Aq = nullptr; st.ws_Aq_cap = 0; return false; }
        st.ws_Aq_cap = need_Aq;
    }
    if (need_sA > st.ws_scaleA_cap) {
        if (st.ws_scaleA) cudaFree(st.ws_scaleA);
        if (cudaMalloc(&st.ws_scaleA, need_sA) != cudaSuccess) { st.ws_scaleA = nullptr; st.ws_scaleA_cap = 0; return false; }
        st.ws_scaleA_cap = need_sA;
    }

    // 3. Quantize activations A [M, K] -> A_q [M, K/4] + scale_A [M]
    kernel_quant_row_f32_to_i8_vec4<<<(int)M, 256, 0, stream>>>(
        (const float*)src1->data, st.ws_Aq, st.ws_scaleA, (int)K);

    // 4. Launch 128x128 DP4A GEMM
    dim3 block(16, 16);
    dim3 grid((unsigned int)((N + 127) / 128), (unsigned int)((M + 127) / 128));
    kernel_w8a8_dp4a_128x128<<<grid, block, 0, stream>>>(
        st.ws_Aq, st.ws_scaleA, B_q_ptr, scale_B_ptr, (float*)dst->data, (int)M, (int)N, (int)K);

    return true;
}

#include "fattn_dp4a.cuh"

bool ggml_cuda_dp4a_flash_attn_ext(const struct ggml_tensor* dst, cudaStream_t stream) {
    return ggml_cuda_dp4a_flash_attn_ext_impl(dst, stream);
}
