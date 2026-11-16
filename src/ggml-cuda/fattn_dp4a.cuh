#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include "ggml.h"

namespace {

__device__ __forceinline__ int32_t fa_pack_i8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int32_t)(uint8_t)a) |
           (((int32_t)(uint8_t)b) << 8) |
           (((int32_t)(uint8_t)c) << 16) |
           (((int32_t)(uint8_t)d) << 24);
}

__device__ __forceinline__ int8_t fa_quant_i8(float x, float inv_sc) {
    int v = __float2int_rn(x * inv_sc);
    v = v > 127 ? 127 : (v < -127 ? -127 : v);
    return (int8_t)v;
}

template <int D, bool K_IS_F16, bool V_IS_F16, bool HAS_MASK>
__global__ void __launch_bounds__(256, 2) kernel_flash_attn_dp4a(
        const char* __restrict__ Q_ptr,
        const char* __restrict__ K_ptr,
        const char* __restrict__ V_ptr,
        const char* __restrict__ mask_ptr,
        float* __restrict__ dst_ptr,
        const float attn_scale,
        const int N_q,
        const int N_kv,
        const int n_heads,
        const int gqa_ratio,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb11, const size_t nb12, const size_t nb13,
        const size_t nb21, const size_t nb22, const size_t nb23,
        const int mask_ne1, const int mask_ne2, const int mask_ne3,
        const size_t mask_nb1, const size_t mask_nb2, const size_t mask_nb3) {

    static_assert(D == 64 || D == 128, "kernel_flash_attn_dp4a supports D=64 and D=128");

    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int D4 = D / 4;                  // 16 (D=64) or 32 (D=128)
    constexpr int QK_STRIDE = D4 + 1;          // 17 or 33
    constexpr int THREADS_PER_ROW = D / 32;    // 2 (D=64) or 4 (D=128)
    constexpr int ELEMS_PER_LANE  = D / 32;    // 2 (D=64) or 4 (D=128)
    constexpr int KV_I8_WORDS = (Br * QK_STRIDE > D * 17) ? (Br * QK_STRIDE) : (D * 17);

    const int q_block = blockIdx.x;
    const int head    = blockIdx.y;
    const int batch   = blockIdx.z;
    const int kv_head = head / gqa_ratio;

    const int q_start = q_block * Br;
    if (q_start >= N_q) return;

    const int tid     = threadIdx.y * 32 + threadIdx.x; // 0..255
    const int warp_id = threadIdx.y;                    // 0..7
    const int lane    = threadIdx.x;                    // 0..31

    __shared__ int32_t Q_i8[Br][QK_STRIDE];
    __shared__ int32_t KV_i8[KV_I8_WORDS];
    __shared__ float   S_f[Br][Bc + 1];                 // 64 * 65 * 4 = 16640 B (reused as half V_smem[64][D])
    __shared__ int32_t P_i8[Br][17];
    __shared__ float   Q_scale[Br];
    __shared__ float   K_scale[Bc];
    __shared__ float   V_scale[D];
    __shared__ float   P_scale[Br];

    int32_t (*K_i8)[QK_STRIDE] = (int32_t (*)[QK_STRIDE])KV_i8;
    int32_t (*V_packed)[17]    = (int32_t (*)[17])KV_i8;
    half    (*V_smem)[D]       = (half (*)[D])S_f;

    // -------------------------------------------------------------------------
    // 1. Load and quantize Q tile [Br=64, D] -> Q_i8[64][QK_STRIDE], Q_scale[64]
    // -------------------------------------------------------------------------
    {
        const int row    = tid / THREADS_PER_ROW;
        const int lane_r = tid & (THREADS_PER_ROW - 1);
        const int d_off  = lane_r * 32;
        const int q_idx  = q_start + row;

        if (row < Br) {
            float q_vals[32];
            float amax = 0.0f;
            if (q_idx < N_q) {
                const float* q_row = (const float*)(Q_ptr + (size_t)batch * nb03 + (size_t)head * nb02 + (size_t)q_idx * nb01) + d_off;
                #pragma unroll
                for (int i = 0; i < 32; i += 4) {
                    float4 v4 = *(const float4*)(q_row + i);
                    q_vals[i + 0] = v4.x;
                    q_vals[i + 1] = v4.y;
                    q_vals[i + 2] = v4.z;
                    q_vals[i + 3] = v4.w;
                    amax = fmaxf(amax, fabsf(v4.x));
                    amax = fmaxf(amax, fabsf(v4.y));
                    amax = fmaxf(amax, fabsf(v4.z));
                    amax = fmaxf(amax, fabsf(v4.w));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; ++i) q_vals[i] = 0.0f;
            }

            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 32));
            if constexpr (THREADS_PER_ROW == 4) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 32));
            }

            const float sc     = amax > 0.0f ? (amax / 127.0f) : 0.0f;
            const float inv_sc = amax > 0.0f ? (127.0f / amax) : 0.0f;
            if (lane_r == 0) Q_scale[row] = sc;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int idx = i * 4;
                Q_i8[row][lane_r * 8 + i] = fa_pack_i8x4(
                    fa_quant_i8(q_vals[idx + 0], inv_sc),
                    fa_quant_i8(q_vals[idx + 1], inv_sc),
                    fa_quant_i8(q_vals[idx + 2], inv_sc),
                    fa_quant_i8(q_vals[idx + 3], inv_sc));
            }
        }
    }

    // Per-warp online softmax state & output accumulators
    // Each warp owns 8 Q rows (r_base = warp_id * 8), each lane owns ELEMS_PER_LANE D columns
    const int r_base = warp_id * 8;
    const int d0     = lane * ELEMS_PER_LANE;

    float m_old[8];
    float l_old[8];
    float O_acc[8][ELEMS_PER_LANE];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        m_old[r] = -FLT_MAX;
        l_old[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < ELEMS_PER_LANE; ++c) {
            O_acc[r][c] = 0.0f;
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. Main loop over K/V blocks [kv_start .. kv_start + Bc)
    // -------------------------------------------------------------------------
    for (int kv_start = 0; kv_start < N_kv; kv_start += Bc) {
        // 2a. Load & quantize K [Bc=64, D] -> K_i8[64][QK_STRIDE], K_scale[64]
        {
            const int row    = tid / THREADS_PER_ROW;
            const int lane_r = tid & (THREADS_PER_ROW - 1);
            const int d_off  = lane_r * 32;
            const int k_idx  = kv_start + row;

            if (row < Bc) {
                float k_vals[32];
                float amax = 0.0f;
                if (k_idx < N_kv) {
                    const char* k_row_ptr = K_ptr + (size_t)batch * nb13 + (size_t)kv_head * nb12 + (size_t)k_idx * nb11;
                    if constexpr (K_IS_F16) {
                        const half2* k_h2 = (const half2*)((const half*)k_row_ptr + d_off);
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            float2 f2 = __half22float2(k_h2[i]);
                            k_vals[2 * i + 0] = f2.x;
                            k_vals[2 * i + 1] = f2.y;
                            amax = fmaxf(amax, fabsf(f2.x));
                            amax = fmaxf(amax, fabsf(f2.y));
                        }
                    } else {
                        const float* k_f = (const float*)k_row_ptr + d_off;
                        #pragma unroll
                        for (int i = 0; i < 32; i += 4) {
                            float4 v4 = *(const float4*)(k_f + i);
                            k_vals[i + 0] = v4.x;
                            k_vals[i + 1] = v4.y;
                            k_vals[i + 2] = v4.z;
                            k_vals[i + 3] = v4.w;
                            amax = fmaxf(amax, fabsf(v4.x));
                            amax = fmaxf(amax, fabsf(v4.y));
                            amax = fmaxf(amax, fabsf(v4.z));
                            amax = fmaxf(amax, fabsf(v4.w));
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) k_vals[i] = 0.0f;
                }

                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 32));
                if constexpr (THREADS_PER_ROW == 4) {
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 32));
                }

                const float sc     = amax > 0.0f ? (amax / 127.0f) : 0.0f;
                const float inv_sc = amax > 0.0f ? (127.0f / amax) : 0.0f;
                if (lane_r == 0) K_scale[row] = sc;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx = i * 4;
                    K_i8[row][lane_r * 8 + i] = fa_pack_i8x4(
                        fa_quant_i8(k_vals[idx + 0], inv_sc),
                        fa_quant_i8(k_vals[idx + 1], inv_sc),
                        fa_quant_i8(k_vals[idx + 2], inv_sc),
                        fa_quant_i8(k_vals[idx + 3], inv_sc));
                }
            }
        }
        __syncthreads();

        // 2b. S = Q_i8 * K_i8^T [64 x 64] via __dp4a
        {
            const int ty = tid >> 4; // 0..15 -> rows ty*4 .. ty*4+3
            const int tx = tid & 15; // 0..15 -> cols tx*4 .. tx*4+3
            const int r0 = ty * 4;
            const int c0 = tx * 4;

            int32_t acc[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = 0;

            #pragma unroll 8
            for (int k4 = 0; k4 < D4; ++k4) {
                int32_t q[4];
                int32_t k[4];
                q[0] = Q_i8[r0 + 0][k4];
                q[1] = Q_i8[r0 + 1][k4];
                q[2] = Q_i8[r0 + 2][k4];
                q[3] = Q_i8[r0 + 3][k4];

                k[0] = K_i8[c0 + 0][k4];
                k[1] = K_i8[c0 + 1][k4];
                k[2] = K_i8[c0 + 2][k4];
                k[3] = K_i8[c0 + 3][k4];

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] = __dp4a(q[i], k[j], acc[i][j]);
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int qr = q_start + r0 + i;
                const float qs = Q_scale[r0 + i] * attn_scale;
                const half* m_row = nullptr;
                if constexpr (HAS_MASK) {
                    if (qr < N_q) {
                        m_row = (const half*)(mask_ptr + (size_t)(batch % mask_ne3) * mask_nb3
                                                       + (size_t)(head % mask_ne2) * mask_nb2
                                                       + (size_t)(qr % mask_ne1) * mask_nb1);
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kc = kv_start + c0 + j;
                    if (kc < N_kv) {
                        float s = (float)acc[i][j] * (qs * K_scale[c0 + j]);
                        if constexpr (HAS_MASK) {
                            if (m_row) s += __half2float(m_row[kc]);
                        }
                        S_f[r0 + i][c0 + j] = s;
                    } else {
                        S_f[r0 + i][c0 + j] = -FLT_MAX;
                    }
                }
            }
        }
        __syncthreads();

        // 2c. Online softmax in pure FP32 & quantize P [64 x 64] -> P_i8[64][17]
        {
            const int c_even = lane * 2;
            const bool valid0 = (kv_start + c_even + 0) < N_kv;
            const bool valid1 = (kv_start + c_even + 1) < N_kv;

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = r_base + r;
                float s0 = S_f[row][c_even + 0];
                float s1 = S_f[row][c_even + 1];

                float m_ij = fmaxf(s0, s1);
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    m_ij = fmaxf(m_ij, __shfl_xor_sync(0xffffffff, m_ij, mask, 32));
                }

                const float m_new = fmaxf(m_old[r], m_ij);
                const float alpha = (m_old[r] == -FLT_MAX) ? 0.0f : expf(m_old[r] - m_new);

                const float p0 = valid0 ? expf(s0 - m_new) : 0.0f;
                const float p1 = valid1 ? expf(s1 - m_new) : 0.0f;

                float p_sum = p0 + p1;
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    p_sum += __shfl_xor_sync(0xffffffff, p_sum, mask, 32);
                }

                l_old[r] = l_old[r] * alpha + p_sum;
                m_old[r] = m_new;

                #pragma unroll
                for (int c = 0; c < ELEMS_PER_LANE; ++c) {
                    O_acc[r][c] *= alpha;
                }

                const float p_max = expf(m_ij - m_new);
                const float p_sc  = p_max > 0.0f ? (p_max / 127.0f) : 0.0f;
                const float inv_p = p_max > 0.0f ? (127.0f / p_max) : 0.0f;
                if (lane == 0) P_scale[row] = p_sc;

                const uint8_t q0 = (uint8_t)fa_quant_i8(p0, inv_p);
                const uint8_t q1 = (uint8_t)fa_quant_i8(p1, inv_p);
                const int32_t pair_lo = (int32_t)q0 | (((int32_t)q1) << 8);
                const int32_t pair_hi = __shfl_xor_sync(0xffffffff, pair_lo, 1, 32);
                if ((lane & 1) == 0) {
                    P_i8[row][lane >> 1] = pair_lo | (pair_hi << 16);
                }
            }
        }
        __syncthreads();

        // 2d. Load V [Bc=64, D] coalesced -> V_smem[64][D] (half),
        //     then transpose-quantize per-column -> V_packed[D][17], V_scale[D]
        {
            const int row    = tid / THREADS_PER_ROW;
            const int lane_r = tid & (THREADS_PER_ROW - 1);
            const int d_off  = lane_r * 32;
            const int v_idx  = kv_start + row;

            if (row < Bc) {
                if (v_idx < N_kv) {
                    const char* v_row_ptr = V_ptr + (size_t)batch * nb23 + (size_t)kv_head * nb22 + (size_t)v_idx * nb21;
                    if constexpr (V_IS_F16) {
                        const uint4* src_u4 = (const uint4*)((const half*)v_row_ptr + d_off);
                        uint4* dst_u4 = (uint4*)(&V_smem[row][d_off]);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            dst_u4[i] = src_u4[i];
                        }
                    } else {
                        const float* v_f = (const float*)v_row_ptr + d_off;
                        half2* dst_h2 = (half2*)(&V_smem[row][d_off]);
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            dst_h2[i] = __floats2half2_rn(v_f[2 * i + 0], v_f[2 * i + 1]);
                        }
                    }
                } else {
                    uint4 zero4 = make_uint4(0, 0, 0, 0);
                    uint4* dst_u4 = (uint4*)(&V_smem[row][d_off]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) dst_u4[i] = zero4;
                }
            }
        }
        __syncthreads();

        // 2 threads per D column: d = tid / 2, half_t = tid % 2 (tokens 0..31 or 32..63)
        {
            const int d      = tid >> 1;
            const int half_t = tid & 1;
            const int j_base = half_t * 32;

            if (d < D) {
                float v_col[32];
                float amax = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    float val = __half2float(V_smem[j_base + j][d]);
                    v_col[j] = val;
                    amax = fmaxf(amax, fabsf(val));
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 32));

                const float sc     = amax > 0.0f ? (amax / 127.0f) : 0.0f;
                const float inv_sc = amax > 0.0f ? (127.0f / amax) : 0.0f;
                if (half_t == 0) V_scale[d] = sc;

                #pragma unroll
                for (int j4 = 0; j4 < 8; ++j4) {
                    const int idx = j4 * 4;
                    V_packed[d][half_t * 8 + j4] = fa_pack_i8x4(
                        fa_quant_i8(v_col[idx + 0], inv_sc),
                        fa_quant_i8(v_col[idx + 1], inv_sc),
                        fa_quant_i8(v_col[idx + 2], inv_sc),
                        fa_quant_i8(v_col[idx + 3], inv_sc));
                }
            }
        }
        __syncthreads();

        // 2e. O_acc += P_i8 * V_packed via __dp4a
        // Warp handles rows r_base..r_base+7, lane handles cols d0..d0+ELEMS_PER_LANE-1
        {
            int32_t pv_acc[8][ELEMS_PER_LANE];
            #pragma unroll
            for (int r = 0; r < 8; ++r)
                #pragma unroll
                for (int c = 0; c < ELEMS_PER_LANE; ++c)
                    pv_acc[r][c] = 0;

            #pragma unroll 4
            for (int j4 = 0; j4 < 16; ++j4) {
                int32_t p_reg[8];
                int32_t v_reg[ELEMS_PER_LANE];
                #pragma unroll
                for (int r = 0; r < 8; ++r) p_reg[r] = P_i8[r_base + r][j4];
                #pragma unroll
                for (int c = 0; c < ELEMS_PER_LANE; ++c) v_reg[c] = V_packed[d0 + c][j4];

                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    #pragma unroll
                    for (int c = 0; c < ELEMS_PER_LANE; ++c) {
                        pv_acc[r][c] = __dp4a(p_reg[r], v_reg[c], pv_acc[r][c]);
                    }
                }
            }

            float vs[ELEMS_PER_LANE];
            #pragma unroll
            for (int c = 0; c < ELEMS_PER_LANE; ++c) {
                vs[c] = V_scale[d0 + c];
            }

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float ps = P_scale[r_base + r];
                #pragma unroll
                for (int c = 0; c < ELEMS_PER_LANE; ++c) {
                    O_acc[r][c] += (float)pv_acc[r][c] * (ps * vs[c]);
                }
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // 3. Finalize O = O_acc / l_old and write dst [batch, N_q, n_heads, D]
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int qr = q_start + r_base + r;
        if (qr < N_q) {
            const float inv_l = (l_old[r] > 0.0f) ? (1.0f / l_old[r]) : 0.0f;
            const size_t dst_idx = (((size_t)batch * N_q + qr) * n_heads + head) * D + d0;
            if constexpr (ELEMS_PER_LANE == 4) {
                float4 out4;
                out4.x = O_acc[r][0] * inv_l;
                out4.y = O_acc[r][1] * inv_l;
                out4.z = O_acc[r][2] * inv_l;
                out4.w = O_acc[r][3] * inv_l;
                *(float4*)(dst_ptr + dst_idx) = out4;
            } else {
                float2 out2;
                out2.x = O_acc[r][0] * inv_l;
                out2.y = O_acc[r][1] * inv_l;
                *(float2*)(dst_ptr + dst_idx) = out2;
            }
        }
    }
}

} // namespace

static bool ggml_cuda_dp4a_flash_attn_ext_impl(const struct ggml_tensor* dst, cudaStream_t stream) {
    const ggml_tensor* Q     = dst->src[0];
    const ggml_tensor* K     = dst->src[1];
    const ggml_tensor* V     = dst->src[2];
    const ggml_tensor* mask  = dst->src[3];
    const ggml_tensor* sinks = dst->src[4];

    if (sinks != nullptr) return false;
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if ((K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) ||
        (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32)) return false;
    if (mask != nullptr && mask->type != GGML_TYPE_F16) return false;

    const int head_dim = (int)Q->ne[0];
    if ((head_dim != 128 && head_dim != 64) ||
        K->ne[0] != head_dim || V->ne[0] != head_dim || dst->ne[0] != head_dim) return false;
    if (Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) return false;
    if (((uintptr_t)Q->data & 15) || ((uintptr_t)dst->data & 15)) return false;
    if ((Q->nb[1] & 15) || (K->nb[1] & 15) || (V->nb[1] & 15)) return false;
    if (((uintptr_t)K->data & 15) || ((uintptr_t)V->data & 15)) return false;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float*)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float*)dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float*)dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) return false;

    const int N_q     = (int)Q->ne[1];
    const int N_kv    = (int)K->ne[1];
    const int n_heads = (int)Q->ne[2];
    const int batch   = (int)Q->ne[3];
    if (K->ne[2] <= 0 || n_heads % (int)K->ne[2] != 0) return false;
    const int gqa_ratio = n_heads / (int)K->ne[2];

    const char* mask_ptr = mask ? (const char*)mask->data : nullptr;
    const int mask_ne1 = mask ? (int)mask->ne[1] : 1;
    const int mask_ne2 = mask ? (int)mask->ne[2] : 1;
    const int mask_ne3 = mask ? (int)mask->ne[3] : 1;
    const size_t mask_nb1 = mask ? mask->nb[1] : 0;
    const size_t mask_nb2 = mask ? mask->nb[2] : 0;
    const size_t mask_nb3 = mask ? mask->nb[3] : 0;

    dim3 grid((N_q + 63) / 64, n_heads, batch);
    dim3 block(32, 8);

    const bool k_f16 = (K->type == GGML_TYPE_F16);
    const bool v_f16 = (V->type == GGML_TYPE_F16);
    const bool has_mask = (mask != nullptr);

#define LAUNCH_FATTN_DP4A(DIM, KF16, VF16, HMASK) \
    kernel_flash_attn_dp4a<DIM, KF16, VF16, HMASK><<<grid, block, 0, stream>>>( \
        (const char*)Q->data, (const char*)K->data, (const char*)V->data, mask_ptr, \
        (float*)dst->data, scale, N_q, N_kv, n_heads, gqa_ratio, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        K->nb[1], K->nb[2], K->nb[3], \
        V->nb[1], V->nb[2], V->nb[3], \
        mask_ne1, mask_ne2, mask_ne3, mask_nb1, mask_nb2, mask_nb3)

#define DISPATCH_FATTN_DIM(DIM) \
    do { \
        if (k_f16 && v_f16) { \
            if (has_mask) LAUNCH_FATTN_DP4A(DIM, true, true, true); \
            else          LAUNCH_FATTN_DP4A(DIM, true, true, false); \
        } else if (!k_f16 && !v_f16) { \
            if (has_mask) LAUNCH_FATTN_DP4A(DIM, false, false, true); \
            else          LAUNCH_FATTN_DP4A(DIM, false, false, false); \
        } else if (k_f16 && !v_f16) { \
            if (has_mask) LAUNCH_FATTN_DP4A(DIM, true, false, true); \
            else          LAUNCH_FATTN_DP4A(DIM, true, false, false); \
        } else { \
            if (has_mask) LAUNCH_FATTN_DP4A(DIM, false, true, true); \
            else          LAUNCH_FATTN_DP4A(DIM, false, true, false); \
        } \
    } while (0)

    if (head_dim == 128) {
        DISPATCH_FATTN_DIM(128);
    } else {
        DISPATCH_FATTN_DIM(64);
    }

#undef DISPATCH_FATTN_DIM
#undef LAUNCH_FATTN_DP4A

    return true;
}
