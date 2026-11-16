#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_dp4a_set_enabled(bool enabled);
bool ggml_cuda_dp4a_is_enabled();
void ggml_cuda_dp4a_clear_device_cache(int device);
void ggml_cuda_dp4a_clear_cache(void);

bool ggml_cuda_dp4a_can_handle(int cc, const struct ggml_tensor* src0, const struct ggml_tensor* src1, const struct ggml_tensor* dst);
bool ggml_cuda_dp4a_mul_mat(const struct ggml_tensor* src0,
                            const struct ggml_tensor* src1,
                            struct ggml_tensor* dst,
                            cudaStream_t stream);

bool ggml_cuda_dp4a_flash_attn_ext(const struct ggml_tensor* dst, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
