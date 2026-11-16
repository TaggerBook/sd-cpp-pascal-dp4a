# DP4A Backend Architecture & Benchmarks (NVIDIA Pascal SM 6.1)

## 1. Why Pascal (SM 6.1) & Why `__dp4a`

NVIDIA Pascal GPUs (**SM 6.1**: `GTX 1060`, `GTX 1070`, `GTX 1080`, `GTX 1080 Ti`, `P102`, `P104`) lack Tensor Cores and suffer a **1:64 FP16 (`half2`) throughput penalty** relative to `FP32`. However, SM 6.1 chips include the native **`__dp4a`** instruction (4× `int8` multiply-accumulate into `int32` in 1 cycle), providing up to **~4× higher peak compute throughput (`TOPS`)** than `FP32` (`TFLOPS`).

Stock `ggml-cuda` already uses `__dp4a` inside its `mmq` kernels (`64×64` tiles), but on a GTX 1070 stock `mmq` reaches only ~`7.8 TOPS` out of `26 TOPS` peak, and `FLASH_ATTN_EXT` remains a major bottleneck.

---

## 2. Optimization Pipeline

1. **W8A8 `128×128` `__dp4a` GEMM (`src/ggml-cuda/dp4a_gemm.cu`):**
   - Supported quantized weight types: `Q3_K`, `Q4_K`, `Q4_0`, `Q4_1`, `Q5_K`, `Q5_0`, `Q5_1`, `Q6_K`, `Q8_0`.
   - `32×8` warp repack kernels convert GGML blocks into flat `int8` rows + per-row `scale_B[N]`.
   - `FP32` activations are quantized on the fly into `int8` + `scale_A[M]` using vectorized `float4` loads.
   - `kernel_w8a8_dp4a_128x128` computes `128×128` output tiles using `LDS.128` (`int4` shared memory loads) and register prefetching.
2. **Native `INT8 __dp4a` FlashAttention (`src/ggml-cuda/fattn_dp4a.cuh`):**
   - Templated `kernel_flash_attn_dp4a<D>` supporting `head_dim = 64` and `head_dim = 128` (with bounds guards for `lane >= 16` when `D = 64`).
   - On-the-fly `INT8` tile quantization for `Q`, `K`, `P`, `V`; `__dp4a` for `Q·Kᵀ` and `P·V`; **pure `FP32` online softmax** (no `half2` 1:64 penalty).
   - **2.12×** speedup on `FLASH_ATTN_EXT`.
3. **Two-Level VRAM Management:**
   - **Level 1 (`Persistent W8 Cache`):** caches repacked `int8` weights in VRAM while preserving **`1536 MiB`** of free headroom (`cudaMemGetInfo < need_bytes + 1536 MiB` disables caching) so compute graphs do not fragment into multi-segment PCIe offloads and VAE decode fits safely.
   - **Level 2 (`On-the-fly Layer Scratchpad`):** when Level 1 cannot fit (or `SD_CUDA_DP4A_CACHE_MB=0`), weights are repacked per layer into a single reusable Scratchpad buffer (`~16–96 MB`) and still executed by the `128×128 __dp4a` kernel without falling back to `mmq`.
4. **Multi-Model Server Safety & `FP16 -> FP32` cuBLAS Guard (`src/ggml-cuda/ggml-cuda.cu`):**
   - Per-device state (`g_state[device]`) with explicit cache cleanup (`ggml_cuda_dp4a_clear_cache()`) hooked to `free_sd_ctx` and `ggml_backend_cuda_free` — safe for model switching in `sd-server` without interfering with `--offload-to-cpu`.
   - Sets `prefer_f32_output = true` (`CUDA_R_32F` compute) for `GGML_TYPE_F16` on `SM 6.1` when `--dp4a` is enabled.

---

## 3. Full Benchmark Suite (GTX 1070 8 GB)

Common flags: `--offload-to-cpu --mmap --vae-tiling --vae-tile-size 512x512 --diffusion-fa --backend te=cpu --seed 42`.

### `512×768`
| Model | Configuration | Step Time | Sampling | Speedup |
|---|---|---:|---:|---:|
| `ANIMA` (`Q4_K`, 20 steps, `CFG 5.0`) | Stock (`mmq`) | `2.41s/it` | `49.24s` | — |
| `ANIMA` (`Q4_K`, 20 steps, `CFG 5.0`) | `--dp4a` (full cache) | **`1.40s/it`** | **`28.66s`** | **+71.8%** |
| `ANIMA` (`Q4_K`, 20 steps, `CFG 5.0`) | `--dp4a` (`CACHE_MB=0`, Scratchpad) | `1.43s/it` | `29.36s` | **+67.7%** |
| `ANIMA` (`Q3_K`, 20 steps, `CFG 5.0`) | Stock (`mmq`) | `2.62s/it` | `52.49s` | — |
| `ANIMA` (`Q3_K`, 20 steps, `CFG 5.0`) | `--dp4a` (full cache) | **`1.39s/it`** | **`28.46s`** | **+84.4%** |
| `Krea 2 Turbo` (`Q3_K`, 8 steps, `CFG 1.0`) | Stock (`mmq`) | `6.03s/it` | `50.17s` | — |
| `Krea 2 Turbo` (`Q3_K`, 8 steps, `CFG 1.0`) | `--dp4a` (Scratchpad) | **`3.25–3.32s/it`** | **`28.59–28.67s`** | **+75.0%…+75.5%** |

### `1024×1024`
| Model | Settings | Stock (`mmq`) | `--dp4a` | Sampling Speedup |
|---|---|---:|---:|---:|
| `ANIMA` (`Q4_K`, no Turbo LoRA) | 30 steps, `CFG 5.0` | `8.10s/it` (`244.48s`) | **`4.47s/it`** (**`137.82s`**) | **1.77x** (step **1.81x**) |
| `Krea 2 Turbo` (`Q3_K`) | 10 steps, `CFG 1.0` | `20.54s/it` (`214.09s`) | **`10.59s/it`** (**`114.02s`**) | **1.88x** (step **1.94x**) |
| `Z-Image` (`Q5_K`) | 8 steps, `CFG 1.0` | `9.99s/it` (`84.55s`) | **`5.33s/it`** (**`43.95s`**) | **1.92x** (step **1.87x**) |

### `832×1216` & FullHD `1920×1080`
- **`Z-Image` (`Q5_K`, `832×1216`, 2 steps):** step `10.24s/it` → **`5.43s/it`**, sampling `22.69s` → **`12.74s`** (**+78.1%**).
- **`Krea 2 Turbo` (`Q3_K`, `1920×1080`, 2 steps, `nvidia-smi` VRAM polling):** step 2 `48.69s/it` → **`24.13s/it`** (**2.02x**), sampling `104.44s` → **`55.51s`**, peak VRAM `6748 MiB` → **`6870 MiB`** (`+122 MiB`).

---

## 4. Practical Limitations

- **Unquantized `.safetensors` (`FP16` / `BF16`):** the `W8A8 __dp4a` GEMM path targets quantized weights (`Q3_K`–`Q8_0`). On raw `FP16`/`BF16` checkpoints, only FlashAttention accelerates; to get the full `__dp4a` GEMM speedup without converting beforehand, pass `--type q8_0` at load time.
- **`--offload-to-cpu` and `ggml_graph_cut` single-segment threshold:** if `weights + compute_buffers` exceeds the single-segment VRAM budget (e.g., ~7 GB `Q4_K` / `Q4_0` 12B models on a 10 GB card with `--offload-to-cpu`), `ggml_graph_cut` splits the graph into multiple segments and streams weights over PCIe every step, which can bottleneck on limited PCIe lanes.
