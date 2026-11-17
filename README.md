# `sd-cpp-pascal-dp4a` — Native INT8 `__dp4a` Backend for `stable-diffusion.cpp` on NVIDIA Pascal (SM 6.1)

Custom `INT8 __dp4a` GEMM and FlashAttention backend for [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) (base commit `b9d7176`, `ggml d803c238`) targeting **NVIDIA Pascal (`SM 6.1`)** GPUs — delivering **~1.7x–2.0x faster sampling** without quality loss and with safe VRAM limits.

### Supported GPUs (`SM 6.1`: `GP102` / `GP104` / `GP106` / `GP107`)
- **GeForce GTX 10-series / TITAN:** `GTX 1050` / `1050 Ti`, `GTX 1060`, `GTX 1070` / `1070 Ti`, `GTX 1080` / `1080 Ti`, `TITAN X (Pascal)` / `TITAN Xp`
- **Data Center / Workstation:** `Tesla P40 (24 GB)`, `Tesla P4 (8 GB)`, `Quadro P6000 (24 GB)`, `Quadro P5000 (16 GB)`, `Quadro P4000 (8 GB)`, `Quadro P2000`
- **Mining / Compute:** `P102-100 (10 GB)`, `P104-100 (8 GB)`, `P106-100 (6 GB)`

> **Note:** On 16–24 GB cards (`Tesla P40`, `Quadro P6000`, `Quadro P5000`), even large 12B models fit entirely into the **Level 1 Persistent W8 Cache** without needing the Level 2 Scratchpad fallback.  
> `Tesla P100` (`SM 6.0`, `GP100`) is **not** supported by `__dp4a` — `GP100` lacks `__dp4a` hardware instructions and uses native `2:1 FP16` instead.

---

## Repository Structure

- **`bin/`** — static Windows x64 binaries (`CUDA 12.9`, `SM 6.1`):
  - `sd-cli_DP4a_AVX2.exe` / `sd-server_DP4a_AVX2.exe` — for modern CPUs with `AVX2 + FMA` (Intel Haswell 4th gen+, AMD Ryzen+; recommended)
  - `sd-cli_DP4a_SSE4.exe` / `sd-server_DP4a_SSE4.exe` — for older CPUs without `AVX2` (Intel Core i 1st–3rd gen, Xeon E5 v1/v2, AMD FX)
  - *(CLI binaries use `--dp4a` flag or `SD_CUDA_DP4A=1` env var and run stock by default; server binaries enable `--dp4a` by default.)*
- **`patches/`**:
  - `ggml-dp4a.patch` — `ggml` patches (`dp4a_gemm.*`, `fattn_dp4a.cuh`, dispatch & FP32 cuBLAS override in `ggml-cuda.cu`)
  - `sd-cpp-dp4a.patch` — `stable-diffusion.cpp` patches (`--dp4a` flag, `SD_USE_CUDA` for `sd-cli` / `sd-server`, cache cleanup on `free_sd_ctx`)
- **`src/`** — modified and added source files for quick inspection or direct copy over a `stable-diffusion.cpp` tree.
- **`docs/`** — [ARCHITECTURE_AND_BENCHMARKS.md](docs/ARCHITECTURE_AND_BENCHMARKS.md) (detailed architecture, test protocol, and full benchmark suite).

---

## Key Optimizations

1. **Two-Level W8A8 `128x128` `__dp4a` GEMM (`Q3_K`, `Q4_K`, `Q4_0`, `Q4_1`, `Q5_K`, `Q5_0`, `Q5_1`, `Q6_K`, `Q8_0`)** ([dp4a_gemm.cu](src/ggml-cuda/dp4a_gemm.cu)):
   - **Level 1 (Persistent W8 Cache):** one-time weight repack into `int8` in VRAM (automatically reserves `1536 MiB` of free VRAM for graph activations and VAE).
   - **Level 2 (On-the-fly Layer Scratchpad):** for large models (e.g., `Krea 2 Turbo` ~5.3 GB on an 8 GB GPU) or when `SD_CUDA_DP4A_CACHE_MB=0`, weights are repacked per layer into a single reusable Scratchpad buffer (`~16–96 MB` VRAM) and executed by the fast `128x128` kernel without falling back to stock `mmq`.
   - `32x8` warp repack kernels (`8` warps/block) and vectorized `float4` on-the-fly activation quantization.
2. **Native `INT8 __dp4a` FlashAttention (`head_dim = 64` & `128`)** ([fattn_dp4a.cuh](src/ggml-cuda/fattn_dp4a.cuh)):
   - On-the-fly `INT8` tile quantization for `Q`, `K`, `P`, `V`, `__dp4a` matrix products `Q·Kᵀ` and `P·V`, and pure **FP32** online softmax (avoiding Pascal's `1:64` `half2` penalty), with GQA and mask support.
   - **2.12x** speedup on `FLASH_ATTN_EXT` with minimal VRAM overhead.
3. **Multi-Model Server Safety & `FP16 -> FP32` cuBLAS Guard** ([ggml-cuda.cu](src/ggml-cuda/ggml-cuda.cu)):
   - Per-device state (`g_state[device]`) with explicit cache cleanup hooked to `free_sd_ctx` and `ggml_backend_cuda_free`.
   - Forces `CUDA_R_32F` compute (`prefer_f32_output = true`) for `F16` cuBLAS calls on SM 6.1 when `--dp4a` is active.

---

## Quick Start (Prebuilt `bin/sd-cli_DP4a.exe`)

Add the **`--dp4a`** flag to your standard command:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:PATH"

.\bin\sd-cli_DP4a.exe `
  --diffusion-model <model.gguf> `
  --llm <text_encoder> `
  --vae <vae> `
  --offload-to-cpu --mmap --vae-tiling --vae-tile-size 512x512 --diffusion-fa --backend te=cpu --dp4a `
  -p "your prompt" --steps 20 --cfg-scale 5.0 --seed 42 -H 768 -W 512 -o out.png
```

### Optional Environment Variables
- `SD_CUDA_DP4A=1` — enable DP4A without passing `--dp4a`.
- `SD_CUDA_DP4A_CACHE_MB=0` (or `SD_CUDA_DP4A_SCRATCHPAD=1`) — disable persistent W8 cache and run strictly through the single Scratchpad buffer (minimum VRAM).
- `SD_CUDA_DP4A_CACHE_MB=N` — cap persistent W8 cache at `N` MiB.

---

## Building from Patches

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git
cd stable-diffusion.cpp
git checkout b9d7176
git submodule update --init --recursive

git apply ../patches/sd-cpp-dp4a.patch
git -C ggml apply ../../patches/ggml-dp4a.patch

cmake -B build_unified -G Ninja -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build build_unified --target sd-cli sd-server -j
```

---

## Benchmarks on GTX 1070 (8 GB VRAM, `--vae-tile-size 512x512`, `--backend te=cpu`)

### `512×768`
| Model | Configuration | Step Time | Sampling Time | Sampling Speedup |
|---|---|---:|---:|---:|
| `ANIMA` (`Q4_K`, 20 steps) | Stock (`mmq`) | `2.41s/it` | `49.24s` | — |
| `ANIMA` (`Q4_K`, 20 steps) | `--dp4a` (full cache) | **`1.40s/it`** | **`28.66s`** | **+71.8%** |
| `ANIMA` (`Q4_K`, 20 steps) | `--dp4a` (`CACHE_MB=0`, Scratchpad ~16 MB) | `1.43s/it` | `29.36s` | **+67.7%** |
| `ANIMA` (`Q3_K`, 20 steps) | Stock (`mmq`) | `2.62s/it` | `52.49s` | — |
| `ANIMA` (`Q3_K`, 20 steps) | `--dp4a` (full cache) | **`1.39s/it`** | **`28.46s`** | **+84.4%** |
| `Krea 2 Turbo` (`Q3_K`, 8 steps) | Stock (`mmq`) | `6.03s/it` | `50.17s` | — |
| `Krea 2 Turbo` (`Q3_K`, 8 steps) | `--dp4a` (auto / Scratchpad ~96 MB) | **`3.25–3.32s/it`** | **`28.59–28.67s`** | **+75.0%…+75.5%** |

### `1024×1024`
| Model | Settings | Stock (`mmq`) | `--dp4a` | Sampling Speedup |
|---|---|---:|---:|---:|
| `ANIMA` (`Q4_K`, no Turbo LoRA) | 30 steps, `CFG 5.0` | `8.10s/it` (`244.48s`) | **`4.47s/it`** (**`137.82s`**) | **1.77x** (step **1.81x**) |
| `Krea 2 Turbo` (`Q3_K`) | 10 steps, `CFG 1.0` | `20.54s/it` (`214.09s`) | **`10.59s/it`** (**`114.02s`**) | **1.88x** (step **1.94x**) |
| `Z-Image` (`Q5_K`) | 8 steps, `CFG 1.0` | `9.99s/it` (`84.55s`) | **`5.33s/it`** (**`43.95s`**) | **1.92x** (step **1.87x**) |

See [docs/ARCHITECTURE_AND_BENCHMARKS.md](docs/ARCHITECTURE_AND_BENCHMARKS.md) for `832×1216`, FullHD `1920×1080` VRAM profiling, and practical limitations.

> [!NOTE]
> **Legacy UNet models (SD 1.5 / SDXL):** `--dp4a` targets modern **DiT** architectures (`~1.7x–1.92x` sampling speedup) where compute is dominated by `mul_mat` and attention. On UNet models, most runtime goes into `conv2d` and VAE decoding, so expected sampling gain on SDXL is **~10–11%** (e.g., `90.98s → 81.35s` on `ILL_ramthrusts_Q8_0` at `1216×832`, 20 steps), which is near the practical ceiling without custom INT8 `conv2d` kernels.

---

## Credits

Built by a hardware enthusiast & overclocker in collaboration with **Gemini 4 Hydrogen**.

