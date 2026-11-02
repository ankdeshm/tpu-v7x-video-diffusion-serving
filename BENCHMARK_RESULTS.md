# TPU v7x-16 Video + Audio Diffusion Serving Benchmark

- **Evaluation Environment:** `gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v7x-cluster`, Nodepool `tpu-v7x-16-pool`
- **Hardware Topology:** 4 hosts × 4 `v7x` chips = 16 chips / 32 TensorCores (`3,072 GiB` HBM3e aggregate, `96 GiB`/TensorCore)
- **Serving Engine:** `sglang-torchtpu` with Pallas FlashAttention, online FP8 quantization, and Ulysses sequence parallelism (`Ulysses=8`)
- **Evaluated Models:** `MiniMax-H3` (`FL2VA`), `Lightricks/LTX-2.5` (including `1080p` 20s 481-frame workload), and `Wan 2.2 A14B` (`T2V` & `I2V`)

---

## 1. Executive Summary

Each TPU `v7x` chip provides 2 TensorCores and `192 GiB` HBM3e (`96 GiB` per TensorCore). Across a 16-chip `2x2x4` slice (4 hosts, 32 TensorCores), we evaluated serving latency, single-host vs. multi-host throughput, and 30-RPM capacity sizing for three video generation models:

1. **`MiniMax-H3` (`FL2VA`):** 15-second `768p` (`1344×768`) 4-step image-to-video + stereo audio generation comparing BF16 Pallas FlashAttention, online FP8 quantization, and stock SDPA.
2. **`Lightricks/LTX-2.5`:** 544p, 720p, and 1080p video + audio generation, including 20-second (`481`-frame) `1920×1088` synthesis with fast and two-stage tiled VAE decoding.
3. **`Wan 2.2 A14B` (`T2V` & `I2V`):** Text-to-Video and Image-to-Video synthesis across 480p, 720p, 10-second (`161`-frame), and `720×1280` portrait formats.

### Key Findings
- **Linear Multi-Host Data Parallelism (`DP=4`):** Scaling from 1 host (4 chips) to 4 hosts (16 chips) scaled throughput linearly (`4.00×–4.03×`):
  - **`LTX-2.5` `1080p` 20s (`481f`):** Improved from `0.371 RPM` (1 host) to `1.494 RPM` (4 hosts, `4.02×`).
  - **`Wan 2.2 I2V` `480p` 4-step:** Improved from `2.640 RPM` (1 host) to `10.633 RPM` (4 hosts, `4.03×`).
  - **`Wan 2.2 I2V` `720p` 20-step:** Improved from `0.311 RPM` (1 host) to `1.243 RPM` (4 hosts, `4.00×`).
- **All-HBM-Resident FP8 vs. BF16 / CPU Offload:** Keeping DiT weights, text encoders, and VAEs resident in HBM (`--dit-cpu-offload=false`) with online FP8 quantization improved single-host throughput from `1.744 RPM` (BF16) and `1.329 RPM` (FP8 with CPU offload) to `2.563 RPM` (`+47%` over BF16) on `LTX-2.5` `544p`, and from `2.198 RPM` (BF16) to `3.387 RPM` (`+54%`) on `Wan 2.2 T2V` `480p` 4-step.
- **PJRT Collective Group Compatibility:** Collectives over `dist.group.WORLD` (`Ulysses=8` across 8 TensorCores or `Ulysses=4` across 4 TensorCores) run reliably without stalls, whereas partitioned subgroups (`dist.new_group` in `CFG=2 × U=4`) deadlock in PJRT on `Wan 2.2`. Standardizing on `Ulysses=8` avoids subgroup collective deadlocks.

---

## 2. MiniMax-H3 (FL2VA) Benchmark Results

`MiniMax-H3` (`33.14B` omni-modal video + stereo-audio DiT) was evaluated on single 4-chip `v7x` hosts (`Ulysses=8` across 8 TensorCores) for `768p` (`1344×768`, 16:9) 15-second 4-step image-to-video + audio (`fl2va`) generation (`num_inference_steps=5`, `concurrency=1`). Raw per-pod metrics are archived in [`results/v7x_flex_minimax_h3_serve_p1/`](./results/v7x_flex_minimax_h3_serve_p1/).

| Model | Configuration | Engine | Precision | Attention & Offload Mode | Workload | Denoise Steps | Cold Compile | Median E2E | Mean Inference | ms/step | Host RPM | Peak HBM | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `MiniMax-H3 (FL2VA)` | `mh3_fp8_u8_pallas` | `sglang-torchtpu` | FP8 (online) | Pallas FlashAttn (Resident DiT, Enc/VAE Offload) | `768p` (`1344×768`) 15s `i2v` | 4 | 217.94s | **175.51s** | 172.26s | 43,065.2 | **0.342** | 86.82 GiB | OK (2/2) |
| `MiniMax-H3 (FL2VA)` | `mh3_bf16_u8_pallas` | `sglang-torchtpu` | BF16 | Pallas FlashAttn + AdaLN-online | `768p` (`1344×768`) 15s `i2v` | 4 | 528.25s | **175.72s** | 172.76s | 43,190.0 | **0.341** | 94.40 GiB | OK (2/2) |
| `MiniMax-H3 (FL2VA)` | `mh3_bf16_u8_sdpa_adaln` | `sglang-torchtpu` | BF16 | Stock SDPA + AdaLN-online | `768p` (`1344×768`) 15s `i2v` | 4 | 526.47s | **178.00s** | 174.54s | 43,635.8 | **0.337** | 94.31 GiB | OK (2/2) |

### MiniMax-H3 Summary
- **Cold Compilation Time:** Online FP8 quantization reduced cold XLA compilation and first-request latency from `528.25s` (BF16) to `217.94s` (`-58.7%`).
- **Memory Headroom:** FP8 reduced peak HBM per TensorCore from `94.40 GiB` (`96.66 GB`) to `86.82 GiB` (`88.90 GB`), providing `~9.2 GiB` of safety margin under the `96 GiB` TensorCore limit.
- **Steady-State Latency:** Warm end-to-end 15-second `1344×768` video + 32 kHz stereo AAC generation improved from `178.00s` (stock SDPA, `0.337 RPM`) to `175.51s` (FP8 + Pallas FlashAttention, `0.342 RPM` per host, scaling to `1.368 RPM` across a 4-host `v7x-16` slice).

---

## 3. Lightricks LTX-2.5 Benchmark Results

Lightricks `LTX-2.5` (`22B` video + audio DiT) was evaluated across single-host configurations (BF16 vs. FP8 All-HBM, CPU offload, stock SDPA, and 2-chip sub-host `Ulysses=4`) and the full 16-chip `DP=4` slice, including `1080p` (`1920×1088`) 20-second `481`-frame generation (`Kling-V4` proxy) with fast and two-stage tiled VAE decoding.

### 3.1 Single-Host (4-Chip & 2-Chip) Results

| Configuration | Chips | Precision | Workload | Steps | Cold Compile | Median E2E | ms/step | RPM | RTF | Peak HBM |
|---|---|---|---|---|---|---|---|---|---|---|
| `sglang_bf16_u8` | 4 | BF16 | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 56.315s | 34.209s | 4,030.5 | **1.744** | 0.145x | 62.16 GB |
| `sglang_bf16_u8` | 4 | BF16 | 544p (960x544) 5s 121f I2VA (8-step distill) | 8 | 36.276s | 34.366s | 4,042.0 | **1.746** | 0.145x | 62.20 GB |
| `sglang_bf16_u8` | 4 | BF16 | 544p (960x544) 10s 241f T2VA (8-step distill) | 8 | 69.904s | 40.85s | 4,732.8 | **1.469** | 0.245x | 64.89 GB |
| `sglang_bf16_u8` | 4 | BF16 | 544p (960x544) 20s 481f T2VA (8-step distill) | 8 | 98.65s | 53.852s | 6,146.2 | **1.114** | 0.371x | 90.76 GB |
| `sglang_bf16_u8` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 49.446s | 36.501s | 4,160.0 | **1.644** | 0.137x | 68.20 GB |
| `sglang_bf16_u8` | 4 | BF16 | 720p (1280x704) 5s 121f I2VA (8-step distill) | 8 | 38.763s | 38.233s | 4,362.9 | **1.569** | 0.131x | 68.17 GB |
| `sglang_bf16_u8` | 4 | BF16 | 720p (1280x704) 10s 241f T2VA (8-step distill) | 8 | 66.803s | 48.249s | 5,447.2 | **1.244** | 0.207x | 69.37 GB |
| `sglang_bf16_u8` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (20-step dev) | 20 | 53.862s | 59.575s | 2,817.4 | **1.007** | 0.084x | 69.58 GB |
| `sglang_bf16_u8` | 4 | BF16 | 1080p (1920x1088) 5s 121f (VAE fast, 8-step) | 8 | 59.3s | 48.752s | 5,427.1 | **1.231** | 0.103x | 70.16 GB |
| `sglang_bf16_u8` | 4 | BF16 | 1080p (1920x1088) 5s 121f (Stage2 VAE decode, 8-step) | 8 | - | 49.106s | 5,436.6 | **1.222** | 0.102x | 70.16 GB |
| `sglang_bf16_u8` | 4 | BF16 | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | 8 | 243.327s | 174.077s | 19,823.1 | **0.345** | 0.115x | 93.14 GB |
| `sglang_bf16_u8` | 4 | BF16 | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode) | 8 | - | 176.113s | 20,019.9 | **0.341** | 0.114x | 93.88 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 49.194s | 23.58s | 2,676.2 | **2.563** | 0.214x | 56.03 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 544p (960x544) 5s 121f I2VA (8-step distill) | 8 | 25.079s | 20.813s | 2,366.4 | **2.883** | 0.24x | 56.09 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 544p (960x544) 10s 241f T2VA (8-step distill) | 8 | 56.238s | 30.093s | 3,411.2 | **1.994** | 0.332x | 70.85 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 544p (960x544) 20s 481f T2VA (8-step distill) | 8 | 88.844s | 42.124s | 4,667.9 | **1.424** | 0.475x | 94.69 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 38.64s | 29.342s | 3,293.0 | **2.045** | 0.17x | 61.97 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 720p (1280x704) 5s 121f I2VA (8-step distill) | 8 | 31.607s | 29.102s | 3,274.1 | **2.062** | 0.172x | 62.04 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 720p (1280x704) 10s 241f T2VA (8-step distill) | 8 | 57.712s | 36.112s | 3,962.5 | **1.661** | 0.277x | 75.29 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 720p (1280x704) 5s 121f T2VA (20-step dev) | 20 | 44.149s | 57.441s | 2,716.6 | **1.045** | 0.087x | 63.49 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 1080p (1920x1088) 5s 121f (VAE fast, 8-step) | 8 | 49.168s | 37.106s | 3,972.5 | **1.617** | 0.135x | 65.62 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 1080p (1920x1088) 5s 121f (Stage2 VAE decode, 8-step) | 8 | - | 36.369s | 3,884.4 | **1.65** | 0.137x | 65.62 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | 8 | 231.153s | 161.903s | 18,305.8 | **0.371** | 0.124x | 94.70 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode) | 8 | - | 161.402s | 18,279.6 | **0.372** | 0.124x | 94.70 GB |
| `sglang_fp8_u8_all_hbm` | 4 | FP8 (online) | 544p (960x544) 5s 121f T2VA (8-step, c=2) | 8 | - | 41.629s | 2,473.9 | **2.734** | 0.228x | 57.45 GB |
| `sglang_bf16_cfg2_u4` | 4 | BF16 | 544p (960x544) 5s 121f T2VA (20-step guided CFG=3.0) | 20 | 59.826s | 49.291s | 2,361.8 | **1.217** | 0.101x | 62.50 GB |
| `sglang_bf16_cfg2_u4` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (20-step guided CFG=3.0) | 20 | 57.761s | 55.779s | 2,645.2 | **1.076** | 0.09x | 62.90 GB |
| `sglang_bf16_cfg2_u4` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (40-step guided CFG=3.0) | 40 | - | 85.573s | 2,065.6 | **0.701** | 0.058x | 62.90 GB |
| `sglang_bf16_cfg2_u4` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 37.739s | 38.972s | 4,504.9 | **1.54** | 0.128x | 62.86 GB |
| `sglang_fp8_2chip_u4` | 2 | FP8 (online) | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 63.766s | 41.651s | 4,935.0 | **1.452** | 0.121x | 44.84 GB |
| `sglang_fp8_2chip_u4` | 2 | FP8 (online) | 544p (960x544) 10s 241f T2VA (8-step distill) | 8 | 77.815s | 47.419s | 5,549.5 | **1.265** | 0.211x | 47.59 GB |
| `sglang_fp8_2chip_u4` | 2 | FP8 (online) | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 57.729s | 46.023s | 5,360.1 | **1.304** | 0.109x | 47.77 GB |
| `sglang_fp8_2chip_u4` | 2 | FP8 (online) | 1080p (1920x1088) 5s 121f (VAE fast, 8-step) | 8 | 81.797s | 62.704s | 7,154.2 | **0.957** | 0.08x | 48.38 GB |
| `sglang_bf16_2chip_u4` | 2 | BF16 | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 36.66s | 29.658s | 3,490.2 | **2.014** | 0.168x | 62.15 GB |
| `sglang_bf16_2chip_u4` | 2 | BF16 | 544p (960x544) 10s 241f T2VA (8-step distill) | 8 | 47.657s | 38.401s | 4,448.0 | **1.562** | 0.26x | 64.85 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 64.455s | 44.386s | 5,365.6 | **1.329** | 0.111x | 44.83 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 59.805s | 48.495s | 5,701.6 | **1.237** | 0.103x | 45.21 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 1080p (1920x1088) 5s 121f (VAE fast, 8-step) | 8 | 76.523s | 57.825s | 6,560.9 | **1.038** | 0.086x | 45.81 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | 8 | 274.552s | 182.547s | 20,882.9 | **0.329** | 0.11x | 81.03 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode) | 8 | - | 182.257s | 20,853.5 | **0.329** | 0.11x | 81.10 GB |
| `sglang_fp8_u8_cpu_offload` | 4 | FP8 (online) | 544p (960x544) 5s 121f T2VA (8-step, c=2) | 8 | - | 88.618s | 5,321.9 | **1.34** | 0.112x | 51.21 GB |
| `sglang_bf16_u8_sdpa` | 4 | BF16 | 544p (960x544) 5s 121f T2VA (8-step distill) | 8 | 38.262s | 33.555s | 3,961.1 | **1.788** | 0.149x | 62.13 GB |
| `sglang_bf16_u8_sdpa` | 4 | BF16 | 720p (1280x704) 5s 121f T2VA (8-step distill) | 8 | 42.196s | 36.469s | 4,228.1 | **1.645** | 0.137x | 62.48 GB |
| `sglang_bf16_u8_sdpa` | 4 | BF16 | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | 8 | 235.197s | 174.269s | 19,874.5 | **0.344** | 0.115x | 93.52 GB |

### 3.2 Full 16-Chip Slice (`DP=4`) Throughput Sweep (FP8 All-HBM)

| Cluster Configuration | Workload | Concurrency | Wall Time | Median E2E | P90 E2E | Slice RPM | Real-Time Factor (RTF) | Success Rate |
|---|---|---|---|---|---|---|---|---|
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 544p (960x544) 5s 121f (8-step) | $c=4$ | 72.742s | 22.82s | 24.587s | **9.898** | **0.825x** | 12/12 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 544p (960x544) 5s 121f (8-step) | $c=8$ | 95.818s | 38.393s | 53.637s | **10.019** | **0.835x** | 16/16 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x704) 5s 121f (8-step) | $c=4$ | 74.264s | 23.825s | 26.589s | **9.695** | **0.808x** | 12/12 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x704) 5s 121f (8-step) | $c=8$ | 103.359s | 47.409s | 55.193s | **9.288** | **0.774x** | 16/16 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | $c=4$ | 323.408s | 161.173s | 161.965s | **1.484** | **0.495x** | 8/8 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast) | $c=8$ | 482.0s | 319.718s | 321.061s | **1.494** | **0.498x** | 12/12 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode) | $c=4$ | 160.484s | 160.435s | 160.482s | **1.495** | **0.498x** | 4/4 |

### LTX-2.5 Summary
- **1080p 20s (`481f`) Synthesis:** Single-host latency improved from `182.55s` (FP8 with CPU offload) and `174.08s` (BF16) to `161.40s` (`0.372 RPM`) in FP8 All-HBM mode, scaling to `1.494 RPM` (`0.498x` RTF) across the 4-host `v7x-16` slice (`DP=4`).
- **Two-Stage Tiled VAE Decoding:** Two-stage spatial tiled VAE decoding added `<0.5s` wall-clock difference compared to fast VAE decoding (`161.40s` vs `161.90s`).
- **Short-Form 544p 5s (`121f`) Synthesis:** Single-host latency improved from `34.21s` (`1.744 RPM`, BF16) to `23.58s` (`2.563 RPM`, FP8 All-HBM), scaling to `10.019 RPM` at `DP=4` ($c=8$).

---

## 4. Wan 2.2 A14B (T2V & I2V) Benchmark Results

`Wan 2.2 A14B` (`14B` active two-stage video diffusion model) was evaluated for both Text-to-Video (`T2V`) and Image-to-Video (`I2V`) conditioning across `480p`, `720p`, 10-second (`161`-frame) sequences, and `720×1280` portrait orientation.

### 4.1 Single-Host (4-Chip & 2-Chip) Results

| Configuration | Chips | Task | Precision | Workload | Steps | Cold Compile | Median E2E | ms/step | RPM | RTF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 480p (832x480) 5s 81f (4-step fast) | 4 | 40.69s | 37.343s | 4,509.8 | **1.835** | 0.153x | OK (3/3) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 480p (832x480) 5s 81f (20-step) | 20 | 42.204s | 69.673s | 1,463.9 | **0.861** | 0.072x | OK (2/2) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 480p (832x480) 5s 81f (40-step std) | 40 | - | 128.591s | 1,437.0 | **0.467** | 0.039x | OK (2/2) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 720p (1280x720) 5s 81f (4-step fast) | 4 | 135.401s | 54.637s | 5,099.2 | **1.095** | 0.091x | OK (3/3) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 720p (1280x720) 5s 81f (20-step) | 20 | 70.165s | 189.917s | 3,767.0 | **0.316** | 0.026x | OK (2/2) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 720p (1280x720) 5s 81f (40-step std) | 40 | - | 340.276s | 3,281.2 | **0.176** | 0.015x | OK (2/2) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 720p (1280x720) 10s 161f (20-step) | 20 | 278.782s | 499.568s | 9,810.0 | **0.12** | 0.02x | OK (2/2) |
| `wan22_i2v_bf16_u8` | 4 | I2V | BF16 | 720p Portrait (720x1280) 5s 81f (20-step) | 20 | 101.291s | 192.935s | 3,846.7 | **0.311** | 0.026x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (4-step fast) | 4 | 90.309s | 22.558s | 2,202.2 | **2.64** | 0.22x | OK (3/3) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (20-step) | 20 | 28.571s | 71.916s | 1,658.0 | **0.834** | 0.07x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (40-step std) | 40 | - | 126.568s | 1,473.5 | **0.474** | 0.04x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (4-step fast) | 4 | 141.436s | 54.136s | 5,073.0 | **1.105** | 0.092x | OK (3/3) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step) | 20 | 70.162s | 193.197s | 3,967.6 | **0.311** | 0.026x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (40-step std) | 40 | - | 347.019s | 3,479.9 | **0.173** | 0.014x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 10s 161f (20-step) | 20 | 273.256s | 506.586s | 10,091.8 | **0.118** | 0.02x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 720p Portrait (720x1280) 5s 81f (20-step) | 20 | 102.806s | 196.214s | 4,038.8 | **0.306** | 0.025x | OK (2/2) |
| `wan22_i2v_fp8_u8_all_hbm` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (4-step, c=2) | 4 | - | 45.103s | 2,167.2 | **2.646** | 0.22x | OK (4/4) |
| `wan22_i2v_fp8_cfg2_u4_all_hbm` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (20-step CFG=2xU4) | 20 | 1225.935s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_i2v_fp8_cfg2_u4_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step CFG=2xU4) | 20 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_i2v_fp8_cfg2_u4_all_hbm` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (40-step CFG=2xU4) | 40 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_i2v_fp8_u8_cpu_offload` | 4 | I2V | FP8 (online) | 480p (832x480) 5s 81f (4-step fast) | 4 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_i2v_fp8_u8_cpu_offload` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (4-step fast) | 4 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_i2v_fp8_u8_cpu_offload` | 4 | I2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step) | 20 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 480p (832x480) 5s 81f (4-step fast) | 4 | 90.365s | 31.679s | 3,383.8 | **2.198** | 0.183x | OK (3/3) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 480p (832x480) 5s 81f (20-step) | 20 | 37.165s | 64.66s | 1,771.2 | **0.928** | 0.077x | OK (2/2) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 480p (832x480) 5s 81f (40-step std) | 40 | - | 123.07s | 1,930.6 | **0.488** | 0.041x | OK (2/2) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 720p (1280x720) 5s 81f (4-step fast) | 4 | 128.409s | 43.105s | 2,927.8 | **1.397** | 0.116x | OK (3/3) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 720p (1280x720) 5s 81f (20-step) | 20 | 57.637s | 177.915s | 4,966.5 | **0.337** | 0.028x | OK (2/2) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 720p (1280x720) 5s 81f (40-step std) | 40 | - | 327.777s | 4,957.3 | **0.183** | 0.015x | OK (2/2) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 720p (1280x720) 10s 161f (20-step) | 20 | 250.158s | 474.538s | 13,781.2 | **0.126** | 0.021x | OK (2/2) |
| `wan22_t2v_bf16_u8` | 4 | T2V | BF16 | 720p Portrait (720x1280) 5s 81f (20-step) | 20 | 86.231s | 180.906s | 5,025.6 | **0.332** | 0.028x | OK (2/2) |
| `wan22_t2v_fp8_2chip_u4_all_hbm` | 2 | T2V | FP8 (online) | 480p (832x480) 5s 81f (4-step fast) | 4 | 80.748s | 22.054s | 2,020.5 | **2.721** | 0.227x | OK (3/3) |
| `wan22_t2v_fp8_2chip_u4_all_hbm` | 2 | T2V | FP8 (online) | 480p (832x480) 5s 81f (20-step) | 20 | 32.08s | 111.494s | 3,392.8 | **0.538** | 0.045x | OK (2/2) |
| `wan22_t2v_fp8_2chip_u4_all_hbm` | 2 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (4-step fast) | 4 | 145.937s | 57.382s | 5,798.5 | **1.046** | 0.087x | OK (2/2) |
| `wan22_t2v_fp8_2chip_u4_all_hbm` | 2 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step) | 20 | 87.183s | 327.427s | 9,916.4 | **0.183** | 0.015x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 480p (832x480) 5s 81f (4-step fast) | 4 | 19.059s | 17.546s | 1,175.0 | **3.387** | 0.282x | OK (3/3) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 480p (832x480) 5s 81f (20-step) | 20 | 23.06s | 66.669s | 1,949.6 | **0.9** | 0.075x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 480p (832x480) 5s 81f (40-step std) | 40 | - | 121.293s | 1,936.7 | **0.495** | 0.041x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (4-step fast) | 4 | 68.666s | 42.601s | 3,016.2 | **1.408** | 0.117x | OK (3/3) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step) | 20 | 58.136s | 181.405s | 5,205.3 | **0.331** | 0.028x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (40-step std) | 40 | - | 335.494s | 5,198.1 | **0.179** | 0.015x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 10s 161f (20-step) | 20 | 249.688s | 481.544s | 14,100.1 | **0.125** | 0.021x | OK (2/2) |
| `wan22_t2v_fp8_u8_all_hbm` | 4 | T2V | FP8 (online) | 720p Portrait (720x1280) 5s 81f (20-step) | 20 | 85.739s | 184.671s | 5,239.8 | **0.325** | 0.027x | OK (2/2) |
| `wan22_t2v_fp8_cfg2_u4_all_hbm` | 4 | T2V | FP8 (online) | 480p (832x480) 5s 81f (20-step CFG=2xU4) | 20 | 1651.959s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_t2v_fp8_cfg2_u4_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (20-step CFG=2xU4) | 20 | 0.502s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_t2v_fp8_cfg2_u4_all_hbm` | 4 | T2V | FP8 (online) | 720p (1280x720) 5s 81f (40-step CFG=2xU4) | 40 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_t2v_bf16_u8_sdpa` | 4 | T2V | BF16 | 480p (832x480) 5s 81f (4-step fast) | 4 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |
| `wan22_t2v_bf16_u8_sdpa` | 4 | T2V | BF16 | 720p (1280x720) 5s 81f (20-step) | 20 | 0.501s | - | - | **0.0** | 0.0x | Failed (Subgroup deadlock) |

### 4.2 Full 16-Chip Slice (`DP=4`) Throughput Sweep (`Wan 2.2 I2V` FP8 All-HBM)

| Cluster Configuration | Workload | Concurrency | Wall Time | Median E2E | P90 E2E | Slice RPM | Real-Time Factor (RTF) | Success Rate |
|---|---|---|---|---|---|---|---|---|
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 480p (832x480) 5s 81f I2V (4-step fast) | $c=4$ | 68.202s | 22.564s | 22.572s | **10.557** | **0.88x** | 12/12 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 480p (832x480) 5s 81f I2V (4-step fast) | $c=8$ | 90.281s | 45.12s | 45.15s | **10.633** | **0.886x** | 16/16 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x720) 5s 81f I2V (4-step fast) | $c=4$ | 162.982s | 54.154s | 54.658s | **4.418** | **0.368x** | 12/12 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x720) 5s 81f I2V (4-step fast) | $c=8$ | 216.654s | 108.257s | 108.33s | **4.431** | **0.369x** | 16/16 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x720) 5s 81f I2V (20-step) | $c=4$ | 386.088s | 193.041s | 193.057s | **1.243** | **0.104x** | 8/8 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x720) 5s 81f I2V (20-step) | $c=8$ | 386.383s | 289.541s | 386.38s | **1.242** | **0.104x** | 8/8 |
| DP=4 × Ulysses=8 (16 v7x chips, FP8, All-HBM-Resident) | 720p (1280x720) 10s 161f I2V (20-step) | $c=4$ | 507.112s | 506.647s | 507.111s | **0.473** | **0.079x** | 4/4 |

### Wan 2.2 Summary
- **4-Step Fast Mode:** Single-host latency improved from `31.68s` (`2.198 RPM`, BF16) to `17.55s` (`3.387 RPM`, FP8 All-HBM) on `480p` T2V, and from `37.34s` (`1.835 RPM`, BF16) to `22.56s` (`2.640 RPM`, FP8 All-HBM) on `480p` I2V.
- **Multi-Host Scaling (`DP=4`):** Across the 16-chip slice, `480p` 4-step I2V reached `10.633 RPM`, `720p` 4-step reached `4.431 RPM`, and `720p` 20-step reached `1.243 RPM` (`4.0×` single-host throughput).
- **10-Second (`161f`) Synthesis:** `720p` 10-second I2V completed in `506.59s` median on 1 host and `506.65s` at slice concurrency $c=4$ (`0.473 RPM`).
- **2-Chip Sub-Host Deployment:** Running `Wan 2.2 T2V` across 2 chips (4 TensorCores, `Ulysses=4`) completed `480p` 4-step in `22.05s` (`2.721 RPM`) and `720p` 4-step in `57.38s` (`1.046 RPM`).

---

## 5. Runtime Fixes & Architecture Notes

### 5.1 Wan 2.2 VAE Spatial Parallel Conv Halo-Exchange Fix
- **Issue:** During initial I2V warmup, the server failed with `RuntimeError: PjRtBuffer::CopyToRemoteDevice: INVALID_ARGUMENT: Remote size 32256 is not identical to local size 1032192` in `parallel_conv.py:344`.
- **Cause:** `WanVAEConfig` defaults to `use_parallel_encode = True`, causing `WanEncoder3d` to run P2P halo exchange via `dist.batch_isend_irecv` across TPU devices with mismatched causal temporal padding sizes.
- **Fix:** Bypassed spatial halo exchange on TPU in `parallel_conv.py` and set `use_parallel_encode = False` in `wanvae.py` so image conditioning runs locally per TensorCore.

### 5.2 Global World Collectives vs. Subgroup Collectives on PJRT
- **Issue:** `wan22_t2v_fp8_cfg2_u4_all_hbm` and `wan22_i2v_fp8_cfg2_u4_all_hbm` stalled during warmup denoising, whereas `Ulysses=8` (8 cores) and 2-chip `Ulysses=4` (4 cores) completed without stalls.
- **Cause:** Stack inspection showed workers blocked at `get_cfg_group().broadcast(p, src=0)`. PJRT supports collectives over `dist.group.WORLD` (`all_to_all`, `all_reduce`, `broadcast`), whereas partitioned subgroups (`dist.new_group` for 2-branch CFG) deadlock.
- **Fix:** Use `dist.group.WORLD` sequence parallelism (`Ulysses=8` on 4 chips or `Ulysses=4` on 2 isolated chips) rather than `CFG=2` subgroup parallelism.

### 5.3 LTX-2.5 Dynamic Boolean Mask Handling
- **Cause:** XLA lowering on TPU encountered dynamic boolean mask issues in `text_connector.py`, producing NaNs during prompt projection.
- **Fix:** Executed connector projection on CPU and transferred conditioning tensors to TPU, preserving numerical accuracy without XLA graph invalidation.

---

## 6. Capacity Sizing for 30 RPM

> [!NOTE]
> Sizing is computed from measured slice and single-host throughput. Each `v7x` host contains 4 chips (8 TensorCores), and one `v7x-16` slice contains 4 hosts (16 chips / 32 TensorCores).

| Model | Precision | Workload Resolution & Length | Target RPM | Measured RPM / Slice (16 chips) | Slices Required | Hosts Required | `v7x` Chips Required | TensorCores Required |
|---|---|---|---|---|---|---|---|---|
| **LTX-2.5** | FP8 All-HBM | 544p (960x544) 5s 121f (c=8) | 30 RPM | 10.019 RPM | **3** | **12** | **48** | **96** |
| **LTX-2.5** | FP8 All-HBM | 720p (1280x720) 5s 121f (c=8) | 30 RPM | 9.288 RPM | **4** | **16** | **64** | **128** |
| **LTX-2.5** | FP8 All-HBM | 1080p (1920x1088) 5s 121f | 30 RPM | 6.468 RPM | **5** | **20** | **80** | **160** |
| **LTX-2.5** | FP8 All-HBM | 1080p Kling-V4 Proxy 20s 481f (c=8) | 30 RPM | 1.494 RPM | **21** | **84** | **336** | **672** |
| **LTX-2.5** | BF16 | 1080p Kling-V4 Proxy 20s 481f | 30 RPM | 1.380 RPM | **22** | **88** | **352** | **704** |
| **Wan 2.2 T2V** | FP8 All-HBM | 480p (832x480) 5s 81f (4-step fast) | 30 RPM | 13.548 RPM | **3** | **12** | **48** | **96** |
| **Wan 2.2 I2V** | FP8 All-HBM | 480p (832x480) 5s 81f (4-step fast, c=8) | 30 RPM | 10.633 RPM | **3** | **12** | **48** | **96** |
| **Wan 2.2 I2V** | FP8 All-HBM | 720p (1280x720) 5s 81f (4-step fast, c=8) | 30 RPM | 4.431 RPM | **7** | **28** | **112** | **224** |
| **Wan 2.2 I2V** | FP8 All-HBM | 720p (1280x720) 5s 81f (20-step, c=4) | 30 RPM | 1.243 RPM | **25** | **100** | **400** | **800** |
| **Wan 2.2 I2V** | FP8 All-HBM | 720p (1280x720) 10s 161f (20-step, c=4) | 30 RPM | 0.473 RPM | **64** | **256** | **1024** | **2048** |
| **MiniMax-H3** | FP8 | 768p (1344x768) 15s (4-step) | 30 RPM | 1.368 RPM | **22** | **88** | **352** | **704** |
| **MiniMax-H3** | BF16 | 768p (1344x768) 15s (4-step) | 30 RPM | 1.364 RPM | **22** | **88** | **352** | **704** |

---

## 7. Deployment Recommendations

1. **Sequence Parallelism (`Ulysses=8`):** Use `dist.group.WORLD` Ulysses sequence parallelism across all 8 TensorCores on a 4-chip host, scaling horizontally with data-parallel host replicas (`DP=4` per 16-chip slice).
2. **All-HBM FP8 Residency:** With `96 GiB` HBM3e per TensorCore, keep DiT weights, text encoders, and VAEs resident in HBM (`--dit-cpu-offload=false`, `--quantization fp8`) to avoid host PCIe transfer overhead (`+47%` to `+54%` throughput improvement on 4/8-step workloads).
3. **Capacity Summary for 30 RPM:**
   - Short-form fast generation (`Wan 2.2` 4-step `480p` or `LTX-2.5` `544p` 5s): **3 slices (12 hosts / 48 chips)**.
   - High-resolution 20-step serving (`Wan 2.2` `720p` 5s): **25 slices (100 hosts / 400 chips)**.
   - Long-duration `1080p` 20s `481`-frame serving (`LTX-2.5`): **21 slices (84 hosts / 336 chips)**, or **22 slices (88 hosts / 352 chips)** for `MiniMax-H3` `768p` 15s.

