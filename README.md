# TPU `v7x-16` Video + Audio Diffusion Serving & Disaggregated CPU Offload Optimization

Multi-host serving configurations, Dockerfiles, Kubernetes Indexed Job manifests, and benchmark results (`125+` evaluated points) for video + stereo-audio diffusion models (`MiniMax-H3`, `Lightricks/LTX-2.5`, and `Wan 2.2 A14B`) on **Cloud TPU `v7x`** (`v7x-8` `2x2x2` & `v7x-16` `2x2x4` = `16` chips / `32` TensorCores, `3,072 GiB` HBM3e) using **`vllm-omni` (`vllm-torchtpu-omni`)**, **`sglang-torchtpu`**, and **Disaggregated AVX-512 CPU Video Encoding (`video-encode-svc`)**.

- **Detailed Baseline Benchmark Report (`109` points):** [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md)
- **Disaggregated CPU Offload & `vllm-omni` / `torch.compile` Suite:** [`scripts/v7x_opt/`](./scripts/v7x_opt/) · [`k8s/v7x_opt/`](./k8s/v7x_opt/) · [`results/v7x_opt/`](./results/v7x_opt/)

---

## 1. Before vs. After Disaggregated CPU Offloading (`video-encode-svc`) & `vllm-omni` / `torch.compile` Optimization

Direct A/B comparison on Cloud TPU `v7x` comparing our **Pre-Offload Baseline** (`sglang-torchtpu` All-HBM uncompiled, on-host MP4 encoding) against **(A) `sglang-torchtpu` + `torch.compile` + Disaggregated AVX-512 `video-encode-svc`** and **(B) `vllm-omni` (`U=8`, `TPU_FLASH_ATTN`) + Disaggregated AVX-512 `video-encode-svc` (`c3-standard-44`)**:

| Model & Workload | Pre-Offload Baseline (`sglang` On-Host) | Optimized (`vllm-omni` / `sglang+compile` + Disagg CPU) | `v7x-16` Slice RPM (`Before -> After`) | `30-RPM` Fleet Chips (`Before -> After`) | Speedup | References (Before vs. After) |
|---|---:|---:|---:|---:|---:|---|
| **`Wan 2.2 T2V`** `480p` (`832×480`) `5s 81f` (`4`-step) *(Stage A: `sglang + compile + Disagg CPU`)* | `17.55 s` E2E | **`14.241 s`** E2E (`0.389 s` DiT + disagg H.264) | `13.55 -> 16.85 RPM` | `48 -> 32` chips | **`1.23x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite2_pod1_summary.json) · [`suite2.py`](./scripts/v7x_opt/run_suite2_sglang_opt.py) |
| **`Wan 2.2 T2V`** `480p` (`832×480`) `5s 81f` (`4`-step) *(Stage B: `vllm-omni U=8`)* | `17.55 s` E2E | **`5.548 s`** E2E (`5.732 s` w/ disagg H.264) | `13.55 ->` **`43.14 RPM`** | `48 ->` **`16` chips** | **`3.16x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) · [`suite1.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) |
| **`Wan 2.2 I2V`** `480p` (`832×480`) `5s 81f` (`4`-step) *(`vllm-omni U=8 + Disagg CPU`)* | `22.73 s` E2E | **`7.392 s`** E2E (`disagg_cpu_svc`) | `10.63 ->` **`32.47 RPM`** | `48 ->` **`16` chips** | **`3.07x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) · [`video_encode_server.py`](./scripts/v7x_opt/video_encode_server.py) |
| **`Wan 2.2 T2V`** `720p` (`1280×720`) `5s 81f` (`4`-step) *(`vllm-omni U=8 + Disagg CPU`)* | `41.20 s` E2E | **`14.562 s`** (`15.487 s` w/ disagg H.264) | `5.83 ->` **`16.60 RPM`** | `96 ->` **`32` chips** | **`2.83x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** `720p` (`1280×720`) `5s 81f` (`4`-step) *(`vllm-omni U=8 + Disagg CPU`)* | `54.30 s` E2E | **`19.143 s`** E2E (`disagg_cpu_svc`) | `4.43 ->` **`12.54 RPM`** | `112 ->` **`48` chips** | **`2.84x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** `720p` (`1280×720`) `5s 81f` (`20`-step) *(`vllm-omni U=8 + Disagg CPU`)* | `193.08 s` E2E | **`77.699 s`** E2E (`disagg_cpu_svc`) | `1.24 ->` **`3.09 RPM`** | `400 ->` **`160` chips** | **`2.48x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** `720p` (`1280×720`) `10s 161f` (`20`-step) *(`vllm-omni U=8 + Disagg CPU`)* | `507.40 s` E2E | **`230.536 s`** E2E (`disagg_cpu_svc`) | `0.47 ->` **`1.04 RPM`** | `1,024 ->` **`464` chips** | **`2.20x`** | [Before JSON](./results/v7x_flex_wan22_serve_p3/summary.json) · [After JSON](./results/v7x_opt/suite1_pod1_summary.json) |

---

## 2. Full Model Matrix & `30-RPM` Capacity Sizing (`LTX-2.5`, `Wan 2.2`, `MiniMax-H3`)

| Model | Best Engine & Mode | Workload | 1-Host (`4` Chips) | 4-Host Slice (`DP=4`) | Slices (`Chips`) for `30 RPM` | Script & Manifest | Result Files |
|---|---|---|---:|---:|---|---|---|
| **`Wan 2.2 T2V`** | `vllm-omni` (`U=8`, `TPU_FLASH_ATTN`) | `480p` (`832×480`) `5s 81f` (`4`-step) | **`10.786 RPM`** | **`43.144 RPM`** | **`1` slice (`16` chips)** | [`run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) · [`yaml`](./k8s/v7x_opt/suite1-v7x8-pool-a.yaml) | [`suite1_pod1_summary.json`](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** | `vllm-omni` (`U=8` + Disagg CPU) | `480p` (`832×480`) `5s 81f` (`4`-step) | **`8.117 RPM`** | **`32.468 RPM`** | **`1` slice (`16` chips)** | [`run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) · [`yaml`](./k8s/v7x_opt/suite1-v7x8-pool-a.yaml) | [`suite1_pod1_summary.json`](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 T2V`** | `vllm-omni` (`U=8`, `TPU_FLASH_ATTN`) | `720p` (`1280×720`) `5s 81f` (`4`-step) | **`4.150 RPM`** | **`16.600 RPM`** | **`2` slices (`32` chips)** | [`run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) · [`yaml`](./k8s/v7x_opt/suite1-v7x8-pool-a.yaml) | [`suite1_pod1_summary.json`](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** | `vllm-omni` (`U=8` + Disagg CPU) | `720p` (`1280×720`) `5s 81f` (`4`-step) | **`3.134 RPM`** | **`12.537 RPM`** | **`3` slices (`48` chips)** | [`run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) · [`yaml`](./k8s/v7x_opt/suite1-v7x8-pool-a.yaml) | [`suite1_pod1_summary.json`](./results/v7x_opt/suite1_pod1_summary.json) |
| **`Wan 2.2 I2V`** | `vllm-omni` (`U=8` + Disagg CPU) | `720p` `5s 81f` / `10s 161f` (`20`-step) | `0.772` / `0.260 RPM` | **`3.089` / `1.041 RPM`** | `10` / `29` slices | [`run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) · [`yaml`](./k8s/v7x_opt/suite1-v7x8-pool-a.yaml) | [`suite1_pod1_summary.json`](./results/v7x_opt/suite1_pod1_summary.json) |
| **`LTX-2.5`** | `sglang` FP8 All-HBM (`U=8`) | `544p` (`960×544`) `5s 121f` (`c=8`) | `2.563 RPM` | **`10.019 RPM`** | `3` slices (`48` chips) | [`ltx25.py#L244`](./scripts/run_v7x_ltx25_serving.py#L244-L320) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) |
| **`LTX-2.5`** | `sglang` FP8 All-HBM (`U=8`) | `720p` (`1280×720`) `5s 121f` (`c=8`) | `2.317 RPM` | **`9.288 RPM`** | `4` slices (`52–64` chips) | [`ltx25.py#L1067`](./scripts/run_v7x_ltx25_serving.py#L1067-L1200) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) · [`suite2`](./results/v7x_opt/suite2_pod0.jsonl) |
| **`LTX-2.5`** | `sglang` FP8 All-HBM (`U=8`) | `1080p` (`1920×1088`) `20s 481f` (`c=8`) | `0.373 RPM` | **`1.494 RPM`** *(`0.498x` RTF)* | `21` slices (`324–336` chips) | [`ltx25.py#L579`](./scripts/run_v7x_ltx25_serving.py#L579-L690) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) · [`suite2`](./results/v7x_opt/suite2_pod0.jsonl) |
| **`MiniMax-H3`** | `sglang` FP8 + Pallas FlashAttn | `768p` (`1344×768`) `15s` (`4`-step, `175.51s`) | **`0.342 RPM`** | **`1.368 RPM`** | `22` slices (`352` chips) | [`minimax.py#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) · [`yaml`](./k8s/v7x-16-flex-minimax-h3-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`Pod1 JSON`](./results/v7x_flex_minimax_h3_serve_p1/pod_1_results.json) |

---

## 3. Engineering Optimizations Unlocked

| Optimization | Change Over Baseline | Code Reference | Measured Effect |
|---|---|---|---|
| **`vllm-omni` `TPU_FLASH_ATTN` + `Ulysses=8` Execution** | Native XLA/Pallas diffusion execution without Python IPC serialization bottleneck | [`scripts/v7x_opt/run_suite1_omni_ltx_wan.py`](./scripts/v7x_opt/run_suite1_omni_ltx_wan.py) | **`2.20×` to `3.16×` E2E speedup** across all `Wan 2.2` `T2V` and `I2V` resolutions (`480p` & `720p`, `4`-step & `20`-step) |
| **Disaggregated AVX-512 CPU Encoding (`video-encode-svc`)** | Offloads multi-threaded `libx264` + `48 kHz` AAC muxing to dedicated `c3-standard-44` (`44` vCPU) CPU pods (`101–111 fps` sustained) | [`scripts/v7x_opt/video_encode_server.py`](./scripts/v7x_opt/video_encode_server.py) · [`k8s/v7x_opt/video-encode-cpu-service.yaml`](./k8s/v7x_opt/video-encode-cpu-service.yaml) | Eliminates host `ffmpeg`/`PyAV` contention with TPU PJRT dispatch; combined with `sglang + torch.compile`, speeds up `Wan 2.2` `480p` `17.55s -> 14.24s` (`32` vs `48` chips) |
| **All-HBM FP8 (`--dit-cpu-offload=false`, `--quantization fp8`)** | Keeps DiT, text encoders, and VAEs resident in `768 GiB`/host HBM3e instead of PCIe offloading | [`run_v7x_ltx25_serving.py#L789`](./scripts/run_v7x_ltx25_serving.py#L789-L878) · [`run_v7x_wan22_serving.py#L511`](./scripts/run_v7x_wan22_serving.py#L511-L600) | `+47%` to `+54%` throughput over BF16 (`23.58s` vs `34.21s` on `LTX-2.5` `544p`) |
| **Pallas FlashAttention (`SGLANG_TPU_SPLASH_VARLEN=1`)** | Routes `TpuSplashAttentionImpl.forward_varlen` through `v7x` `64 MiB` VMEM Pallas `flash_attention` (`256×256` tiles) | [`run_v7x_minimax_h3_serving.py#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) | Avoids $O(S^2)$ attention score materialization on `MiniMax-H3` `768p 15s` (`175.51s` median E2E, `86.82 GiB` peak HBM) |
| **Single-Host Metadata Isolation (`DP=4 × Ulysses=8`)** | `LD_PRELOAD=/workspace/metadata_hook.so` + `mock_metadata.py` (`CHIPS_PER_HOST_BOUNDS=2,2,1`) isolates each 4-chip host on `2x2x4` | [`run_v7x_ltx25_serving.py#L694`](./scripts/run_v7x_ltx25_serving.py#L694-L722) | `4.00×–4.03×` linear scaling from 1 host (`4` chips) to 4 hosts (`16` chips) |

---

## 4. Repository Structure & Reproduction

- [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md) — Complete `109`-point baseline report across `MiniMax-H3`, `LTX-2.5`, and `Wan 2.2`
- [`scripts/v7x_opt/`](./scripts/v7x_opt/) — Disaggregated CPU `video_encode_server.py` + `run_suite1_omni_ltx_wan.py` (`vllm-omni`) + `run_suite2_sglang_opt.py` (`sglang + torch.compile`) + `run_suite3a_minimax_opt.py`
- [`k8s/v7x_opt/`](./k8s/v7x_opt/) — Kubernetes manifests for `video-encode-svc` (`c3-standard-44`) and `v7x-8` / `v7x-16` optimization jobs
- [`results/v7x_opt/`](./results/v7x_opt/) — Raw JSON & JSONL outputs (`suite1_pod1_summary.json`, `suite2_pod1_summary.json`, etc.)
- [`docker/`](./docker/), [`scripts/run_v7x_*.py`](./scripts/), [`k8s/v7x-16-flex-*.yaml`](./k8s/), [`results/v7x_flex_*`](./results/) — Baseline containers, harnesses, manifests, and raw slice JSONs

```bash
# 1. Deploy disaggregated AVX-512 CPU video-encode-svc
kubectl apply -f k8s/v7x_opt/video-encode-cpu-service.yaml

# 2. Launch optimized vllm-omni + disaggregated CPU serving benchmark on TPU v7x
kubectl apply -f k8s/v7x_opt/suite1-v7x8-pool-a.yaml
kubectl apply -f k8s/v7x_opt/suite2-v7x8-pool-b.yaml
```
