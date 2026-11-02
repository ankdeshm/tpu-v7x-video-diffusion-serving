# Cloud TPU `v7x` (`Ironwood`, `2x2x4` `v7x-16`) Video + Stereo-Audio Diffusion Serving (`MiniMax-H3`, `LTX-2.5`, `Wan 2.2 A14B`)

Reproducible multi-host serving infrastructure, containers, Kubernetes Indexed Job manifests, and **`109`-point empirical benchmark suite** for **video + stereo-audio diffusion models** on **Cloud TPU `v7x-16` (`Ironwood`, `2x2x4` = 4 hosts × 4 chips/host = `16` physical chips / `32` TensorCores, `3,072 GiB` HBM3e)** using **`sglang-torchtpu`** (`SGLANG_TPU_SPLASH_VARLEN=1` Pallas Mosaic FlashAttention + `Ulysses=8` + All-HBM FP8/BF16) and **`vllm-torchtpu` (`vLLM-Omni`)**.

- **Full Exhaustive 109-Point Benchmark Report:** [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md)

---

## 1. Benchmark Highlights & 30-RPM Fleet Sizing → Exact Script (`#line`), Manifest & Proof

| Model | Precision & Mode | Workload | Single-Host (`4` Chips) | `16`-Chip Slice (`DP=4`) | Slices (`Hosts` / `Chips`) for `30 RPM` | Script (`#line`) & Manifest | Empirical Result Proof |
|---|---|---|---:|---:|---|---|---|
| **`LTX-2.5`** | **FP8 All-HBM (`U=8`)** | `544p` (`960×544`) `5s 121f` (`c=8`) | `2.563 RPM` | **`10.019 RPM`** | `3` slices (`12` hosts / `48` chips) | [`ltx25.py#L244`](./scripts/run_v7x_ltx25_serving.py#L244-L320) · [`#L1067`](./scripts/run_v7x_ltx25_serving.py#L1067-L1200) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) · [`Pod0`](./results/v7x_flex_ltx25_serve_p2/pod_0_results.json) |
| **`LTX-2.5`** | **FP8 All-HBM (`U=8`)** | `720p` (`1280×720`) `5s 121f` (`c=8`) | `2.045 RPM` | **`9.288 RPM`** | `4` slices (`16` hosts / `64` chips) | [`ltx25.py#L1067`](./scripts/run_v7x_ltx25_serving.py#L1067-L1200) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) |
| **`LTX-2.5`** | **FP8 All-HBM (`U=8`)** | **`1080p` Kling-V4 Proxy `20s 481f`** (`c=8`) | `0.371 RPM` | **`1.494 RPM`** *(`0.498x` RTF)* | `21` slices (`84` hosts / `336` chips) | [`ltx25.py#L579`](./scripts/run_v7x_ltx25_serving.py#L579-L690) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) |
| **`Wan 2.2 T2V`** | **FP8 All-HBM (`U=8`)** | `480p` (`832×480`) `5s 81f` (`4`-step fast) | **`3.387 RPM`** | **`13.548 RPM`** | `3` slices (`12` hosts / `48` chips) | [`wan22.py#L249`](./scripts/run_v7x_wan22_serving.py#L249-L425) · [`#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) · [`Pod0`](./results/v7x_flex_wan22_serve_p3/pod_0_results.json) |
| **`Wan 2.2 I2V`** | **FP8 All-HBM (`U=8`)** | `480p` (`832×480`) `5s 81f` (`4`-step, `c=8`) | `2.640 RPM` | **`10.633 RPM`** | `3` slices (`12` hosts / `48` chips) | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`Wan 2.2 I2V`** | **FP8 All-HBM (`U=8`)** | `720p` (`1280×720`) `5s 81f` (`4`-step, `c=8`) | `1.105 RPM` | **`4.431 RPM`** | `7` slices (`28` hosts / `112` chips) | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`Wan 2.2 I2V`** | **FP8 All-HBM (`U=8`)** | `720p` `5s 81f` / `10s 161f` (`20`-step, `c=4`) | `0.311` / `0.118` | **`1.243` / `0.473 RPM`** | `25` / `64` slices | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`MiniMax-H3`** | **FP8 + Pallas FlashAttn** | `768p` (`1344×768`) **`15s` (`4`-step, `175.51s`)** | **`0.342 RPM`** | **`1.368 RPM`** | `22` slices (`88` hosts / `352` chips) | [`minimax.py#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) · [`#L1270`](./scripts/run_v7x_minimax_h3_serving.py#L1270-L1420) · [`yaml`](./k8s/v7x-16-flex-minimax-h3-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`Pod0 JSON`](./results/v7x_flex_minimax_h3_serve_p1/pod_0_results.json) |

---

## 2. How Baseline Serving Was Optimized on TPU `v7x-16`

| Step | What Changed Over Stock `sglang-torchtpu` Baseline | Code Anchor | Measured Impact |
|---|---|---|---|
| **1. All-HBM-Resident FP8 (`--dit-cpu-offload=false`, `--quantization fp8`)** | Eliminated default host-CPU PCIe model offloading by keeping DiT + text/vision encoders + VAEs resident in `768 GiB`/host HBM3e | [`build_tpu_server_env#L789`](./scripts/run_v7x_ltx25_serving.py#L789-L878) · [`wan22.py#L511`](./scripts/run_v7x_wan22_serving.py#L511-L600) | **`+47%` to `+54%` throughput speedup** |
| **2. Hardware Pallas Mosaic FlashAttention (`SGLANG_TPU_SPLASH_VARLEN=1`)** | Routed `TpuSplashAttentionImpl.forward_varlen` through `v7x` `64 MiB` VMEM Pallas `flash_attention` (`256×256` MXU blocks) instead of fallback `SDPA` | [`apply_sglang_tpu_runtime_patches#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) | Eliminated $O(S^2)$ HBM OOM at `MiniMax-H3` `768p 15s` (`S ≈ 95K` tokens -> **`175.51s`**) |
| **3. Single-Host `2x2x1` Metadata Isolation (`DP=4 × Ulysses=8`)** | `LD_PRELOAD=/workspace/metadata_hook.so` + `mock_metadata.py` (`CHIPS_PER_HOST_BOUNDS=2,2,1`) isolates all 4 hosts on `2x2x4` + avoids PJRT `CFG=2` subgroup collective deadlocks | [`setup_single_host_metadata_hook#L694`](./scripts/run_v7x_ltx25_serving.py#L694-L722) | **`4.02×–4.03×` linear scaling** from 1 host (`4` chips) to full `v7x-16` slice (`16` chips) |

---

## 3. Repository Structure & Quickstart

- [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md) — Complete `109`-evaluation report across `MiniMax-H3`, `LTX-2.5`, and `Wan 2.2`
- [`docker/Dockerfile.sglang_tpu`](./docker/Dockerfile.sglang_tpu), [`docker/prep_sglang_tpu.py`](./docker/prep_sglang_tpu.py), [`docker/Dockerfile.vllm_omni`](./docker/Dockerfile.vllm_omni), [`docker/prep_vllm_omni.py`](./docker/prep_vllm_omni.py)
- [`scripts/run_v7x_minimax_h3_serving.py`](./scripts/run_v7x_minimax_h3_serving.py) — Priority #1: `MiniMax-H3` (`FL2VA` `i2v` + `t2va`) & `FastH3` benchmark suite
- [`scripts/run_v7x_ltx25_serving.py`](./scripts/run_v7x_ltx25_serving.py) — Priority #2: `LTX-2.5` (incl. Two-Stage `1080p` `20s` Kling-V4 proxy) suite
- [`scripts/run_v7x_wan22_serving.py`](./scripts/run_v7x_wan22_serving.py) — Priority #3: `Wan 2.2 A14B` (`I2V` & `T2V`) suite
- [`scripts/stage_all_video_models_gcs.py`](./scripts/stage_all_video_models_gcs.py) — Parallel HF Hub -> regional GCS stager
- [`k8s/v7x-16-flex-minimax-h3-serving.yaml`](./k8s/v7x-16-flex-minimax-h3-serving.yaml), [`k8s/v7x-16-flex-ltx25-serving.yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml), [`k8s/v7x-16-flex-wan22-serving.yaml`](./k8s/v7x-16-flex-wan22-serving.yaml)
- [`results/`](./results/) — Per-pod & slice `summary.json` outputs (`v7x_flex_ltx25_serve_p2`, `v7x_flex_wan22_serve_p3`, `v7x_flex_minimax_h3_serve_p1`)

```bash
# 1. Upload serving benchmark harnesses to regional GCS
gcloud storage cp scripts/run_v7x_*.py gs://YOUR_GCS_BUCKET/video-serving-scripts/

# 2. Launch Priority #1 (MiniMax-H3), #2 (LTX-2.5), and #3 (Wan 2.2) on TPU v7x-16 (2x2x4)
kubectl apply -f k8s/v7x-16-flex-minimax-h3-serving.yaml
kubectl apply -f k8s/v7x-16-flex-ltx25-serving.yaml
kubectl apply -f k8s/v7x-16-flex-wan22-serving.yaml
```
