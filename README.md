# TPU v7x-16 Video + Audio Diffusion Serving

Multi-host serving configurations, Dockerfiles, Kubernetes Indexed Job manifests, and benchmark results (`109` evaluated points) for video + stereo-audio diffusion models (`MiniMax-H3`, `Lightricks/LTX-2.5`, and `Wan 2.2 A14B`) on **Cloud TPU `v7x-16`** (`2x2x4` = 4 hosts × 4 chips/host = 16 chips / 32 TensorCores, `3,072 GiB` HBM3e) using `sglang-torchtpu` and `vllm-torchtpu`.

- **Detailed Benchmark Report:** [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md)

---

## 1. Summary Results & 30-RPM Capacity Sizing

| Model | Precision & Mode | Workload | 1-Host (`4` Chips) | 4-Host Slice (`DP=4`) | Slices (`Hosts` / `Chips`) for `30 RPM` | Script & Manifest | Result Files |
|---|---|---|---:|---:|---|---|---|
| **`LTX-2.5`** | FP8 All-HBM (`U=8`) | `544p` (`960×544`) `5s 121f` (`c=8`) | `2.563 RPM` | **`10.019 RPM`** | `3` slices (`12` hosts / `48` chips) | [`ltx25.py#L244`](./scripts/run_v7x_ltx25_serving.py#L244-L320) · [`#L1067`](./scripts/run_v7x_ltx25_serving.py#L1067-L1200) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) · [`Pod0`](./results/v7x_flex_ltx25_serve_p2/pod_0_results.json) |
| **`LTX-2.5`** | FP8 All-HBM (`U=8`) | `720p` (`1280×720`) `5s 121f` (`c=8`) | `2.045 RPM` | **`9.288 RPM`** | `4` slices (`16` hosts / `64` chips) | [`ltx25.py#L1067`](./scripts/run_v7x_ltx25_serving.py#L1067-L1200) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) |
| **`LTX-2.5`** | FP8 All-HBM (`U=8`) | `1080p` (`1920×1088`) `20s 481f` (`c=8`) | `0.371 RPM` | **`1.494 RPM`** *(`0.498x` RTF)* | `21` slices (`84` hosts / `336` chips) | [`ltx25.py#L579`](./scripts/run_v7x_ltx25_serving.py#L579-L690) · [`yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_ltx25_serve_p2/summary.json) |
| **`Wan 2.2 T2V`** | FP8 All-HBM (`U=8`) | `480p` (`832×480`) `5s 81f` (`4`-step) | **`3.387 RPM`** | **`13.548 RPM`** | `3` slices (`12` hosts / `48` chips) | [`wan22.py#L249`](./scripts/run_v7x_wan22_serving.py#L249-L425) · [`#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) · [`Pod0`](./results/v7x_flex_wan22_serve_p3/pod_0_results.json) |
| **`Wan 2.2 I2V`** | FP8 All-HBM (`U=8`) | `480p` (`832×480`) `5s 81f` (`4`-step, `c=8`) | `2.640 RPM` | **`10.633 RPM`** | `3` slices (`12` hosts / `48` chips) | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`Wan 2.2 I2V`** | FP8 All-HBM (`U=8`) | `720p` (`1280×720`) `5s 81f` (`4`-step, `c=8`) | `1.105 RPM` | **`4.431 RPM`** | `7` slices (`28` hosts / `112` chips) | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`Wan 2.2 I2V`** | FP8 All-HBM (`U=8`) | `720p` `5s 81f` / `10s 161f` (`20`-step, `c=4`) | `0.311` / `0.118` | **`1.243` / `0.473 RPM`** | `25` / `64` slices | [`wan22.py#L762`](./scripts/run_v7x_wan22_serving.py#L762-L885) · [`yaml`](./k8s/v7x-16-flex-wan22-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`JSON`](./results/v7x_flex_wan22_serve_p3/summary.json) |
| **`MiniMax-H3`** | FP8 + Pallas FlashAttn | `768p` (`1344×768`) `15s` (`4`-step, `175.51s`) | **`0.342 RPM`** | **`1.368 RPM`** | `22` slices (`88` hosts / `352` chips) | [`minimax.py#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) · [`#L1270`](./scripts/run_v7x_minimax_h3_serving.py#L1270-L1420) · [`yaml`](./k8s/v7x-16-flex-minimax-h3-serving.yaml) | [`Report`](./BENCHMARK_RESULTS.md) · [`Pod1 JSON`](./results/v7x_flex_minimax_h3_serve_p1/pod_1_results.json) |

---

## 2. Optimization Summary

| Optimization | Change Over Default Baseline | Code Reference | Measured Effect |
|---|---|---|---|
| **All-HBM FP8 (`--dit-cpu-offload=false`, `--quantization fp8`)** | Kept DiT, text encoders, and VAEs resident in `768 GiB`/host HBM3e instead of CPU PCIe offloading | [`build_tpu_server_env#L789`](./scripts/run_v7x_ltx25_serving.py#L789-L878) · [`wan22.py#L511`](./scripts/run_v7x_wan22_serving.py#L511-L600) | Improved throughput by `+47%` to `+54%` over BF16 (`23.58s` vs `34.21s` on `LTX-2.5` `544p`; `17.55s` vs `31.68s` on `Wan 2.2` `480p`) |
| **Pallas FlashAttention (`SGLANG_TPU_SPLASH_VARLEN=1`)** | Routed `TpuSplashAttentionImpl.forward_varlen` through `v7x` `64 MiB` VMEM Pallas `flash_attention` (`256×256` tiles) instead of fallback `SDPA` | [`apply_sglang_tpu_runtime_patches#L246`](./scripts/run_v7x_minimax_h3_serving.py#L246-L420) | Avoided $O(S^2)$ attention score materialization on `MiniMax-H3` `768p 15s` (`175.51s` median E2E, `86.82 GiB` peak HBM) |
| **Single-Host Metadata Isolation (`DP=4 × Ulysses=8`)** | `LD_PRELOAD=/workspace/metadata_hook.so` + `mock_metadata.py` (`CHIPS_PER_HOST_BOUNDS=2,2,1`) isolates each 4-chip host on `2x2x4` | [`setup_single_host_metadata_hook#L694`](./scripts/run_v7x_ltx25_serving.py#L694-L722) | `4.00×–4.03×` scaling from 1 host (`4` chips) to 4 hosts (`16` chips) |

---

## 3. Repository Structure & Reproduction

- [`BENCHMARK_RESULTS.md`](./BENCHMARK_RESULTS.md) — Complete `109`-point benchmark report across `MiniMax-H3`, `LTX-2.5`, and `Wan 2.2`
- [`docker/Dockerfile.sglang_tpu`](./docker/Dockerfile.sglang_tpu), [`docker/prep_sglang_tpu.py`](./docker/prep_sglang_tpu.py), [`docker/Dockerfile.vllm_omni`](./docker/Dockerfile.vllm_omni), [`docker/prep_vllm_omni.py`](./docker/prep_vllm_omni.py)
- [`scripts/run_v7x_minimax_h3_serving.py`](./scripts/run_v7x_minimax_h3_serving.py) — `MiniMax-H3` (`FL2VA` `i2v` + `t2va`) & `FastH3` benchmark harness
- [`scripts/run_v7x_ltx25_serving.py`](./scripts/run_v7x_ltx25_serving.py) — `LTX-2.5` (including Two-Stage `1080p` `20s` `481f` workload) benchmark harness
- [`scripts/run_v7x_wan22_serving.py`](./scripts/run_v7x_wan22_serving.py) — `Wan 2.2 A14B` (`I2V` & `T2V`) benchmark harness
- [`scripts/stage_all_video_models_gcs.py`](./scripts/stage_all_video_models_gcs.py) — Parallel Hugging Face Hub to regional GCS checkpoint stager
- [`k8s/v7x-16-flex-minimax-h3-serving.yaml`](./k8s/v7x-16-flex-minimax-h3-serving.yaml), [`k8s/v7x-16-flex-ltx25-serving.yaml`](./k8s/v7x-16-flex-ltx25-serving.yaml), [`k8s/v7x-16-flex-wan22-serving.yaml`](./k8s/v7x-16-flex-wan22-serving.yaml)
- [`results/`](./results/) — Per-pod & slice JSON outputs (`v7x_flex_minimax_h3_serve_p1`, `v7x_flex_ltx25_serve_p2`, `v7x_flex_wan22_serve_p3`)

```bash
# 1. Upload serving benchmark scripts to regional GCS
gcloud storage cp scripts/run_v7x_*.py gs://YOUR_GCS_BUCKET/video-serving-scripts/

# 2. Launch MiniMax-H3, LTX-2.5, and Wan 2.2 jobs on TPU v7x-16 (2x2x4)
kubectl apply -f k8s/v7x-16-flex-minimax-h3-serving.yaml
kubectl apply -f k8s/v7x-16-flex-ltx25-serving.yaml
kubectl apply -f k8s/v7x-16-flex-wan22-serving.yaml
```
