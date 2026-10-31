# Cloud TPU `v7x` (`2x2x1` Single-Host & `2x2x4` `v7x-16` Slice) Video+Audio Diffusion Serving (`MiniMax-H3`, `LTX-2.5`, `Wan 2.2 A14B`)

Reproducible serving infrastructure, containers, Kubernetes Indexed Job manifests, and multi-configuration benchmark harnesses for **multimodal video + stereo audio diffusion models** on **Cloud TPU `v7x` (Ironwood)** using **`sglang-torchtpu`** (`TpuOOTPlatform` + `tpu_splash_attn` Pallas Mosaic FlashAttention) and **`vllm-torchtpu` (`vLLM-Omni` v0.32.0)**:

1. **Priority #1 — [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) (`FL2VA` Joint Video+Audio `t2va` & `fl2va` I2V) & [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)**
   - **Target Spec:** `i2v 768P 15s` (`1344×768`, `15.0s` @ `24 fps` = `360` frames + `32 kHz` stereo AAC audio, `4`-step distilled / `12`-step / `25`-step / `49`-step) + `5s` (`124f`), `10s` (`244f`), `9:16` Portrait (`768×1344`), and `512p` (`896×512`).
   - **Configurations:** Single-host `4x TPU v7x` chips (`8` TensorCores, `Ulysses=8`) & `2x TPU v7x` chips (`Ulysses=4`), `BF16` vs Online `FP8` (`--quantization fp8`), `Pallas Mosaic FlashAttention varlen` (`SGLANG_TPU_SPLASH_VARLEN=1`) vs stock `SDPA varlen`, Precomputed vs Online `AdaLN`, and `16`-chip `v7x-16` `DP=4 × Ulysses=8` 30-RPM slice sizing.
2. **Priority #2 — [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) (`22B` Asymmetric Dual-Stream Video+Audio DiT + `2×` Diffusion Decoder Upscaler)**
   - **Target Specs:** Standard `8`-step distilled (`960×544` & `1280×704` `720p` across `5s` `121f`, `10s` `241f`, `20s` `481f`), `20`-step dev schedule, and **Two-Stage `1080p` (`1920×1088`, `use_diffusion_decoder=True`) `20s` (`481` frames @ `24 fps`, `bs=1`) Kling-V4 Proxy**.
   - **Configurations:** `sglang-torchtpu` (`BF16` vs `FP8`, `Ulysses=8` vs `CFG=2 × Ulysses=4`, `4`-chip vs `2`-chip) and `vllm-torchtpu` (`vLLM-Omni` `--omni --usp 8`), plus `16`-chip `DP=4` concurrency sweep (`c=4, 8`).
3. **Priority #3 — [`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) & [`Wan-AI/Wan2.2-T2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (`28.6B` Total / `14.3B` Active Two-Stage High/Low-Noise MoE DiT)**
   - **Target Specs:** `480p` (`832×480`, `81f` `5s` @ `16 fps`), `720p` (`1280×720`, `81f` `5s` & `161f` `10s`), `720p` `9:16` Portrait (`720×1280`, `81f`), across `4`-step fast, `20`-step, and `40`-step schedules.
   - **Configurations:** `sglang-torchtpu` (`BF16` & `FP8`, `Ulysses=8` & `CFG=2 × Ulysses=4`) and `vllm-torchtpu` (`vLLM-Omni` `--omni --usp 8`), plus `16`-chip `DP=4` slice throughput.

---

## 1. Key TPU `v7x` Serving Architecture & Optimizations

| Feature | Implementation | Impact |
|---|---|---|
| **Single-Host `2x2x1` Isolation on `2x2x4` Multi-Host Slices** | `LD_PRELOAD=/workspace/metadata_hook.so` + local `mock_metadata.py` (`TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`) | Runs **4 independent `4`-chip (`8`-TensorCore, `768 GiB` HBM3e) servers** simultaneously on one `v7x-16` slice (`DP=4 × Ulysses=8`) with zero cross-host ICI barriers |
| **Hardware Pallas Mosaic FlashAttention (`SGLANG_TPU_SPLASH_VARLEN=1`)** | Routes `TpuSplashAttentionImpl.forward_varlen` through `v7x` `64 MiB` VMEM Pallas `flash_attention` (`256×256` MXU blocks) instead of fallback `SDPA` | Eliminates $O(S^2)$ attention HBM materialization at `MiniMax-H3` `768p 15s` (`S \approx 95\text{K}` tokens) |
| **TPU Seeded Generator Deadlock Prevention** | Passes `"generator_device": "cpu"` in `/v1/videos` requests | Avoids GIL deadlock in blocking host-to-device RNG seed copy on `torch_tpu` |
| **850 GiB `/dev/shm` RAM-Disk Checkpoint Staging** | Multi-threaded HTTP Range streaming from regional GCS (`us-central1`) into `/dev/shm` | Hydrates `160–288 GB` diffusion + VLM encoder checkpoints in `<35s` with zero disk bottlenecks |

> **Compatibility with TPU `v5p-16` / `v5p-32`:** Each `v5p` host (`4` chips = `8` TensorCores = `384 GiB` HBM2e) also fits all three models (`44–114 GB` BF16) on a single host (`DP=4` on `v5p-16`, `DP=8` on `v5p-32`) in `BF16`. Because `tpu_splash_attn.py` guards its `64 MiB`-VMEM Pallas kernel to `TPU7x`, `v5p` (`16 MiB` VMEM, `128×128` MXU, BF16-native) automatically falls back to XLA `SDPA` + `Ulysses=8` (or `Ulysses=16` across 2 hosts for ultra-long `15s–20s` sequences).

---

## 2. Repository Structure

```text
├── README.md
├── docker/
│   ├── Dockerfile.sglang_tpu                   # TorchTPU 0.2.0 + sglang-torchtpu + Pallas varlen patch + ffmpeg
│   ├── prep_sglang_tpu.py                      # ABI constraint lock + TOML-verified sglang-torchtpu setup
│   ├── Dockerfile.vllm_omni                    # TorchTPU 0.2.0 + vllm-torchtpu v0.32.0 + vLLM-Omni (CPU/TPU target)
│   └── prep_vllm_omni.py                       # ABI constraint lock for vLLM-Omni on TPU
├── scripts/
│   ├── stage_all_video_models_gcs.py           # Parallel HF Hub -> regional GCS stager for MiniMax-H3, FastH3, LTX-2.5, Wan 2.2
│   ├── run_v7x_minimax_h3_serving.py           # Priority #1: MiniMax-H3 (FL2VA i2v + t2va) & FastH3 4-step VSA benchmark suite
│   ├── run_v7x_ltx25_serving.py                # Priority #2: LTX-2.5 (incl. 1080p 20s Kling-V4 proxy) across sglang & vLLM-Omni
│   └── run_v7x_wan22_serving.py                # Priority #3: Wan 2.2 A14B (I2V & T2V) across sglang & vLLM-Omni
├── k8s/
│   ├── v7x-16-flex-minimax-h3-serving.yaml     # Indexed Job (4x4=16 v7x chips) for Priority #1 MiniMax-H3 + FastH3
│   ├── v7x-16-flex-ltx25-serving.yaml          # Indexed Job (4x4=16 v7x chips) for Priority #2 LTX-2.5
│   └── v7x-16-flex-wan22-serving.yaml          # Indexed Job (4x4=16 v7x chips) for Priority #3 Wan 2.2 A14B
└── results/                                    # Auto-synced Markdown & JSON benchmark summaries
```

---

## 3. Quickstart

```bash
# 1. Upload serving benchmark harnesses to regional GCS
gcloud storage cp scripts/run_v7x_*.py gs://ank-usc1-bucket-314837540096/video-serving-scripts/

# 2. Run Priority #1: MiniMax-H3 (i2v 768p 15s + input/config matrix) & FastH3 on steady Flex v7x-16
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v7x apply -f k8s/v7x-16-flex-minimax-h3-serving.yaml

# 3. Run Priority #2: LTX-2.5 (incl. Two-Stage 1080p 20s Kling-V4 proxy) on steady Flex v7x-16
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v7x apply -f k8s/v7x-16-flex-ltx25-serving.yaml

# 4. Run Priority #3: Wan 2.2 A14B (I2V & T2V) on steady Flex v7x-16
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v7x apply -f k8s/v7x-16-flex-wan22-serving.yaml
```
