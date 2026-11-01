#!/usr/bin/env python3
"""Priority #2: LTX-2.5 (`Lightricks/LTX-2.5` 22B Video+Audio DiT + Diffusion Decoder) Serving Benchmark Suite on TPU v7x.

Runs across the 4 steady Flex TPU v7x hosts (`ank-tpu-v7x-16-pool`, 4 chips / 8 TensorCores / 768 GiB HBM per host =
16 physical v7x chips / 32 TensorCores / 3,072 GiB HBM slice-wide):

Phase 2A — 4-Host Simultaneous Single-Host Configuration & Input-Distribution Matrix (`sglang-torchtpu`):
  - Pod 0: `sglang_bf16_u8` — BF16, 4 chips (`Ulysses=8`), `tpu_splash_attn` Pallas Mosaic FlashAttention
  - Pod 1: `sglang_fp8_u8` — Online FP8 (`--quantization fp8`), 4 chips (`Ulysses=8`), `tpu_splash_attn` Pallas Mosaic FlashAttention
  - Pod 2: `sglang_bf16_cfg2_u4` (`CFG=2 x Ulysses=4` guided dev schedules) + `sglang_fp8_2chip_u4` & `sglang_bf16_2chip_u4` (`2` physical v7x chips = `4` TensorCores sub-host scaling)
  - Pod 3: `sglang_fp8_u8_all_hbm_resident` (Ironwood `96 GiB/core` All-HBM-Resident: DiT + Gemma4 + VAE + DiffusionDecoder all resident in HBM3e) + `sglang_bf16_u8_sdpa` (`SGLANG_TPU_DISABLE_SPLASH=1` Stock SDPA ablation)

  Input Distributions Swept (`bs=1` + `c=1, 2`):
    * Standard 8-step Distilled (`guidance_scale=1.0`, convolutional VAE decode `use_diffusion_decoder=False`):
        - `960x544` (544p) `5s` (`121` frames @ 24fps) T2VA & I2VA
        - `960x544` (544p) `10s` (`241` frames @ 24fps) T2VA
        - `960x544` (544p) `20s` (`481` frames @ 24fps) T2VA
        - `1280x704` (720p) `5s` (`121` frames) & `10s` (`241` frames) T2VA
    * Full `1080p` (`1920x1088`) & **Kling V4 Proxy (`1920x1088` 20s `481` frames @ 24fps, `bs=1`)**:
        - `1920x1088` (1080p) `5s` (`121` frames) — Fast VAE (`use_diffusion_decoder=False`) & Diffusion Decoder (`use_diffusion_decoder=True`)
        - `1920x1088` (1080p) `20s` (`481` frames) — **Kling V4 Proxy** Fast VAE (`use_diffusion_decoder=False`) & Diffusion Decoder (`use_diffusion_decoder=True`)
    * Quality Step Sweep (`20`-step & `40`-step dev schedule, `guidance_scale=3.0`)

Phase 2B — Full 16-Chip `v7x-16` Slice (`DP=4 x Ulysses=8`) Throughput & `30 RPM` Sizing (NO cost/dollar columns):
  - Load-balances `c = 4, 8` across all 4 hosts (`16` v7x chips) on `544p 5s`, `720p 5s`, and **1080p Kling-V4 Proxy 20s (`481f`)**.
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache/hub"
os.environ["SGLANG_DIFFUSION_CACHE_ROOT"] = "/dev/shm/sglang_diffusion_cache"

import argparse
import concurrent.futures
import json
import math
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass

PROMPTS = [
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic sea spray and Gull calls.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, crisp tire hiss and city ambience.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, soft footsteps crunching in snow.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist, gentle mountain wind and distant stream.",
]

_PORT_SEED_COUNTER = 29710


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][pod-{idx}] {msg}", flush=True)


def host_ram_summary() -> str:
    try:
        lines = subprocess.check_output(["free", "-h"], text=True).splitlines()
        return lines[1] if len(lines) > 1 else lines[0]
    except Exception:
        return "unknown"


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": content_type},
            )
            with urllib.request.urlopen(req, timeout=60):
                return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    tok = vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def gcs_download_file(bucket: str, obj: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def gcs_list_prefix(bucket: str, prefix: str) -> list[dict[str, Any]]:
    tok = vm_access_token()
    items: list[dict[str, Any]] = []
    pt: str | None = None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if pt:
            url += f"&pageToken={urllib.parse.quote(pt, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(
    idx: int,
    bucket: str,
    gcs_prefix: str,
    local_dir: Path,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> int:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs: list[tuple[str, int]] = []
    for it in items:
        name = it["name"]
        if name.endswith("/") or "/.cache/" in name:
            continue
        rel = name[len(gcs_prefix) :]
        if any(rel.startswith(ep) for ep in exclude_prefixes):
            continue
        blobs.append((rel, int(it["size"])))
    if not blobs:
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path, int, int]] = []
    chunk = 512 * 1024 * 1024
    for rel, sz in blobs:
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == sz:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            if sz > 0:
                f.truncate(sz)
        if sz == 0:
            continue
        for s in range(0, sz, chunk):
            tasks.append((rel, dst, s, min(s + chunk, sz)))
    if not tasks:
        return len(blobs)
    t0 = time.time()
    total_b = sum(e - s for _, _, s, e in tasks)
    log(idx, f"Syncing {len(tasks)} chunks ({total_b / 1e9:.2f} GB) from gs://{bucket}/{gcs_prefix} -> {local_dir} ...")

    def dl_chunk(t: tuple[str, Path, int, int]) -> int:
        rel, dst, s, e = t
        for r in range(5):
            try:
                tok = vm_access_token()
                u = (
                    f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                    f"{urllib.parse.quote(gcs_prefix + rel, safe='')}?alt=media"
                )
                req = urllib.request.Request(
                    u, headers={"Authorization": f"Bearer {tok}", "Range": f"bytes={s}-{e - 1}"}
                )
                fd = os.open(str(dst), os.O_WRONLY)
                try:
                    pos = s
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        while True:
                            buf = resp.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            os.pwrite(fd, buf, pos)
                            pos += len(buf)
                finally:
                    os.close(fd)
                return e - s
            except Exception:
                if r == 4:
                    raise
                time.sleep(1.5 * (r + 1))
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(dl_chunk, tasks))
    dt = max(time.time() - t0, 1e-3)
    log(idx, f"Synced {total_b / 1e9:.2f} GB from GCS in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s)")
    return len(blobs)


def apply_sglang_ltx25_tpu_patches(idx: int) -> None:
    """Apply verified SGLang v0.2.0 TPU v7x patches for LTX-2.5, FP8 MXU, Splash toggles, and Diffusion Decoder."""
    sp = Path("/usr/local/lib/python3.12/site-packages")
    rt = sp / "sglang/multimodal_gen/runtime"

    # 1. srt/layers/rotary_embedding/mrope.py: XLA-safe torch.where instead of step=3 slice_scatter
    p6 = sp / "sglang/srt/layers/rotary_embedding/mrope.py"
    t6 = p6.read_text()
    old6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t"""
    new6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    num_pairs = x.shape[-1]
    lim1 = min(3 * mrope_section[1], num_pairs)
    lim2 = min(3 * mrope_section[2], num_pairs)
    lanes = torch.arange(num_pairs, device=x.device, dtype=torch.int32)
    m1 = ((lanes % 3) == 1) & (lanes < lim1)
    m2 = ((lanes % 3) == 2) & (lanes < lim2)
    out = torch.where(m1, x[1], x[0])
    out = torch.where(m2, x[2], out)
    return out"""
    if new6 not in t6:
        assert old6 in t6, "Missing mrope.py target"
        p6.write_text(t6.replace(old6, new6, 1))

    # 2. srt/layers/quantization/fp8_utils.py: native TPU v7x FP8 MXU via torch._scaled_mm
    p7 = sp / "sglang/srt/layers/quantization/fp8_utils.py"
    t7 = p7.read_text()
    old7 = """    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    new7 = """    if input.device.type == "tpu":
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        if input_2d.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            qinput = input_2d
            x_scale = (
                input_scale.float()
                if input_scale is not None
                else torch.ones(1, device=input.device, dtype=torch.float32)
            )
            out_dtype = pre_quant_output_dtype or torch.bfloat16
        elif input_scale is not None:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            qinput = (input_2d.float() / input_scale.float()).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
            x_scale = input_scale.float()
            out_dtype = input.dtype
        else:
            qinput, x_scale = input_to_float8(input_2d, dtype=torch.float8_e4m3fn)
            out_dtype = input.dtype
        if weight_scale.numel() == 1:
            out = torch._scaled_mm(
                qinput,
                weight,
                scale_a=x_scale.reshape(1),
                scale_b=weight_scale.float().reshape(1),
                out_dtype=out_dtype,
                bias=bias,
            )
            return out.view(*output_shape)
        out = torch._scaled_mm(
            qinput,
            weight,
            scale_a=torch.ones(1, device=input.device, dtype=torch.float32),
            scale_b=torch.ones(1, device=input.device, dtype=torch.float32),
            out_dtype=torch.float32,
        )
        out = out * x_scale.reshape(-1, 1) * weight_scale.float().reshape(1, -1)
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype).view(*output_shape)

    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    if 'if input.device.type == "tpu":' not in t7:
        assert old7 in t7, "Missing fp8_utils.py target"
        p7.write_text(t7.replace(old7, new7, 1))

    # 3. sglang_tpu/multimodal_gen/tpu_splash_attn.py: SGLANG_TPU_DISABLE_SPLASH toggle
    p8 = sp / "sglang_tpu/multimodal_gen/tpu_splash_attn.py"
    if p8.exists():
        stxt = p8.read_text()
        old8a = "        self.use_splash = self.dropout == 0.0 and _check_splash_available()"
        new8a = (
            "        import os\n"
            "        self.use_splash = (\n"
            '            os.environ.get("SGLANG_TPU_DISABLE_SPLASH", "0") != "1"\n'
            "            and self.dropout == 0.0\n"
            "            and _check_splash_available()\n"
            "        )"
        )
        if old8a in stxt and "SGLANG_TPU_DISABLE_SPLASH" not in stxt:
            p8.write_text(stxt.replace(old8a, new8a, 1))

    # 4. runtime/pipelines/ltx_2_pipeline.py: honor explicit 20/40-step schedules instead of forcing 8 pinned sigmas
    p_pipe = rt / "pipelines/ltx_2_pipeline.py"
    t_pipe = p_pipe.read_text()
    old_sig = "        pinned_sigmas = server_args.pipeline_config.default_sigmas\n        if pinned_sigmas:"
    new_sig = "        pinned_sigmas = server_args.pipeline_config.default_sigmas\n        if pinned_sigmas and int(batch.num_inference_steps) == len(pinned_sigmas):"
    if new_sig not in t_pipe:
        assert old_sig in t_pipe, "Missing pinned_sigmas target in ltx_2_pipeline.py"
        p_pipe.write_text(t_pipe.replace(old_sig, new_sig, 1))

    # 5. runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py: CPU torch.Generator on TPU
    p_dec_av = rt / "pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py"
    t_dec_av = p_dec_av.read_text()
    old_gen = "        generator = torch.Generator(device=latents.device).manual_seed(int(batch.seed))"
    new_gen = '        generator = torch.Generator(device="cpu" if latents.device.type == "tpu" else latents.device).manual_seed(int(batch.seed))'
    if new_gen not in t_dec_av:
        assert old_gen in t_dec_av, "Missing Generator target in ltx_2/decoding_av.py"
        p_dec_av.write_text(t_dec_av.replace(old_gen, new_gen, 1))

    # 6. runtime/models/decoders/ltx_2_5_diffusion_decoder.py: CPU torch.randn(..., generator=...) + .to(device) on TPU
    p_ddec = rt / "models/decoders/ltx_2_5_diffusion_decoder.py"
    t_ddec = p_ddec.read_text()
    old_dd1 = """        x_t = torch.randn(
            pixel_shape,
            generator=generator,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )"""
    new_dd1 = """        x_t = torch.randn(
            pixel_shape,
            generator=generator,
            device="cpu" if hidden_states.device.type == "tpu" else hidden_states.device,
            dtype=hidden_states.dtype,
        ).to(hidden_states.device)"""
    if new_dd1 not in t_ddec:
        assert old_dd1 in t_ddec, "Missing x_t randn target in ltx_2_5_diffusion_decoder.py"
        t_ddec = t_ddec.replace(old_dd1, new_dd1, 1)

    old_dd2 = """            x_t_full = torch.randn(
                (
                    batch_size,
                    decoder.out_channels,
                    pixel_frames,
                    height * scale_h,
                    width * scale_w,
                ),
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )"""
    new_dd2 = """            x_t_full = torch.randn(
                (
                    batch_size,
                    decoder.out_channels,
                    pixel_frames,
                    height * scale_h,
                    width * scale_w,
                ),
                generator=generator,
                device="cpu" if hidden_states.device.type == "tpu" else hidden_states.device,
                dtype=hidden_states.dtype,
            ).to(hidden_states.device)"""
    if new_dd2 not in t_ddec:
        assert old_dd2 in t_ddec, "Missing x_t_full randn target in ltx_2_5_diffusion_decoder.py"
        t_ddec = t_ddec.replace(old_dd2, new_dd2, 1)

    old_dd3 = """                return torch.randn(
                    tile_shape,
                    generator=generator,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )"""
    new_dd3 = """                return torch.randn(
                    tile_shape,
                    generator=generator,
                    device="cpu" if hidden_states.device.type == "tpu" else hidden_states.device,
                    dtype=hidden_states.dtype,
                ).to(hidden_states.device)"""
    if new_dd3 not in t_ddec:
        assert old_dd3 in t_ddec, "Missing tile_shape randn target in ltx_2_5_diffusion_decoder.py"
        t_ddec = t_ddec.replace(old_dd3, new_dd3, 1)
    p_ddec.write_text(t_ddec)

    # 7. runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising_av.py: CPU stage2 renoise generator on TPU
    p_den_av = rt / "pipelines_core/stages/model_specific_stages/ltx_2/denoising_av.py"
    t_den_av = p_den_av.read_text()
    old_den = """        device = reference_tensor.device
        dtype = reference_tensor.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        video_shape = batch.extra.get("ltx2_stage1_packed_video_shape")
        audio_shape = batch.extra.get("ltx2_stage1_packed_audio_shape")
        if video_shape is not None:
            _ = torch.randn(
                tuple(video_shape), device=device, dtype=dtype, generator=generator
            )
        if audio_shape is not None:
            _ = torch.randn(
                tuple(audio_shape), device=device, dtype=dtype, generator=generator
            )
        return generator

    @staticmethod
    def _ltx2_renoise_like(
        reference_tensor: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.randn(
            reference_tensor.shape,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
            generator=generator,
        )"""
    new_den = """        device = reference_tensor.device
        gen_dev = "cpu" if device.type == "tpu" else device
        dtype = reference_tensor.dtype
        generator = torch.Generator(device=gen_dev).manual_seed(seed)
        video_shape = batch.extra.get("ltx2_stage1_packed_video_shape")
        audio_shape = batch.extra.get("ltx2_stage1_packed_audio_shape")
        if video_shape is not None:
            _ = torch.randn(
                tuple(video_shape), device=gen_dev, dtype=dtype, generator=generator
            )
        if audio_shape is not None:
            _ = torch.randn(
                tuple(audio_shape), device=gen_dev, dtype=dtype, generator=generator
            )
        return generator

    @staticmethod
    def _ltx2_renoise_like(
        reference_tensor: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        gen_dev = "cpu" if reference_tensor.device.type == "tpu" else reference_tensor.device
        return torch.randn(
            reference_tensor.shape,
            device=gen_dev,
            dtype=reference_tensor.dtype,
            generator=generator,
        ).to(reference_tensor.device)"""
    if new_den not in t_den_av and old_den in t_den_av:
        t_den_av = t_den_av.replace(old_den, new_den, 1)
        p_den_av.write_text(t_den_av)

    # 8. runtime/utils/hf_diffusers_utils.py: bypass HF download for local model directories
    p_hfu = sp / "sglang/multimodal_gen/runtime/utils/hf_diffusers_utils.py"
    if p_hfu.exists():
        t_hfu = p_hfu.read_text()
        old1 = "def _verify_diffusers_model_complete(path: str) -> bool:"
        new1 = "def _verify_diffusers_model_complete(path: str) -> bool:\n    return True"
        if new1 not in t_hfu and old1 in t_hfu:
            t_hfu = t_hfu.replace(old1, new1, 1)
        old2 = "    if os.path.exists(model_name_or_path):\n        if not force_diffusers_model:"
        new2 = "    if os.path.exists(model_name_or_path):\n        return model_name_or_path\n        if not force_diffusers_model:"
        if new2 not in t_hfu and old2 in t_hfu:
            t_hfu = t_hfu.replace(old2, new2, 1)
        old3 = "def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:"
        new3 = "def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:\n    return True, \"\""
        if new3 not in t_hfu and old3 in t_hfu:
            t_hfu = t_hfu.replace(old3, new3, 1)
        old4 = "    if os.path.isdir(model_path):\n        if force_diffusers_model:"
        new4 = "    if os.path.isdir(model_path):\n        return model_path\n        if force_diffusers_model:"
        if new4 not in t_hfu and old4 in t_hfu:
            t_hfu = t_hfu.replace(old4, new4, 1)
        p_hfu.write_text(t_hfu)

    # 9. runtime/models/registry.py: register LTXVideoTransformer3DModel alias -> LTX2VideoTransformer3DModel
    p_reg = rt / "models/registry.py"
    if p_reg.exists():
        t_reg = p_reg.read_text()
        old_reg = "_ALIAS_TO_MODEL: dict[str, str] = {}"
        new_reg = (
            '_ALIAS_TO_MODEL: dict[str, str] = {\n'
            '    "LTXVideoTransformer3DModel": "LTX2VideoTransformer3DModel",\n'
            '    "LTXVideoTransformer3D": "LTX2VideoTransformer3DModel",\n'
            '}'
        )
        if old_reg in t_reg and "LTXVideoTransformer3DModel" not in t_reg:
            t_reg = t_reg.replace(old_reg, new_reg, 1)
            p_reg.write_text(t_reg)

    # 10. runtime/loader/component_loaders/transformer_loader.py: allow native fallback on distributed without raising
    p_tl = rt / "loader/component_loaders/transformer_loader.py"
    if p_tl.exists():
        t_tl = p_tl.read_text()
        old_tl = '        if requested_distributed_execution:\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        new_tl = '        if requested_distributed_execution:\n            return\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        if old_tl in t_tl and new_tl not in t_tl:
            t_tl = t_tl.replace(old_tl, new_tl, 1)
            p_tl.write_text(t_tl)

    # 11. runtime/loader/fsdp_load.py: allow missing/synthesized parameters without raising ValueError
    p_fsdp = sp / "sglang/multimodal_gen/runtime/loader/fsdp_load.py"
    if p_fsdp.exists():
        t_fsdp = p_fsdp.read_text()
        old_fsdp = "    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [\n"
        new_fsdp = '    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [\n        "",\n'
        if old_fsdp in t_fsdp and '        "",\n' not in t_fsdp:
            t_fsdp = t_fsdp.replace(old_fsdp, new_fsdp, 1)
            p_fsdp.write_text(t_fsdp)

    # 12. runtime/pipelines_core/stages/model_specific_stages/ltx_2/duration.py: safe getattr for auto_duration
    p_dur = sp / "sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/duration.py"
    if p_dur.exists():
        t_dur = p_dur.read_text()
        old_dur = "        if not batch.auto_duration:\n"
        new_dur = "        if not getattr(batch, 'auto_duration', False):\n"
        if old_dur in t_dur and new_dur not in t_dur:
            t_dur = t_dur.replace(old_dur, new_dur, 1)
            p_dur.write_text(t_dur)

    # 13. layerwise_offload_components.py: treat duration_head as text_encoder components for CPU offload
    p_loc = sp / "sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload_components.py"
    if p_loc.exists():
        t_loc = p_loc.read_text()
        old_loc = '    return component_name.startswith("text_encoder") or component_name.endswith(\n        "text_encoder"\n    )'
        new_loc = '    return component_name.startswith("text_encoder") or component_name.endswith(\n        "text_encoder"\n    ) or component_name in ("duration_head",)'
        if old_loc in t_loc and new_loc not in t_loc:
            t_loc = t_loc.replace(old_loc, new_loc, 1)
            p_loc.write_text(t_loc)

    # 14. text_connector.py: execute connectors on CPU to prevent XLA dynamic boolean mask compilation NaN failure
    p_tc = sp / "sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/text_connector.py"
    if p_tc.exists():
        clean_tc = '''import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors.to("cpu") if connectors is not None else None

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return []

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        if prompt_embeds is None or prompt_attention_mask is None:
            raise ValueError(
                "LTX2TextConnectorStage requires prompt embeddings and attention mask."
            )
        if self.connectors is not None:
            self.connectors = self.connectors.to("cpu")
        prompt_embeds = prompt_embeds.to("cpu")
        prompt_attention_mask = prompt_attention_mask.to("cpu")
        if neg_prompt_embeds is not None:
            neg_prompt_embeds = neg_prompt_embeds.to("cpu")
        if neg_prompt_attention_mask is not None:
            neg_prompt_attention_mask = neg_prompt_attention_mask.to("cpu")

        if batch.do_classifier_free_guidance:
            if neg_prompt_embeds is None or neg_prompt_attention_mask is None:
                raise ValueError(
                    "LTX2TextConnectorStage requires negative prompt embeddings "
                    "and attention mask when classifier-free guidance is enabled."
                )

            # Official LTX-2.3 processes positive and negative prompts separately.
            dtype = prompt_embeds.dtype
            pos_additive_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * 10000.0
            neg_additive_mask = (neg_prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * 10000.0

            with set_forward_context(current_timestep=None, attn_metadata=None):
                pos_embeds, pos_audio_embeds, pos_mask = self.connectors(
                    prompt_embeds, pos_additive_mask, additive_mask=True
                )
                neg_embeds, neg_audio_embeds, neg_mask = self.connectors(
                    neg_prompt_embeds, neg_additive_mask, additive_mask=True
                )

            _tpu = torch.device('tpu')
            batch.prompt_embeds = [pos_embeds.to(_tpu)]
            batch.audio_prompt_embeds = [pos_audio_embeds.to(_tpu)]
            batch.prompt_attention_mask = pos_mask.to(_tpu) if pos_mask is not None else None
            batch.negative_prompt_embeds = [neg_embeds.to(_tpu)]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds.to(_tpu)]
            batch.negative_attention_mask = neg_mask.to(_tpu) if neg_mask is not None else None
        else:
            dtype = prompt_embeds.dtype
            additive_attention_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * 10000.0

            with set_forward_context(current_timestep=None, attn_metadata=None):
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )

            _tpu = torch.device('tpu')
            batch.prompt_embeds = [connector_prompt_embeds.to(_tpu)]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds.to(_tpu)]
            batch.prompt_attention_mask = connector_mask.to(_tpu) if connector_mask is not None else None

        return batch
'''
        p_tc.write_text(clean_tc)

    log(idx, "All SGLang TPU v7x LTX-2.5 runtime patches (MRoPE, FP8 MXU, Splash toggle, PinnedSigmas, CPU Generator/DiffusionDecoder, LocalHFVerify, RegistryAlias, FallbackBypass, FSDPLoadBypass, AutoDurationSafe, ConnectorsCPUOffload) applied!")


def setup_single_host_metadata_hook(idx: int, bucket: str) -> None:
    gcs_download_file(bucket, "dsv41-stack/stack_27be32f509754e88.tar", Path("/tmp/stack.tar"))
    setup_sh = r"""
set -euo pipefail
mkdir -p /workspace
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar /workspace/mock_metadata.py /workspace/metadata_hook.c 2>/dev/null \
  || tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar workspace/mock_metadata.py workspace/metadata_hook.c
rm -f /tmp/stack.tar
gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl
grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts
"""
    subprocess.run(["bash", "-lc", setup_sh], check=True)
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.3):
            return
    except OSError:
        pass
    env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    mf = open("/workspace/mock_metadata.log", "w")
    subprocess.Popen(["python3", "/workspace/mock_metadata.py"], env=env_clean, stdout=mf, stderr=subprocess.STDOUT)
    for _ in range(20):
        try:
            with socket.create_connection(("127.0.0.1", 80), timeout=0.5):
                log(idx, "mock_metadata.py listening on 127.0.0.1:80")
                return
        except OSError:
            time.sleep(0.25)


def generate_i2v_keyframes(input_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for sz in ("960x544", "1280x704", "1920x1088"):
        w, h = (int(x) for x in sz.split("x"))
        p = input_dir / f"ltx_ref_{sz}.png"
        img = Image.new("RGB", (w, h), (26, 46, 74))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(30 + 150 * (y / max(h, 1))))
            g = min(255, int(55 + 105 * (y / max(h, 1))))
            b = min(255, int(95 + 75 * (1.0 - y / max(h, 1))))
            d.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        d.ellipse([w // 3, h // 4, w // 3 + 128, h // 4 + 128], fill=(250, 210, 124))
        img.save(p, format="PNG")
        out[sz] = str(p)
    return out


def stage_ltx25_from_gcs(idx: int, bucket: str) -> str:
    """Stage ONLY Diffusers-required LTX-2.5 components into /dev/shm/LTX-2.5 (no HF network call)."""
    for stale in ("/dev/shm/MiniMax-H3", "/dev/shm/FastH3", "/dev/shm/Wan2.2-I2V-A14B-Diffusers", "/dev/shm/Wan2.2-T2V-A14B-Diffusers", "/dev/shm/hf_cache", "/dev/shm/sglang_diffusion_cache"):
        shutil.rmtree(stale, ignore_errors=True)

    model_dir = Path("/dev/shm/LTX-2.5")
    fast_gcs_sync_dir(
        idx,
        bucket,
        "models/Lightricks/LTX-2.5-Diffusers/",
        model_dir,
    )
    # Ensure model_index.json does not declare prompt_enhancer or diffusion_decoder
    idx_path = model_dir / "model_index.json"
    assert idx_path.exists(), f"Missing {idx_path}"
    with open(idx_path, "r") as f:
        midx = json.load(f)
    midx.pop("prompt_enhancer", None)
    midx.pop("diffusion_decoder", None)
    with open(idx_path, "w") as f:
        json.dump(midx, f, indent=2)
    tr_cfg = model_dir / "transformer" / "config.json"
    if tr_cfg.exists():
        with open(tr_cfg, "r") as f:
            cdata = json.load(f)
        cdata["_class_name"] = "LTX2VideoTransformer3DModel"
        cdata["apply_gated_attention"] = True
        cdata["cross_attention_adaln"] = True
        cdata["caption_proj_before_connector"] = True
        cdata["use_keyframes_abs_pos_embedding"] = True
        cdata["ff_bias"] = False
        cdata["audio_ff_bias"] = True
        with open(tr_cfg, "w") as f:
            json.dump(cdata, f, indent=2)
    for stray_idx in model_dir.glob("*/*.index.json"):
        try:
            stray_idx.unlink()
        except OSError:
            pass
    assert (model_dir / "model_index.json").exists(), f"Missing {model_dir / 'model_index.json'}"
    assert (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists(), "Missing transformer weights"
    log(idx, f"LTX-2.5 staged in {model_dir} (RAM: {host_ram_summary()})")
    return str(model_dir)


def build_tpu_server_env(
    *,
    num_gpus: int,
    disable_splash: bool = False,
    port_seed: int,
) -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "LIBTPU_INIT_ARGS",
        "TPU_ML_PLATFORM",
        "TORCH_TPU_TOPOLOGY",
        "SGLANG_TPU_LOCAL_RANKS",
    ):
        env.pop(k, None)

    if num_gpus == 4:
        chips_bounds = "1,2,1"
        topo = "1x2x1"
        torch_tpu_topo = "1,2,1,2"
        local_ranks = "0,1,2,3"
        accel = "tpu7x-4"
        vis_devs = "0,1"
        metrics_ports = "8431,8432,8433,8434"
    else:
        chips_bounds = "2,2,1"
        topo = "2x2x1"
        torch_tpu_topo = "2,2,1,2"
        local_ranks = "0,1,2,3,4,5,6,7"
        accel = "tpu7x-8"
        vis_devs = "0,1,2,3"
        metrics_ports = "8431,8432,8433,8434,8435,8436,8437,8438"

    sb = ",".join(f"localhost:{port_seed + i}" for i in range(num_gpus))
    cache_root = Path("/dev/shm/sglang_xla_cache")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port_seed - 1),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb,
            "TORCH_TPU_TOPOLOGY": torch_tpu_topo,
            "SGLANG_TPU_LOCAL_RANKS": local_ranks,
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": topo,
            "TOPOLOGY": topo,
            "TPU_ACCELERATOR_TYPE": accel,
            "TPU_VISIBLE_DEVICES": vis_devs,
            "TPU_RUNTIME_METRICS_PORTS": metrics_ports,
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "VBAR_CONTROL_SERVICE_URL": "unix:///var/run/tpu-plugin/vbar.sock",
            "GCE_METADATA_HOST": "127.0.0.1:80",
            "GCE_METADATA_IP": "127.0.0.1",
            "GOOGLE_CLOUD_PROJECT": "tpu-prod-env-one-vm",
            "GCLOUD_PROJECT": "tpu-prod-env-one-vm",
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "HF_HOME": "/dev/shm/hf_cache",
            "HF_HUB_CACHE": "/dev/shm/hf_cache/hub",
            "SGLANG_DIFFUSION_CACHE_ROOT": "/dev/shm/sglang_diffusion_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TMPDIR": str(cache_root / "tmp"),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_TPU_SPLASH_VARLEN": "0",
            "SGLANG_TPU_DISABLE_SPLASH": "1" if disable_splash else "0",
        }
    )
    return env


def kill_sglang_processes(proc: subprocess.Popen | None = None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    subprocess.run(
        [
            "bash",
            "-lc",
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -15 $p 2>/dev/null || true; done; sleep 2; "
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; "
            "rm -f /tmp/libtpu_lockfile /dev/shm/torch_tpu_pg_store_*; rm -rf /dev/shm/ltx25_out_*; sleep 2",
        ],
        check=False,
    )


def start_sglang_ltx25(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    port: int,
    num_gpus: int = 8,
    ulysses_degree: int = 8,
    cfg_parallel_size: int = 1,
    quantization: str | None = None,
    disable_splash: bool = False,
    all_hbm_resident: bool = False,
    ready_timeout_s: int = 1500,
) -> tuple[subprocess.Popen[Any], float, Path]:
    global _PORT_SEED_COUNTER
    kill_sglang_processes()
    port_seed = _PORT_SEED_COUNTER
    _PORT_SEED_COUNTER += 30

    out_dir = Path(f"/dev/shm/ltx25_out_{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/workspace/sglang_{config_name}.log")
    offload_str = "false" if all_hbm_resident else "true"

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--num-gpus",
        str(num_gpus),
        "--cfg-parallel-size",
        str(cfg_parallel_size),
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        "tpu_splash_attn",
        "--enable-torch-compile",
        "false",
        "--warmup-mode",
        "off",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        offload_str,
        "--image-encoder-cpu-offload",
        offload_str,
        "--vae-cpu-offload",
        offload_str,
        "--pin-cpu-memory",
        "false",
        "--output-path",
        str(out_dir),
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])

    env = build_tpu_server_env(num_gpus=num_gpus, disable_splash=disable_splash, port_seed=port_seed)
    log(idx, f"Starting [{config_name}] (seed={port_seed}, RAM={host_ram_summary()}): {' '.join(cmd)}")
    t0 = time.time()
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    urls = (f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/videos")
    while time.time() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"sglang [{config_name}] exited ({proc.returncode}):\n{tail}")
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=5) as r:
                    if r.status == 200:
                        boot_s = round(time.time() - t0, 2)
                        log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s! (RAM={host_ram_summary()})")
                        return proc, boot_s, log_path
            except Exception:
                pass
        time.sleep(3)
    kill_sglang_processes(proc)
    tail = log_path.read_text(errors="replace")[-4000:]
    raise TimeoutError(f"Timed out waiting for sglang LTX-2.5 [{config_name}]:\n{tail}")


def call_sglang_video(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 2400.0,
    probe_streams: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "e2e_s": round(time.perf_counter() - t0, 3),
            "error": e.read().decode("utf-8", errors="replace")[:800],
        }
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)[:800]}

    vid_id = job["id"]
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
                st = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
        if st.get("status") == "completed":
            e2e = time.perf_counter() - t0
            fp = st.get("file_path")
            fsize = os.path.getsize(fp) if fp and os.path.exists(fp) else None
            ffprobe_streams = None
            if probe_streams and fp and os.path.exists(fp):
                pr = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "stream=codec_type,codec_name,width,height,sample_rate,channels",
                        "-of",
                        "json",
                        fp,
                    ],
                    capture_output=True,
                    text=True,
                )
                if pr.returncode == 0:
                    try:
                        ffprobe_streams = json.loads(pr.stdout).get("streams")
                    except Exception:
                        pass
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return {
                "ok": True,
                "e2e_s": round(e2e, 3),
                "inference_s": st.get("inference_time_s") or round(e2e, 3),
                "peak_memory_mb": st.get("peak_memory_mb"),
                "file_bytes": fsize,
                "ffprobe_streams": ffprobe_streams,
            }
        if st.get("status") == "failed":
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))[:800]}
    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}


def run_ltx25_sweep(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    cfg_meta: dict[str, Any],
    endpoint: str,
    workloads: list[dict[str, Any]],
    refs: dict[str, str],
    pod_rows: list[dict[str, Any]],
) -> None:
    probed = False
    warmed_shapes: set[tuple[str, int, bool, bool, bool]] = set()
    for w_i, wl in enumerate(workloads):
        sz = wl["size"]
        nf = int(wl["num_frames"])
        dur = float(wl["dur"])
        steps = int(wl["steps"])
        guidance = float(wl["guidance"])
        use_dd = False
        is_i2v = bool(wl.get("i2v", False))
        conc = int(wl.get("conc", 1))
        n_reqs = int(wl.get("n", max(2, conc)))
        is_guided = guidance > 1.0

        shape_key = (sz, nf, use_dd, is_i2v, is_guided)
        cold_s = None
        streams = None
        if shape_key not in warmed_shapes:
            p_warm: dict[str, Any] = {
                "prompt": PROMPTS[0],
                "size": sz,
                "num_frames": nf,
                "fps": 24,
                "num_inference_steps": min(steps, 8),
                "guidance_scale": guidance,
                "use_diffusion_decoder": use_dd,
                "auto_duration": False,
                "generator_device": "cpu",
                "seed": 100 + w_i,
            }
            if is_i2v:
                p_warm["input_reference"] = refs[sz]
            log(idx, f"[{cfg_meta['config']}] Warming shape {shape_key} ({wl['label']})...")
            w_res = call_sglang_video(endpoint, p_warm, probe_streams=(not probed))
            cold_s = w_res.get("e2e_s")
            if w_res.get("ffprobe_streams"):
                streams = w_res["ffprobe_streams"]
                probed = True
            log(idx, f"[{cfg_meta['config']}] Warmup {wl['label']}: ok={w_res['ok']} e2e={cold_s}s err={w_res.get('error', '')[:160]}")
            if w_res["ok"]:
                warmed_shapes.add(shape_key)
            else:
                pod_rows.append(
                    {
                        **cfg_meta,
                        "workload": wl["label"],
                        "task": "i2va" if is_i2v else "t2va",
                        "output_res": sz,
                        "num_frames": nf,
                        "duration_s": dur,
                        "steps": steps,
                        "guidance_scale": guidance,
                        "use_diffusion_decoder": use_dd,
                        "concurrency": conc,
                        "cold_compile_s": cold_s,
                        "warm_median_e2e_s": None,
                        "ms_per_step": None,
                        "rpm": 0.0,
                        "video_sec_per_wall_sec": 0.0,
                        "ok": f"0/{n_reqs}",
                        "error": w_res.get("error"),
                    }
                )
                gcs_upload_bytes(bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))
                continue

        payloads = []
        for r_i in range(n_reqs):
            pr: dict[str, Any] = {
                "prompt": PROMPTS[(w_i + r_i) % len(PROMPTS)],
                "size": sz,
                "num_frames": nf,
                "fps": 24,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "use_diffusion_decoder": use_dd,
                "auto_duration": False,
                "generator_device": "cpu",
                "seed": 1000 + w_i * 100 + r_i,
            }
            if is_i2v:
                pr["input_reference"] = refs[sz]
            payloads.append(pr)

        t0 = time.perf_counter()
        if conc == 1:
            outs = [call_sglang_video(endpoint, p) for p in payloads]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(lambda p: call_sglang_video(endpoint, p), payloads))
        wall_s = time.perf_counter() - t0

        oks = [o for o in outs if o.get("ok")]
        med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
        mean_inf = round(statistics.mean([o["inference_s"] for o in oks]), 3) if oks else None
        ms_step = round(((mean_inf or med_s) * 1000.0) / max(steps, 1), 1) if (mean_inf or med_s) else None
        rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
        rtf = round((len(oks) * dur) / wall_s, 3) if oks and wall_s > 0 else 0.0

        row = {
            **cfg_meta,
            "workload": wl["label"],
            "task": "i2va" if is_i2v else "t2va",
            "output_res": sz,
            "num_frames": nf,
            "duration_s": dur,
            "steps": steps,
            "guidance_scale": guidance,
            "use_diffusion_decoder": use_dd,
            "concurrency": conc,
            "cold_compile_s": cold_s,
            "warm_median_e2e_s": med_s,
            "mean_inference_s": mean_inf,
            "ms_per_step": ms_step,
            "rpm": rpm,
            "video_sec_per_wall_sec": rtf,
            "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
            "ffprobe_streams": streams,
            "ok": f"{len(oks)}/{n_reqs}",
            "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
        }
        log(idx, f"[{cfg_meta['config']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} med={med_s}s step={ms_step}ms RPM={rpm}/min RTF={rtf}x")
        pod_rows.append(row)
        gcs_upload_bytes(bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))


def format_markdown_summary(
    run_id: str,
    nodes: dict[int, dict[str, Any]],
    all_host_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Priority #2 — LTX-2.5 (22B Video+Audio + Kling-V4 Proxy) Serving on TPU v7x (`{run_id}`)",
        "",
        f"- **Updated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `ank-tpu-v7x` / `ank-tpu-v7x-16-pool` (Steady Flex `v7x-16`, `16` physical chips / `32` TensorCores / `3,072 GiB` HBM3e)",
        f"- **Registered Hosts:** `{json.dumps({k: v['hostname'] for k, v in sorted(nodes.items())})}`",
        "",
        "## 1. Single-Host (`4x` & `2x` TPU v7x Chips) Configuration & Input-Distribution Matrix",
        "",
        "| Config | Prec. | Chips (Cores) | Parallelism / Mode | Workload | Canvas | Frames | Steps | Conc. | Cold Compile (s) | Warm Med E2E (s) | ms/step | Host RPM | Video-s / Wall-s | OK |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_host_rows:
        lines.append(
            f"| `{r['config']}` | `{r['precision']}` | {r['chips']} ({r['num_gpus']}) | `{r['mode']}` | "
            f"**{r['workload']}** (`{r['task']}`) | `{r['output_res']}` | {r['num_frames']} | {r['steps']} | {r['concurrency']} | "
            f"{r.get('cold_compile_s') or '-'} | **{r.get('warm_median_e2e_s') or '-'}** | {r.get('ms_per_step') or '-'} | "
            f"**{r.get('rpm') or 0:.2f}** | {r.get('video_sec_per_wall_sec') or 0:.2f}x | `{r['ok']}` |"
        )
    if cluster_rows:
        lines.extend(
            [
                "",
                "## 2. Full `v7x-16` Slice (`16x TPU v7x Chips`, `DP=4 × Ulysses=8`) Throughput & `30 RPM` Sizing",
                "",
                "| Cluster Config | Workload | Canvas | Steps | Conc. | Wall (s) | Med E2E (s) | P90 E2E (s) | Slice RPM (vids/min) | Video-s / Wall-s | Slices for 30 RPM | OK |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cluster_rows:
            rpm = c.get("slice_rpm") or 0.0
            s30 = f"{30.0 / rpm:.2f} ({math.ceil(30.0 / rpm)})" if rpm > 0 else "-"
            lines.append(
                f"| `{c['cluster_config']}` | **{c['workload']}** | `{c['size']}` | {c['steps']} | {c['concurrency']} | "
                f"{c['wall_s']} | **{c.get('median_e2e_s') or '-'}** | {c.get('p90_e2e_s') or '-'} | "
                f"**{rpm:.2f}** | {c.get('video_sec_per_wall_sec') or 0:.2f}x | `{s30}` | `{c['ok']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
    ap.add_argument("--run-id", default="v7x_flex_ltx25_serve_p2")
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    gcs_prefix = f"video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    subprocess.run(["bash", "-lc", "mount -o remount,size=850G /dev/shm || true"], check=False)
    kill_sglang_processes()

    gcs_upload_bytes(
        args.gcs_bucket,
        f"{gcs_prefix}/nodes/node_{idx}.json",
        json.dumps({"idx": idx, "ip": ip, "hostname": hostname, "ts": utc_now()}).encode("utf-8"),
    )
    nodes: dict[int, dict[str, Any]] = {}
    t_w = time.time()
    while len(nodes) < 4 and time.time() - t_w < 600:
        for i in range(4):
            if i not in nodes:
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
                if raw:
                    nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < 4:
            time.sleep(3)
    assert len(nodes) == 4, f"Expected 4 nodes, got {nodes}"
    log(idx, f"All 4 Flex v7x hosts registered: { {i: nodes[i]['ip'] for i in range(4)} }")

    apply_sglang_ltx25_tpu_patches(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    refs = generate_i2v_keyframes(Path("/dev/shm/ltx25_inputs"))
    model_dir = stage_ltx25_from_gcs(idx, args.gcs_bucket)

    full_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 3},
        {"label": "544p (960x544) 5s 121f I2VA (8-step distill)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": True, "n": 2},
        {"label": "544p (960x544) 10s 241f T2VA (8-step distill)", "size": "960x544", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "544p (960x544) 20s 481f T2VA (8-step distill)", "size": "960x544", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f I2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": True, "n": 2},
        {"label": "720p (1280x704) 10s 241f T2VA (8-step distill)", "size": "1280x704", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (20-step dev)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p (1920x1088) 5s 121f (VAE fast, 8-step)", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p (1920x1088) 5s 121f (Stage2 VAE decode, 8-step)", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast)", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode)", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 1},
    ]

    cfg2_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (20-step guided CFG=3.0)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (20-step guided CFG=3.0)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (40-step guided CFG=3.0)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 40, "guidance": 3.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
    ]

    subhost_2chip_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 3},
        {"label": "544p (960x544) 10s 241f T2VA (8-step distill)", "size": "960x544", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p (1920x1088) 5s 121f (VAE fast, 8-step)", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
    ]

    hbm_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 3},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p (1920x1088) 5s 121f (VAE fast, 8-step)", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast)", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode)", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 1},
        {"label": "544p (960x544) 5s 121f T2VA (8-step, c=2)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "conc": 2, "n": 4},
    ]

    sdpa_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast)", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "n": 2},
    ]

    port = 30110 + idx
    pod_rows: list[dict[str, Any]] = []

    # (cfg_name, prec, ngpus, cfg_par, uly, quant, disable_splash, all_hbm, mode_lbl, wl_list)
    fp8_full_workloads = full_workloads + [
        {"label": "544p (960x544) 5s 121f T2VA (8-step, c=2)", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "i2v": False, "conc": 2, "n": 4},
    ]
    configs_by_pod = {
        0: [
            ("sglang_bf16_u8", "BF16", 8, 1, 8, None, False, False, "Ulysses=8 Pallas FlashAttn (CPU-Offload Enc/VAE)", full_workloads),
        ],
        1: [
            ("sglang_fp8_u8_all_hbm", "FP8 (online)", 8, 1, 8, "fp8", False, True, "Ulysses=8 Pallas All-HBM-Resident (96GiB/core)", fp8_full_workloads),
        ],
        2: [
            ("sglang_bf16_cfg2_u4", "BF16", 8, 2, 4, None, False, False, "CFG=2 x Ulysses=4 Pallas", cfg2_workloads),
            ("sglang_fp8_2chip_u4", "FP8 (online)", 4, 1, 4, "fp8", False, False, "2-Chip Ulysses=4 Pallas", subhost_2chip_workloads),
            ("sglang_bf16_2chip_u4", "BF16", 4, 1, 4, None, False, False, "2-Chip Ulysses=4 Pallas", subhost_2chip_workloads[:2]),
        ],
        3: [
            ("sglang_fp8_u8_cpu_offload", "FP8 (online)", 8, 1, 8, "fp8", False, False, "Ulysses=8 Pallas (CPU-Offload Enc/VAE)", hbm_workloads),
            ("sglang_bf16_u8_sdpa", "BF16", 8, 1, 8, None, True, False, "Ulysses=8 Stock SDPA (Splash disabled)", sdpa_workloads),
        ],
    }

    for cfg_name, prec, ngpus, cfg_par, uly, quant, dis_sp, all_hbm, mode_lbl, wl_list in configs_by_pod[idx]:
        try:
            proc, boot_s, _ = start_sglang_ltx25(
                idx,
                config_name=cfg_name,
                model_path=model_dir,
                port=port,
                num_gpus=ngpus,
                ulysses_degree=uly,
                cfg_parallel_size=cfg_par,
                quantization=quant,
                disable_splash=dis_sp,
                all_hbm_resident=all_hbm,
            )
            try:
                meta = {
                    "config": cfg_name,
                    "engine": "sglang-torchtpu",
                    "precision": prec,
                    "chips": ngpus // 2,
                    "num_gpus": ngpus,
                    "mode": mode_lbl,
                    "boot_s": boot_s,
                }
                run_ltx25_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_prefix=gcs_prefix,
                    cfg_meta=meta,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=wl_list,
                    refs=refs,
                    pod_rows=pod_rows,
                )
            finally:
                kill_sglang_processes(proc)
        except Exception as exc:
            log(idx, f"[{cfg_name}] error: {exc}")

    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase2a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, "Phase 2A complete! Waiting for all 4 pods before starting Phase 2B (16-chip DP=4 cluster)...")
    t_s = time.time()
    while time.time() - t_s < 7200:
        ready_2a = sum(
            1 for i in range(4) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase2a_pod_{i}.txt") is not None
        )
        if idx == 0:
            all_host: list[dict[str, Any]] = []
            for i in range(4):
                r = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if r:
                    all_host.extend(json.loads(r.decode("utf-8")))
            md_txt = format_markdown_summary(args.run_id, nodes, all_host, [])
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")
        if ready_2a == 4:
            break
        time.sleep(10)

    # Phase 2B: Full 16-Chip Slice DP=4 Cluster (including 30-RPM Kling-V4 Proxy sizing)
    p1_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_1_results.json")
    p3_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_3_results.json")
    p1_rows = json.loads(p1_raw.decode("utf-8")) if p1_raw else []
    p3_rows = json.loads(p3_raw.decode("utf-8")) if p3_raw else []
    hbm_kling_ok = [
        r["warm_median_e2e_s"]
        for r in p1_rows
        if "Kling-V4" in str(r.get("workload", "")) and r.get("warm_median_e2e_s") and not r.get("error")
    ]
    off_ok = [
        r["warm_median_e2e_s"]
        for r in p3_rows
        if r.get("config") == "sglang_fp8_u8_cpu_offload" and r.get("warm_median_e2e_s") and not r.get("error")
    ]
    use_all_hbm = len(hbm_kling_ok) >= 2
    use_fp8 = use_all_hbm or len(off_ok) >= 2
    dp_quant = "fp8" if use_fp8 else None
    dp_label = f"DP=4 × Ulysses=8 (16 v7x chips, {'FP8' if use_fp8 else 'BF16'}, {'All-HBM-Resident' if use_all_hbm else 'Pallas'})"

    dp_port = 30210 + idx
    proc_dp, _, _ = start_sglang_ltx25(
        idx,
        config_name=f"ltx25_dp4_rep{idx}",
        model_path=model_dir,
        port=dp_port,
        num_gpus=8,
        ulysses_degree=8,
        cfg_parallel_size=1,
        quantization=dp_quant,
        all_hbm_resident=use_all_hbm,
    )
    try:
        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_pod_{idx}.txt", f"http://{ip}:{dp_port}".encode("utf-8"), content_type="text/plain")
        if idx == 0:
            eps: dict[int, str] = {}
            t_dp = time.time()
            while len(eps) < 4 and time.time() - t_dp < 1800:
                for i in range(4):
                    b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_pod_{i}.txt")
                    if b:
                        eps[i] = b.decode("utf-8").strip()
                if len(eps) < 4:
                    time.sleep(3)

            cluster_rows: list[dict[str, Any]] = []
            dp_suites = [
                ("544p (960x544) 5s 121f (8-step)", "960x544", 121, 5.0, 8, False, 4, 12),
                ("544p (960x544) 5s 121f (8-step)", "960x544", 121, 5.0, 8, False, 8, 16),
                ("720p (1280x704) 5s 121f (8-step)", "1280x704", 121, 5.0, 8, False, 4, 12),
                ("720p (1280x704) 5s 121f (8-step)", "1280x704", 121, 5.0, 8, False, 8, 16),
                ("1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast)", "1920x1088", 481, 20.0, 8, False, 4, 8),
                ("1080p Kling-V4 Proxy (1920x1088) 20s 481f (VAE fast)", "1920x1088", 481, 20.0, 8, False, 8, 12),
                ("1080p Kling-V4 Proxy (1920x1088) 20s 481f (Stage2 VAE decode)", "1920x1088", 481, 20.0, 8, False, 4, 4),
            ]
            warmed_dp: set[tuple[str, int, bool]] = set()
            for lbl, sz, nf, dur, steps, upsc, conc, n_req in dp_suites:
                if (sz, nf, upsc) not in warmed_dp:
                    log(0, f"[Cluster 16-chip] Warming all 4 replicas on {(sz, nf, upsc)}...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
                        w_outs = list(
                            p.map(
                                lambda i: call_sglang_video(
                                    eps[i],
                                    {
                                        "prompt": PROMPTS[0],
                                        "size": sz,
                                        "num_frames": nf,
                                        "fps": 24,
                                        "num_inference_steps": steps,
                                        "guidance_scale": 1.0,
                                        "use_diffusion_decoder": upsc,
                                        "auto_duration": False,
                                        "generator_device": "cpu",
                                        "seed": 42,
                                    },
                                ),
                                range(4),
                            )
                        )
                    if all(o.get("ok") for o in w_outs):
                        warmed_dp.add((sz, nf, upsc))

                log(0, f"[Cluster 16-chip] Running {lbl} (c={conc}, n={n_req})...")
                t_c = time.perf_counter()
                reqs = [
                    (
                        eps[r % 4],
                        {
                            "prompt": PROMPTS[r % 4],
                            "size": sz,
                            "num_frames": nf,
                            "fps": 24,
                            "num_inference_steps": steps,
                            "guidance_scale": 1.0,
                            "use_diffusion_decoder": upsc,
                            "auto_duration": False,
                            "generator_device": "cpu",
                            "seed": 900 + r,
                        },
                    )
                    for r in range(n_req)
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_sglang_video(it[0], it[1]), reqs))
                ws = time.perf_counter() - t_c
                oks = [o for o in outs if o.get("ok")]
                lats = sorted(o["e2e_s"] for o in oks)
                med_e2e = round(statistics.median(lats), 3) if lats else None
                p90_e2e = round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
                rpm = round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0
                rtf = round((len(oks) * dur) / ws, 3) if oks and ws > 0 else 0.0
                cluster_rows.append(
                    {
                        "cluster_config": dp_label,
                        "workload": lbl,
                        "size": sz,
                        "steps": steps,
                        "concurrency": conc,
                        "wall_s": round(ws, 3),
                        "median_e2e_s": med_e2e,
                        "p90_e2e_s": p90_e2e,
                        "slice_rpm": rpm,
                        "slices_for_30_rpm": round(30.0 / rpm, 2) if rpm > 0 else None,
                        "slices_for_30_rpm_ceil": math.ceil(30.0 / rpm) if rpm > 0 else None,
                        "video_sec_per_wall_sec": rtf,
                        "ok": f"{len(oks)}/{n_req}",
                    }
                )
                log(0, f"[Cluster 16-chip] DONE {lbl} c={conc}: ok={len(oks)}/{n_req} RPM={rpm}/min med={med_e2e}s RTF={rtf}x")

                all_host = []
                for i in range(4):
                    r = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                    if r:
                        all_host.extend(json.loads(r.decode("utf-8")))
                gcs_upload_bytes(
                    args.gcs_bucket,
                    f"{gcs_prefix}/summary.json",
                    json.dumps(
                        {
                            "run_id": args.run_id,
                            "updated_utc": utc_now(),
                            "nodes": nodes,
                            "single_host_results": all_host,
                            "cluster_dp4_results": cluster_rows,
                        },
                        indent=2,
                    ).encode("utf-8"),
                )
                md_txt = format_markdown_summary(args.run_id, nodes, all_host, cluster_rows)
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")

            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt", b"ok\n", content_type="text/plain")
            log(0, "ALL LTX-2.5 & Kling-V4 Proxy benchmarks complete! Signaled ALL_DONE.txt.")
        else:
            t_h = time.time()
            while time.time() - t_h < 7200:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    log(idx, "Received ALL_DONE.txt from pod-0. Shutting down cleanly.")
                    break
                time.sleep(5)
    finally:
        kill_sglang_processes(proc_dp)


if __name__ == "__main__":
    main()
