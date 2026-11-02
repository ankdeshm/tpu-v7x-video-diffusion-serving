#!/usr/bin/env python3
"""MiniMax-H3 & FastH3 (4-step VSA Distilled) Video+Audio Serving Benchmark Suite on TPU v7x.

Runs across the 4 steady Flex TPU v7x hosts (`tpu-v7x-16-pool`, 4 chips / 8 TensorCores / 768 GB HBM per host =
16 physical v7x chips / 32 TensorCores / 3,072 GB HBM slice-wide):

Phase 1A — 4-Host Simultaneous Single-Host (4x TPU v7x chips = 8 TensorCores) Configuration & Input-Distribution Matrix:
  - Pod 0: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), BF16, Ulysses=8, `tpu_splash_attn` + Pallas Mosaic FlashAttn varlen (`SGLANG_TPU_SPLASH_VARLEN=1`)
  - Pod 1: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), Online FP8 (`--quantization fp8`), Ulysses=8, `tpu_splash_attn` + Pallas Mosaic FlashAttn varlen (`SGLANG_TPU_SPLASH_VARLEN=1`)
  - Pod 2: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition), Stock `tpu_splash_attn` (`SGLANG_TPU_SPLASH_VARLEN=0`, SDPA varlen) + Online AdaLN (`--minimax-h3-adaln-online`), Ulysses=8 (followed by 2-chip `Ulysses=4` ablation)
  - Pod 3: `FastH3` (`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` — 4-step VSA distilled MiniMax-H3), Ulysses=8, BF16 then Online FP8 (`--quantization fp8`)

  Input Distributions Swept (`bs=1` + concurrency `c=1, 2`):
    * Exact Cloud TPU Spec: `i2v 768P 15s` (`fl2va`, 1344x768, 15.0s @ 24fps = 360 frames + 32kHz stereo audio) & `t2va 768P 15s`
      across 4 denoising steps (`num_inference_steps=5`), 12 steps (`13`), 25 steps (`26`), and 49 steps (`50`)
    * Standard 768p 16:9 Durations: `768p 5s` (124 frames), `768p 10s` (244 frames), `768p 15s` (360 frames)
    * Multi-Aspect & Preview Tiers: `768p 9:16 Portrait 5s` (768x1344), `512p 16:9 5s & 15s` (896x512)

Phase 1B — Full 16-Chip `v7x-16` Slice (`4x` Data-Parallel Replicas, `DP=4 x Ulysses=8`) Throughput & 30-RPM Sizing:
  - Load-balances concurrent requests (`c = 4, 8`) across all 4 hosts (`16` v7x chips) on `768p 15s`, `768p 5s`,
    and a realistic mixed video production distribution (`50%` 5s, `30%` 10s, `20%` 15s).
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache/hub"
os.environ["SGLANG_DIFFUSION_CACHE_ROOT"] = "/dev/shm/sglang_diffusion_cache"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import argparse
import concurrent.futures
import glob
import json
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import sys
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

_PORT_SEED_COUNTER = 29610


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
    """Multi-threaded chunked download from regional GCS into /dev/shm."""
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


def apply_sglang_tpu_runtime_patches(idx: int) -> None:
    """Apply verified SGLang v0.2.0 TPU v7x compatibility & FP8 MXU patches."""
    sp = Path("/usr/local/lib/python3.12/site-packages")
    rt = sp / "sglang/multimodal_gen/runtime"

    # 1. minimax_h3_adaln_cache.py: TPU int32 collective probe + CPU safe_open + int32 resolve_slots
    p1 = rt / "models/dits/minimax_h3_adaln_cache.py"
    t1 = p1.read_text()
    old1 = """    probe = torch.tensor(
        [value],
        dtype=torch.int64,
        device=device if device.type == "cuda" else "cpu",
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    new1 = """    from sglang.multimodal_gen.runtime.platforms import current_platform
    _coll_dev = (
        current_platform.get_local_torch_device()
        if current_platform.device_type == "tpu"
        else (device if device.type == "cuda" else "cpu")
    )
    probe = torch.tensor(
        [value],
        dtype=torch.int32 if current_platform.device_type == "tpu" else torch.int64,
        device=_coll_dev,
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    if new1 not in t1:
        assert old1 in t1, "Patch 1a target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1, new1, 1)

    old1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor"""
    new1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device="cpu"))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor.to(device)"""
    if new1b not in t1:
        assert old1b in t1, "Patch 1b target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1b, new1b, 1)

    old1c = "        return torch.tensor(slots, dtype=torch.int64, device=self.block_params.device)"
    new1c = "        return torch.tensor(slots, dtype=torch.int32 if self.block_params.device.type == 'tpu' else torch.int64, device=self.block_params.device)"
    if new1c not in t1:
        assert old1c in t1, "Patch 1c target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1c, new1c, 1)
    p1.write_text(t1)

    # 2. minimax_h3_vdn_attention.py: guard CUDA stream overlap behind q.is_cuda
    p2 = rt / "models/dits/minimax_h3_vdn_attention.py"
    t2 = p2.read_text()
    old2 = """    main_stream = torch.cuda.current_stream(q.device)
    side_stream = _linear_branch_stream(q.device)
    side_stream.wait_stream(main_stream)
    softmax_out = softmax()
    softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
    with torch.cuda.stream(side_stream):
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
    main_stream.wait_stream(side_stream)
    softmax_work.wait()
    linear_work.wait()"""
    new2 = """    if q.is_cuda:
        main_stream = torch.cuda.current_stream(q.device)
        side_stream = _linear_branch_stream(q.device)
        side_stream.wait_stream(main_stream)
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        with torch.cuda.stream(side_stream):
            linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        main_stream.wait_stream(side_stream)
        softmax_work.wait()
        linear_work.wait()
    else:
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        softmax_work.wait()
        linear_work.wait()"""
    if new2 not in t2:
        assert old2 in t2, "Patch 2 target missing in minimax_h3_vdn_attention.py"
        p2.write_text(t2.replace(old2, new2, 1))

    # 3. minimax_h3_qwen3vl.py: always compute get_rope_index on CPU and pass int32 indices to TPU
    p3 = rt / "models/encoders/minimax_h3_qwen3vl.py"
    t3 = p3.read_text()
    old3 = """        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)"""
    new3 = """        position_ids, _ = self.model.get_rope_index(
            host_ids,
            host_image_grid_thw,
            host_video_grid_thw,
            attention_mask=torch.ones_like(host_ids),
        )
        self.model._tpu_host_ids = host_ids
        idx_dtype = torch.int32 if self.device.type == "tpu" else torch.long
        ids = host_ids.to(device=self.device, dtype=idx_dtype)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": position_ids.to(device=self.device, dtype=idx_dtype),
        }"""
    if new3 not in t3:
        assert old3 in t3, "Patch 3 target missing in minimax_h3_qwen3vl.py"
        p3.write_text(t3.replace(old3, new3, 1))

    # 4. qwen3vl.py: static torch.cat splice for image_embeds and _deepstack_process on TPU
    p4 = rt / "models/encoders/qwen3vl.py"
    t4 = p4.read_text()
    old4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    new4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if isinstance(visual_pos_masks, tuple):
            s, e = visual_pos_masks
            return torch.cat(
                [
                    hidden_states[:, :s, :],
                    hidden_states[:, s:e, :] + visual_embeds.unsqueeze(0),
                    hidden_states[:, e:, :],
                ],
                dim=1,
            )
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    if new4a not in t4:
        assert old4a in t4, "Patch 4a target missing in qwen3vl.py"
        t4 = t4.replace(old4a, new4a, 1)

    old4b = """        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(image_mask, image_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = None
        deepstack_visual_embeds = None"""
    new4b = """        image_mask = None
        video_mask = None
        _tpu_visual_slice = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if (
                inputs_embeds.device.type == "tpu"
                and getattr(self, "_tpu_host_ids", None) is not None
                and inputs_embeds.shape[0] == 1
                and pixel_values_videos is None
            ):
                _h_ids = self._tpu_host_ids[0]
                _pos = (_h_ids == self.config.image_token_id).nonzero(as_tuple=False).squeeze(-1)
                _s = int(_pos[0].item())
                _e = int(_pos[-1].item()) + 1
                inputs_embeds = torch.cat(
                    [inputs_embeds[:, :_s, :], image_embeds.unsqueeze(0), inputs_embeds[:, _e:, :]],
                    dim=1,
                )
                inputs_embeds_owned = True
                _tpu_visual_slice = (_s, _e)
            else:
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                if inputs_embeds_owned:
                    inputs_embeds.masked_scatter_(image_mask, image_embeds)
                else:
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = _tpu_visual_slice
        deepstack_visual_embeds = deepstack_image_embeds if _tpu_visual_slice is not None else None"""
    if new4b not in t4:
        assert old4b in t4, "Patch 4b target missing in qwen3vl.py"
        t4 = t4.replace(old4b, new4b, 1)
    p4.write_text(t4)

    # 5. denoising.py: allow tpu platform in minimax_h3 full-loop denoise stage
    p5 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py"
    t5 = p5.read_text()
    old5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
        ):"""
    new5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
            or current_platform.device_type == "tpu"
        ):"""
    if new5 not in t5:
        assert old5 in t5, "Patch 5 target missing in minimax_h3/stages/denoising.py"
        p5.write_text(t5.replace(old5, new5, 1))

    # 6. srt/layers/rotary_embedding/mrope.py: replace step=3 strided slice_scatter with XLA-safe torch.where
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
        assert old6 in t6, "Patch 6 target missing in srt/layers/rotary_embedding/mrope.py"
        p6.write_text(t6.replace(old6, new6, 1))

    # 7. srt/layers/quantization/fp8_utils.py: native TPU v7x FP8 MXU via torch._scaled_mm
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
        assert old7 in t7, "Patch 7 target missing in srt/layers/quantization/fp8_utils.py"
        p7.write_text(t7.replace(old7, new7, 1))

    # 8. sglang_tpu/multimodal_gen/tpu_splash_attn.py: wire SGLANG_TPU_DISABLE_SPLASH & SGLANG_TPU_SPLASH_VARLEN toggles
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
            stxt = stxt.replace(old8a, new8a, 1)
        target8 = "        return self._sdpa.forward_varlen("
        repl8 = """        import os
        if self.use_splash and os.environ.get("SGLANG_TPU_SPLASH_VARLEN", "0") == "1":
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return self._sdpa.forward_varlen("""
        if target8 in stxt and "SGLANG_TPU_SPLASH_VARLEN" not in stxt:
            stxt = stxt.replace(target8, repl8, 1)
        p8.write_text(stxt)

    # 9. minimax_h3/stages/decoding.py: avoid rank-0-only TPU launch barrier hang during _decode_audio
    #    by moving AudioVAE (~578 MB) and draining audio_latent to CPU symmetrically across all ranks,
    #    decoding BigVGAN on CPU on rank 0, and broadcasting the ~3.8 MB CPU waveform dict over Gloo.
    p9 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
    t9 = p9.read_text()
    old9 = """        # Audio VAE weights are replicated. Decode on replica rank 0 and broadcast
        # only within the request's replica, excluding independent DP replicas.
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        if is_audio_owner:
            try:
                audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    new9 = """        # Audio VAE weights are replicated. On TPU, avoid rank-0-only TPU execution
        # (which hangs at the multi-core PJRT launch barrier while ranks 1..7 block in Gloo):
        # ensure audio_vae & audio_latent are on CPU symmetrically across all ranks, decode BigVGAN
        # on CPU on rank 0, and broadcast the compact CPU waveform dict over Gloo broadcast_object.
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        _is_tpu_decode = visual_latent.device.type == "tpu"
        if _is_tpu_decode and self.audio_vae is not None:
            _avae_param = next(self.audio_vae.parameters(), None)
            if _avae_param is not None and _avae_param.device.type == "tpu":
                self.audio_vae = self.audio_vae.to(device="cpu", dtype=torch.float32)
        _audio_latent_cpu = (
            audio_latent.detach().to(device="cpu", dtype=torch.float32)
            if _is_tpu_decode
            else None
        )
        if is_audio_owner:
            try:
                if _is_tpu_decode:
                    assert self.audio_vae is not None
                    _avae = self.audio_vae.to(device="cpu", dtype=torch.float32)
                    if _avae.training:
                        _avae.eval()
                    _a_cfg = server_args.pipeline_config.audio_vae_config.arch_config
                    _a_lat = _reverse_normalize_latents(
                        _audio_latent_cpu,
                        mean_values=_a_cfg.latents_mean,
                        std_values=_a_cfg.latents_std,
                        name="audio_vae",
                    ).to(device="cpu", dtype=torch.float32)
                    _wf = _required_tensor(_avae.decode(_a_lat), "audio_vae.decode")
                    audio_payload = {
                        "waveform": _wf.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                        "sample_rate": int(_avae.sample_rate),
                    }
                else:
                    audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            if _is_tpu_decode:
                audio_payload = replica_group.broadcast_object(audio_payload, src=0)
            else:
                audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    if "_is_tpu_decode = visual_latent.device.type == \"tpu\"" not in t9:
        assert old9 in t9, "Patch 9 target missing in minimax_h3/stages/decoding.py"
        p9.write_text(t9.replace(old9, new9, 1))

    log(idx, "All SGLang TPU v7x runtime patches (1a/1b/1c, VDN, Qwen3-VL, MRoPE, Denoise, FP8 MXU, Splash toggles, CPU AudioDecode) verified & applied!")


def ensure_sglang_tpu_installed(idx: int) -> None:
    check = subprocess.run(
        [
            "python3",
            "-c",
            "import shutil, sglang, sglang_tpu, av, cv2, diffusers; assert shutil.which('ffmpeg') and shutil.which('ffprobe')",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(f"Container pre-installed environment check failed:\n{check.stderr}")
    apply_sglang_tpu_runtime_patches(idx)


def setup_single_host_metadata_hook(idx: int, bucket: str) -> None:
    log(idx, "Setting up 2x2x1 single-host metadata isolation (4 v7x chips / 8 TensorCores per host)...")
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
            log(idx, "mock_metadata.py already listening on 127.0.0.1:80")
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
    log(idx, "WARNING: mock_metadata.py port 80 check timed out; check /workspace/mock_metadata.log")


def generate_synthetic_keyframes(input_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "16:9_768": (1344, 768, "keyframe_1344x768.png"),
        "9:16_768": (768, 1344, "keyframe_768x1344.png"),
        "16:9_512": (896, 512, "keyframe_896x512.png"),
    }
    paths: dict[str, str] = {}
    for k, (w, h, fn) in specs.items():
        p = input_dir / fn
        img = Image.new("RGB", (w, h), color=(24, 42, 68))
        draw = ImageDraw.Draw(img)
        # Gradient-like horizon bands + sunlit ocean/cliff silhouette for deterministic i2v conditioning
        for y in range(0, h, 8):
            r = min(255, int(30 + 160 * (y / max(h, 1))))
            g = min(255, int(55 + 110 * (y / max(h, 1))))
            b = min(255, int(95 + 80 * (1.0 - y / max(h, 1))))
            draw.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        draw.ellipse([w // 3, h // 5, w // 3 + 140, h // 5 + 140], fill=(252, 212, 128))
        img.save(p, format="PNG")
        paths[k] = str(p)
    return paths


def ensure_fl2va_symlinks(fl2va_dir: Path) -> None:
    if not (fl2va_dir / "vae").exists() and (fl2va_dir / "video_vae").exists():
        (fl2va_dir / "vae").symlink_to("video_vae")
    v_sf = fl2va_dir / "video_vae" / "model.safetensors"
    v_src = fl2va_dir / "video_vae" / "source" / "model.safetensors"
    if not v_sf.exists() and v_src.exists():
        v_sf.symlink_to("source/model.safetensors")


def stage_minimax_base_from_gcs(idx: int, bucket: str) -> str:
    """Ensure ONLY MiniMax-H3 FL2VA (~134 GiB) is staged in /dev/shm (remove any duplicate root copies)."""
    shutil.rmtree("/dev/shm/FastH3", ignore_errors=True)
    shutil.rmtree("/dev/shm/sglang_diffusion_cache", ignore_errors=True)
    shutil.rmtree("/dev/shm/hf_cache", ignore_errors=True)

    base_dir = Path("/dev/shm/MiniMax-H3")
    for dup in ("text_encoder", "transformer", "vae", "video_vae", "audio_vae", "Ref2VA", "R2VA", "docs"):
        shutil.rmtree(base_dir / dup, ignore_errors=True)
    for f in base_dir.glob("*.json"):
        try:
            f.unlink()
        except OSError:
            pass

    fl2va_dir = base_dir / "FL2VA"
    fast_gcs_sync_dir(
        idx,
        bucket,
        "models/MiniMaxAI/MiniMax-H3/FL2VA/",
        fl2va_dir,
    )
    ensure_fl2va_symlinks(fl2va_dir)
    assert (fl2va_dir / "model_index.json").exists(), f"Missing {fl2va_dir / 'model_index.json'}"
    log(idx, f"MiniMax-H3 FL2VA staged in {fl2va_dir} (RAM: {host_ram_summary()})")
    return str(fl2va_dir)


def stage_fasth3_from_gcs(idx: int, bucket: str) -> str:
    """Stage materialized FastH3 directly into /dev/shm/FastH3 (~138 GiB) matching overlay materialize.py."""
    shutil.rmtree("/dev/shm/MiniMax-H3", ignore_errors=True)
    shutil.rmtree("/dev/shm/sglang_diffusion_cache", ignore_errors=True)
    shutil.rmtree("/dev/shm/hf_cache", ignore_errors=True)

    fh_dir = Path("/dev/shm/FastH3")
    fast_gcs_sync_dir(
        idx,
        bucket,
        "models/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/",
        fh_dir,
        exclude_prefixes=("vae/",),
    )
    fast_gcs_sync_dir(
        idx,
        bucket,
        "models/MiniMaxAI/MiniMax-H3/FL2VA/video_vae/",
        fh_dir / "video_vae",
    )
    gcs_download_file(
        bucket,
        "models/kevin-mi/FastH3-4step-Preview-overlay/model_index.json",
        fh_dir / "model_index.json",
    )
    ensure_fl2va_symlinks(fh_dir)

    # Patch _class_name in video_vae/config.json & audio_vae/config.json + generate sglang_rope_inv_freq.safetensors
    rope_py = r"""
import json, os, torch
from safetensors.torch import save_file

for comp, cls_name in [("video_vae", "MiniMaxH3VideoVAE"), ("audio_vae", "MiniMaxH3AudioVAE")]:
    cp = os.path.join("/dev/shm/FastH3", comp, "config.json")
    cfg_c = json.load(open(cp))
    cfg_c["_class_name"] = cls_name
    with open(cp, "w") as f:
        json.dump(cfg_c, f, indent=2, sort_keys=True)

t_dir = "/dev/shm/FastH3/transformer"
cfg = json.load(open(os.path.join(t_dir, "config.json")))
d = int(cfg["rope_freq_dim"])
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 2 * d, 2, dtype=torch.float32) / (2 * d)))
save_file({"rope.inv_freq": inv_freq}, os.path.join(t_dir, "sglang_rope_inv_freq.safetensors"))
idx_p = os.path.join(t_dir, "diffusion_pytorch_model.safetensors.index.json")
idx_obj = json.load(open(idx_p))
idx_obj["weight_map"]["rope.inv_freq"] = "sglang_rope_inv_freq.safetensors"
with open(idx_p, "w") as f:
    json.dump(idx_obj, f, indent=2, sort_keys=True)
"""
    subprocess.run(["python3", "-c", rope_py], check=True)
    assert (fh_dir / "model_index.json").exists(), "Missing /dev/shm/FastH3/model_index.json"
    log(idx, f"FastH3 materialized directly in {fh_dir} (RAM: {host_ram_summary()})")
    return str(fh_dir)


def build_tpu_server_env(
    *,
    num_gpus: int,
    splash_varlen: bool,
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
        # 2 physical v7x chips = 4 TensorCores (1x2x1 sub-host slice)
        chips_bounds = "1,2,1"
        topo = "1x2x1"
        torch_tpu_topo = "1,2,1,2"
        local_ranks = "0,1,2,3"
        accel = "tpu7x-4"
        vis_devs = "0,1"
        metrics_ports = "8431,8432,8433,8434"
    else:
        # Full 4 physical v7x chips = 8 TensorCores (2x2x1 single host)
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
            "GOOGLE_CLOUD_PROJECT": "mock-tpu-project",
            "GCLOUD_PROJECT": "mock-tpu-project",
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
            "SGLANG_TPU_SPLASH_VARLEN": "1" if splash_varlen else "0",
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
            "rm -f /tmp/libtpu_lockfile /dev/shm/torch_tpu_pg_store_*; rm -rf /dev/shm/sglang_out_*; sleep 2",
        ],
        check=False,
    )


def start_sglang_server(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    port: int,
    num_gpus: int,
    ulysses_degree: int,
    quantization: str | None,
    splash_varlen: bool = False,
    disable_splash: bool = False,
    adaln_online: bool = False,
    dit_cpu_offload: bool = False,
    all_hbm_resident: bool = False,
    extra_args: list[str] | None = None,
    ready_timeout_s: int = 1500,
) -> tuple[subprocess.Popen, float, Path]:
    global _PORT_SEED_COUNTER
    kill_sglang_processes()
    port_seed = _PORT_SEED_COUNTER
    _PORT_SEED_COUNTER += 30

    out_dir = Path(f"/dev/shm/sglang_out_{config_name}")
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
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        "tpu_splash_attn",
        "--warmup-mode",
        "off",
        "--dit-cpu-offload",
        "true" if dit_cpu_offload else "false",
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
    if adaln_online:
        cmd.extend(["--minimax-h3-adaln-online", "--minimax-h3-adaln-host-cache-gb", "1"])
    if extra_args:
        cmd.extend(extra_args)

    env = build_tpu_server_env(
        num_gpus=num_gpus,
        splash_varlen=splash_varlen,
        disable_splash=disable_splash,
        port_seed=port_seed,
    )
    log(idx, f"Starting [{config_name}] (seed={port_seed}, RAM={host_ram_summary()}): {' '.join(cmd)}")
    t0 = time.time()
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    urls = (f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/videos")
    while time.time() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"sglang [{config_name}] exited early rc={proc.returncode}:\n{tail}")
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=5) as resp:
                    if resp.status == 200:
                        boot_s = round(time.time() - t0, 2)
                        log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s! (RAM={host_ram_summary()})")
                        return proc, boot_s, log_path
            except Exception:
                pass
        time.sleep(3)

    kill_sglang_processes(proc)
    tail = log_path.read_text(errors="replace")[-4000:]
    raise TimeoutError(f"Server [{config_name}] timed out after {ready_timeout_s}s:\n{tail}")


def submit_and_poll_video(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 2400.0,
    probe_streams: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": f"http_{e.code}",
            "error": err_body[:800],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "submit_error",
            "error": str(exc)[:800],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }

    vid_id = job["id"]
    poll_url = f"{endpoint}/v1/videos/{vid_id}"
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(poll_url, timeout=30) as resp:
                st = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        status = st.get("status")
        if status == "completed":
            e2e_s = time.perf_counter() - t0
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
            # Clean up large mp4 after measuring size so /dev/shm stays clean
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return {
                "ok": True,
                "id": vid_id,
                "status": "completed",
                "e2e_s": round(e2e_s, 3),
                "inference_time_s": st.get("inference_time_s"),
                "peak_memory_mb": st.get("peak_memory_mb"),
                "size": st.get("size"),
                "seconds": st.get("seconds"),
                "file_bytes": fsize,
                "ffprobe_streams": ffprobe_streams,
            }
        if status == "failed":
            return {
                "ok": False,
                "id": vid_id,
                "status": "failed",
                "error": str(st.get("error"))[:800],
                "e2e_s": round(time.perf_counter() - t0, 3),
            }

    return {
        "ok": False,
        "id": vid_id,
        "status": "timeout",
        "error": f"Timed out after {timeout_s}s",
        "e2e_s": round(time.perf_counter() - t0, 3),
    }


def make_minimax_payload(
    *,
    prompt: str,
    task: str,
    short_edge: int,
    aspect_ratio: str,
    duration_seconds: float,
    num_inference_steps: int,
    seed: int,
    keyframe_uri: str | None = None,
) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    if task == "fl2va":
        assert keyframe_uri is not None, "fl2va requires keyframe_uri"
        conditions = [
            {
                "type": "image",
                "role": "keyframe",
                "uri": keyframe_uri,
                "frame_index": 0,
            }
        ]
    return {
        "prompt": prompt,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": short_edge,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration_seconds),
        },
        "num_inference_steps": int(num_inference_steps),
        "quality": "lossless",
        "generator_device": "cpu",
        "seed": seed,
    }


def run_host_sweep(
    idx: int,
    *,
    bucket: str,
    gcs_out_prefix: str,
    config_meta: dict[str, Any],
    endpoint: str,
    workloads: list[dict[str, Any]],
    keyframes: dict[str, str],
    existing_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    warmed_shapes: set[tuple[str, int, str, float]] = set()
    probed_once = False
    for w_i, wl in enumerate(workloads):
        task = wl["task"]
        short_edge = wl["short_edge"]
        aspect = wl["aspect_ratio"]
        dur = float(wl["duration_seconds"])
        steps_grid = wl["num_inference_steps"]
        denoise_steps = steps_grid - 1
        conc = wl.get("concurrency", 1)
        n_reqs = wl.get("num_requests", max(2, conc))
        kf_key = f"{aspect}_{short_edge}"
        kf_uri = keyframes.get(kf_key, keyframes["16:9_768"]) if task == "fl2va" else None

        shape_key = (task, short_edge, aspect, dur)
        cold_compile_s = None
        sample_streams = None
        if shape_key not in warmed_shapes:
            # Compile/warmup pass (use 5 sigma points = 4 steps so compile is fast!)
            warm_steps = 5 if config_meta["is_fasth3"] else min(steps_grid, 5)
            log(
                idx,
                f"[{config_meta['name']}] Warming XLA shape {shape_key} (grid={warm_steps}, RAM={host_ram_summary()})...",
            )
            w_payload = make_minimax_payload(
                prompt=PROMPTS[0],
                task=task,
                short_edge=short_edge,
                aspect_ratio=aspect,
                duration_seconds=dur,
                num_inference_steps=warm_steps,
                seed=100 + w_i,
                keyframe_uri=kf_uri,
            )
            w_res = submit_and_poll_video(endpoint, w_payload, probe_streams=(not probed_once))
            cold_compile_s = w_res["e2e_s"]
            if w_res.get("ffprobe_streams"):
                sample_streams = w_res["ffprobe_streams"]
                probed_once = True
            log(
                idx,
                f"[{config_meta['name']}] Warmup {shape_key}: ok={w_res['ok']} e2e={cold_compile_s}s RAM=({host_ram_summary()}) err={w_res.get('error', '')[:160]}",
            )
            if w_res["ok"]:
                warmed_shapes.add(shape_key)
            else:
                row = {
                    **config_meta,
                    "workload_label": wl["label"],
                    "task": task,
                    "short_edge": short_edge,
                    "aspect_ratio": aspect,
                    "duration_seconds": dur,
                    "denoise_steps": denoise_steps,
                    "num_inference_steps": steps_grid,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": 0,
                    "cold_compile_e2e_s": cold_compile_s,
                    "error": w_res.get("error"),
                }
                existing_results.append(row)
                gcs_upload_bytes(
                    bucket,
                    f"{gcs_out_prefix}/pod_{idx}_results.json",
                    json.dumps(existing_results, indent=2).encode("utf-8"),
                )
                continue

        log(
            idx,
            f"[{config_meta['name']}] Running {wl['label']} ({task} {short_edge}p {aspect} {dur}s, {denoise_steps} steps, c={conc}, n={n_reqs})...",
        )
        payloads = [
            make_minimax_payload(
                prompt=PROMPTS[(w_i + r_i) % len(PROMPTS)],
                task=task,
                short_edge=short_edge,
                aspect_ratio=aspect,
                duration_seconds=dur,
                num_inference_steps=steps_grid,
                seed=1000 + w_i * 100 + r_i,
                keyframe_uri=kf_uri,
            )
            for r_i in range(n_reqs)
        ]

        t_wall0 = time.perf_counter()
        if conc == 1:
            outs = [submit_and_poll_video(endpoint, p) for p in payloads]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(lambda p: submit_and_poll_video(endpoint, p), payloads))
        wall_s = time.perf_counter() - t_wall0

        oks = [o for o in outs if o.get("ok")]
        lats = [o["e2e_s"] for o in oks]
        inf_ts = [o["inference_time_s"] for o in oks if o.get("inference_time_s") is not None]

        mean_e2e = round(statistics.mean(lats), 3) if lats else None
        med_e2e = round(statistics.median(lats), 3) if lats else None
        mean_inf = round(statistics.mean(inf_ts), 3) if inf_ts else None
        step_ms = round(((mean_inf or mean_e2e) * 1000.0) / max(denoise_steps, 1), 1) if (mean_inf or mean_e2e) else None
        rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
        vid_sec_per_s = round((len(oks) * dur) / wall_s, 3) if oks and wall_s > 0 else 0.0

        row = {
            **config_meta,
            "workload_label": wl["label"],
            "task": task,
            "resolved_size": oks[0].get("size") if oks else None,
            "short_edge": short_edge,
            "aspect_ratio": aspect,
            "duration_seconds": dur,
            "denoise_steps": denoise_steps,
            "num_inference_steps": steps_grid,
            "concurrency": conc,
            "num_requests": n_reqs,
            "ok_count": len(oks),
            "cold_compile_e2e_s": cold_compile_s,
            "wall_s": round(wall_s, 3),
            "mean_e2e_s": mean_e2e,
            "median_e2e_s": med_e2e,
            "mean_inference_s": mean_inf,
            "ms_per_denoise_step": step_ms,
            "rpm_videos_per_min": rpm,
            "video_sec_per_wall_sec": vid_sec_per_s,
            "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
            "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
            "ffprobe_streams": sample_streams,
            "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
        }
        log(
            idx,
            f"[{config_meta['name']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} mean_e2e={mean_e2e}s step={step_ms}ms RPM={rpm}/min RTF={vid_sec_per_s}x",
        )
        existing_results.append(row)
        gcs_upload_bytes(
            bucket,
            f"{gcs_out_prefix}/pod_{idx}_results.json",
            json.dumps(existing_results, indent=2).encode("utf-8"),
        )
    return existing_results


def format_markdown_summary(
    run_id: str,
    nodes: dict[int, dict[str, Any]],
    all_host_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# MiniMax-H3 & FastH3 Video+Audio Serving on TPU v7x (`{run_id}`)",
        "",
        f"- **Updated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `tpu-v7x-cluster` / `tpu-v7x-16-pool` (Steady Flex `v7x-16`, `4` hosts × `4` physical `v7x` chips = `16` physical chips / `32` TensorCores / `3,072 GiB` HBM3e)",
        "- **Serving Engine:** `sglang-torchtpu` (`TpuOOTPlatform`, `tpu_splash_attn` Pallas Mosaic FlashAttention + Ulysses Sequence Parallelism, 24 fps H.264 + 32 kHz stereo AAC muxed `.mp4`)",
        f"- **Registered Hosts:** `{json.dumps({k: v['hostname'] for k, v in sorted(nodes.items())})}`",
        "",
        "## 1. Single-Host (`4x TPU v7x Chips` = `8` TensorCores / `768 GiB` HBM) Configuration & Input-Distribution Matrix",
        "",
        "| Config | Model | Prec. | Chips (Cores) | Attn Mode | Task & Input Spec | Canvas | Steps | Conc. | Cold Compile (s) | Warm Med E2E (s) | ms/step | Host RPM (vids/min) | Video-s / Wall-s | OK |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_host_rows:
        chips_str = f"{r['chips']} ({r['num_gpus']})"
        lines.append(
            f"| `{r['name']}` | `{r['model_short']}` | `{r['precision']}` | {chips_str} | `{r['attn_mode']}` | "
            f"**{r['workload_label']}** (`{r['task']}`) | `{r.get('resolved_size') or '-'}` | {r['denoise_steps']} | {r['concurrency']} | "
            f"{r.get('cold_compile_e2e_s') or '-'} | **{r.get('median_e2e_s') or '-'}** | {r.get('ms_per_denoise_step') or '-'} | "
            f"**{r.get('rpm_videos_per_min') or 0:.2f}** | {r.get('video_sec_per_wall_sec') or 0:.2f}x | `{r['ok_count']}/{r['num_requests']}` |"
        )

    if cluster_rows:
        lines.extend(
            [
                "",
                "## 2. Full `v7x-16` Slice (`16x TPU v7x Chips` = `4×` Data-Parallel Replicas, `DP=4 × Ulysses=8`) Throughput & `30 RPM` Sizing",
                "",
                "| Cluster Config | Workload / Input Distribution | Steps | Concurrency | Wall (s) | Med E2E (s) | P90 E2E (s) | Slice RPM (vids/min) | Video-s / Wall-s | Slices for 30 RPM | OK |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cluster_rows:
            rpm = c.get("rpm_videos_per_min") or 0.0
            slices_30 = f"{30.0 / rpm:.2f}" if rpm > 0 else "-"
            lines.append(
                f"| `{c['cluster_config']}` | **{c['workload_label']}** | {c['denoise_steps']} | {c['concurrency']} | "
                f"{c['wall_s']} | **{c.get('median_e2e_s') or '-'}** | {c.get('p90_e2e_s') or '-'} | "
                f"**{rpm:.2f}** | {c.get('video_sec_per_wall_sec') or 0:.2f}x | `{slices_30}` | `{c['ok_count']}/{c['num_requests']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
    ap.add_argument("--run-id", default="v7x_flex_minimax_h3_serve_p1")
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    gcs_prefix = f"video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    subprocess.run(["bash", "-lc", "mount -o remount,size=850G /dev/shm || true"], check=False)
    kill_sglang_processes()

    # 1. Register node IP in GCS & wait for all 4 pods
    node_info = {"idx": idx, "ip": ip, "hostname": hostname, "ts": utc_now()}
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{idx}.json", json.dumps(node_info).encode("utf-8"))
    log(idx, f"Registered pod-{idx} ({ip} / {hostname}). Waiting for all 4 Flex v7x hosts...")

    nodes: dict[int, dict[str, Any]] = {}
    t_wait = time.time()
    while len(nodes) < 4 and time.time() - t_wait < 600:
        for i in range(4):
            if i not in nodes:
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
                if raw:
                    nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < 4:
            time.sleep(3)
    assert len(nodes) == 4, f"Expected 4 nodes, found {nodes}"
    log(idx, f"All 4 Flex v7x hosts registered: { {i: nodes[i]['ip'] for i in range(4)} }")

    # 2. Verify sglang-torchtpu + apply TPU v7x patches, compile 2x2x1 metadata hook, create keyframes
    ensure_sglang_tpu_installed(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    keyframes = generate_synthetic_keyframes(Path("/dev/shm/inputs"))

    # Strict 1-model-at-a-time /dev/shm staging for Phase 1A
    base_dir = ""
    fasth3_dir = ""
    if idx == 3:
        fasth3_dir = stage_fasth3_from_gcs(idx, args.gcs_bucket)
    else:
        base_dir = stage_minimax_base_from_gcs(idx, args.gcs_bucket)

    # Define input-distribution workloads
    base_workloads = [
        # Exact Cloud TPU target: i2v 768P 15s (4-step, 12-step, 25-step, 49-step) + t2va 768P 15s
        {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s t2va (4-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (12-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 13, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (49-step full)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 50, "concurrency": 1, "num_requests": 1},
        # Standard 5s & 10s 768p 16:9 + Portrait 9:16 + 512p preview + c=2 concurrency
        {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 5s t2va (49-step full)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 50, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 10s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 10s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 9:16 Portrait 5s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "512p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 512, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (4-step, c=2)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 2, "num_requests": 4},
    ]

    fasth3_workloads = [
        {"label": "768p 16:9 15s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 10s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 5s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 9:16 Portrait 15s FastH3 (4-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "512p 16:9 15s FastH3 (4-step)", "task": "t2va", "short_edge": 512, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 15s FastH3 (4-step, c=2)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 2, "num_requests": 4},
    ]

    pod_results: list[dict[str, Any]] = []
    port = 30010 + idx

    # 3. Phase 1A: 4-Way Simultaneous Single-Host Matrix
    if idx == 0:
        try:
            proc, boot_s, _ = start_sglang_server(
                idx,
                config_name="mh3_bf16_u8_pallas",
                model_path=base_dir,
                port=port,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                splash_varlen=False,
                disable_splash=False,
                adaln_online=True,
            )
            try:
                meta = {
                    "name": "mh3_bf16_u8_pallas",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 4,
                    "num_gpus": 8,
                    "ulysses_degree": 8,
                    "attn_mode": "Pallas FlashAttn + AdaLN-online",
                    "boot_s": boot_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=base_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc)
        except Exception as exc:
            log(idx, f"[mh3_bf16_u8_pallas] error: {exc}")

    elif idx == 1:
        try:
            proc, boot_s, _ = start_sglang_server(
                idx,
                config_name="mh3_fp8_u8_pallas",
                model_path=base_dir,
                port=port,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                splash_varlen=False,
                disable_splash=False,
                adaln_online=False,
            )
            try:
                meta = {
                    "name": "mh3_fp8_u8_pallas",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "FP8 (online)",
                    "chips": 4,
                    "num_gpus": 8,
                    "ulysses_degree": 8,
                    "attn_mode": "Pallas FlashAttn (Resident DiT, Encoder/VAE Offload)",
                    "boot_s": boot_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=base_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc)
        except Exception as exc:
            log(idx, f"[mh3_fp8_u8_pallas] error: {exc}")

    elif idx == 2:
        ablation_workloads = [
            {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        ]
        # Sub-pass 2A: Stock SDPA (SGLANG_TPU_DISABLE_SPLASH=1) + AdaLN online on 4 chips (Ulysses=8) — apples-to-apples vs Pod 0
        try:
            proc, boot_s, _ = start_sglang_server(
                idx,
                config_name="mh3_bf16_u8_sdpa_adaln",
                model_path=base_dir,
                port=port,
                num_gpus=8,
                ulysses_degree=8,
                quantization=None,
                splash_varlen=False,
                disable_splash=True,
                adaln_online=True,
            )
            try:
                meta = {
                    "name": "mh3_bf16_u8_sdpa_adaln",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 4,
                    "num_gpus": 8,
                    "ulysses_degree": 8,
                    "attn_mode": "Stock SDPA (Splash disabled) + AdaLN-online",
                    "boot_s": boot_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=ablation_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc)
        except Exception as exc:
            log(idx, f"SDPA ablation note: {exc}")

        # Sub-pass 2B: FP8 + Pallas FlashAttn + AdaLN-online — apples-to-apples Resident DiT vs AdaLN-online vs Pod 1
        try:
            adaln_cmp_workloads = ablation_workloads[:2]
            proc_ad, boot_ad_s, _ = start_sglang_server(
                idx,
                config_name="mh3_fp8_u8_pallas_adaln",
                model_path=base_dir,
                port=port,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                splash_varlen=False,
                disable_splash=False,
                adaln_online=True,
            )
            try:
                meta_ad = {
                    "name": "mh3_fp8_u8_pallas_adaln",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "FP8 (online)",
                    "chips": 4,
                    "num_gpus": 8,
                    "ulysses_degree": 8,
                    "attn_mode": "Pallas FlashAttn + AdaLN-online",
                    "boot_s": boot_ad_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta_ad,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=adaln_cmp_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc_ad)
        except Exception as exc:
            log(idx, f"FP8 AdaLN comparison note: {exc}")

        # Sub-pass 2C: 2-chip (`num_gpus=4, ulysses_degree=4`, FP8 + Pallas FlashAttn) scaling test!
        try:
            proc2, boot2_s, _ = start_sglang_server(
                idx,
                config_name="mh3_fp8_u4_2chips",
                model_path=base_dir,
                port=port,
                num_gpus=4,
                ulysses_degree=4,
                quantization="fp8",
                splash_varlen=False,
                disable_splash=False,
                adaln_online=False,
            )
            try:
                meta2 = {
                    "name": "mh3_fp8_u4_2chips",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "FP8 (online)",
                    "chips": 2,
                    "num_gpus": 4,
                    "ulysses_degree": 4,
                    "attn_mode": "Pallas FlashAttn (Resident DiT)",
                    "boot_s": boot2_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta2,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=ablation_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc2)
        except Exception as exc:
            log(idx, f"2-chip ablation note: {exc}")

        # Sub-pass 2D: Ironwood 96 GiB/core All-HBM-Resident (FP8 DiT + Qwen3-VL + VideoVAE all resident in HBM3e, zero PCIe weight swaps!)
        try:
            hbm_workloads = [
                {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
                {"label": "768p 16:9 15s t2va (4-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
                {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
                {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
                {"label": "768p 16:9 15s i2v (4-step, c=2)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 2, "num_requests": 4},
            ]
            proc_hbm, boot_hbm_s, _ = start_sglang_server(
                idx,
                config_name="mh3_fp8_u8_all_hbm_resident",
                model_path=base_dir,
                port=port,
                num_gpus=8,
                ulysses_degree=8,
                quantization="fp8",
                splash_varlen=False,
                disable_splash=False,
                adaln_online=False,
                all_hbm_resident=True,
            )
            try:
                meta_hbm = {
                    "name": "mh3_fp8_u8_all_hbm_resident",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "FP8 (online)",
                    "chips": 4,
                    "num_gpus": 8,
                    "ulysses_degree": 8,
                    "attn_mode": "Pallas FlashAttn (All-HBM-Resident 96GiB/core)",
                    "boot_s": boot_hbm_s,
                }
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta=meta_hbm,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=hbm_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            finally:
                kill_sglang_processes(proc_hbm)
        except Exception as exc:
            log(idx, f"All-HBM-resident ablation note: {exc}")

    elif idx == 3:
        # Sub-pass 3A: FastH3 (4-step VSA distilled):
        #   1) BF16 + AdaLN-online (NEVER dit_cpu_offload=True, which duplicates 8x61.7=493.6GiB in host RAM!)
        #   2) Online FP8 (Resident DiT, Encoder/VAE CPU offload)
        #   3) Online FP8 (All-HBM-Resident 96GiB/core: FP8 DiT + Qwen3-VL + VideoVAE all on HBM3e!)
        fasth3_configs = [
            (None, "fasth3_bf16_u8_pallas", "BF16", True, False, "Pallas FlashAttn + AdaLN-online", fasth3_workloads),
            ("fp8", "fasth3_fp8_u8_pallas", "FP8 (online)", False, False, "Pallas FlashAttn (Resident DiT, Encoder/VAE Offload)", fasth3_workloads),
            ("fp8", "fasth3_fp8_u8_all_hbm_resident", "FP8 (online)", False, True, "Pallas FlashAttn (All-HBM-Resident 96GiB/core)", fasth3_workloads[:3] + [fasth3_workloads[-1]]),
        ]
        for q_mode, cfg_tag, prec_lbl, adaln_on, all_hbm, attn_lbl, wl_list in fasth3_configs:
            try:
                proc, boot_s, _ = start_sglang_server(
                    idx,
                    config_name=cfg_tag,
                    model_path=fasth3_dir,
                    port=port,
                    num_gpus=8,
                    ulysses_degree=8,
                    quantization=q_mode,
                    splash_varlen=False,
                    disable_splash=False,
                    adaln_online=adaln_on,
                    dit_cpu_offload=False,
                    all_hbm_resident=all_hbm,
                )
                try:
                    meta = {
                        "name": cfg_tag,
                        "model_short": "FastH3 (4-step VSA)",
                        "is_fasth3": True,
                        "precision": prec_lbl,
                        "chips": 4,
                        "num_gpus": 8,
                        "ulysses_degree": 8,
                        "attn_mode": attn_lbl,
                        "boot_s": boot_s,
                    }
                    run_host_sweep(
                        idx,
                        bucket=args.gcs_bucket,
                        gcs_out_prefix=gcs_prefix,
                        config_meta=meta,
                        endpoint=f"http://127.0.0.1:{port}",
                        workloads=wl_list,
                        keyframes=keyframes,
                        existing_results=pod_results,
                    )
                finally:
                    kill_sglang_processes(proc)
            except Exception as exc:
                log(idx, f"[{cfg_tag}] error: {exc}")

        # Swap /dev/shm on Pod 3 from FastH3 -> MiniMax-H3 before Phase 1B 16-chip DP=4 sweep!
        base_dir = stage_minimax_base_from_gcs(idx, args.gcs_bucket)

    # Mark Phase 1A complete for this pod
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase1a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, "Phase 1A complete! Waiting for all 4 pods before starting Phase 1B (16-chip DP=4 cluster)...")

    t_sync = time.time()
    while time.time() - t_sync < 7200:
        ready_1a = sum(
            1
            for i in range(4)
            if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase1a_pod_{i}.txt") is not None
        )
        if idx == 0:
            # Flush interim SUMMARY.md while waiting
            all_rows: list[dict[str, Any]] = []
            for i in range(4):
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if raw:
                    all_rows.extend(json.loads(raw.decode("utf-8")))
            md_txt = format_markdown_summary(args.run_id, nodes, all_rows, [])
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")
        if ready_1a == 4:
            break
        time.sleep(10)

    # 4. Phase 1B: Full 16-Chip `v7x-16` Slice (4x DP Replicas = 16 chips / 32 TensorCores)
    # Pick best single-host config (FP8 + All-HBM-Resident if Pod 2 Sub-pass 2D succeeded faster, else FP8 Offload, else BF16)
    p0_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_0_results.json")
    p1_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_1_results.json")
    p2_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_2_results.json")
    p0_rows = json.loads(p0_raw.decode("utf-8")) if p0_raw else []
    p1_rows = json.loads(p1_raw.decode("utf-8")) if p1_raw else []
    p2_rows = json.loads(p2_raw.decode("utf-8")) if p2_raw else []
    fp8_15s = [r["median_e2e_s"] for r in p1_rows if r.get("ok_count", 0) > 0 and r.get("duration_seconds") == 15.0 and r.get("denoise_steps") == 4 and r.get("concurrency") == 1 and r.get("median_e2e_s")]
    bf16_15s = [r["median_e2e_s"] for r in p0_rows if r.get("ok_count", 0) > 0 and r.get("duration_seconds") == 15.0 and r.get("denoise_steps") == 4 and r.get("concurrency") == 1 and r.get("median_e2e_s")]
    hbm_15s = [
        r["median_e2e_s"]
        for r in p2_rows
        if r.get("name") == "mh3_fp8_u8_all_hbm_resident"
        and r.get("ok_count", 0) == r.get("num_requests", 0)
        and r.get("ok_count", 0) > 0
        and r.get("duration_seconds") == 15.0
        and r.get("denoise_steps") == 4
        and r.get("concurrency") == 1
        and r.get("median_e2e_s")
    ]
    use_all_hbm = bool(hbm_15s) and (not fp8_15s or min(hbm_15s) < min(fp8_15s))
    use_fp8 = use_all_hbm or (bool(fp8_15s) and (not bf16_15s or min(fp8_15s) <= min(bf16_15s) * 1.05))
    dp_quant = "fp8" if use_fp8 else None
    res_tag = "All-HBM-Resident" if use_all_hbm else "Pallas"
    dp_label = f"DP=4 × Ulysses=8 (16 v7x chips, {'FP8' if use_fp8 else 'BF16'}, {res_tag})"

    dp_port = 30100 + idx
    proc_dp, _, _ = start_sglang_server(
        idx,
        config_name=f"dp4_rep{idx}",
        model_path=base_dir,
        port=dp_port,
        num_gpus=8,
        ulysses_degree=8,
        quantization=dp_quant,
        splash_varlen=False,
        disable_splash=False,
        adaln_online=(dp_quant is None),
        all_hbm_resident=use_all_hbm,
    )

    try:
        # Warm 768p 16:9 5s, 10s, 15s i2v on every replica in parallel!
        ep_local = f"http://127.0.0.1:{dp_port}"
        for dur in (5.0, 10.0, 15.0):
            w_p = make_minimax_payload(
                prompt=PROMPTS[0],
                task="fl2va",
                short_edge=768,
                aspect_ratio="16:9",
                duration_seconds=dur,
                num_inference_steps=5,
                seed=700 + int(dur),
                keyframe_uri=keyframes["16:9_768"],
            )
            submit_and_poll_video(ep_local, w_p)

        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/dp4_ready_pod_{idx}.txt", b"ready\n", content_type="text/plain")
        log(idx, f"Pod-{idx} warmed and ready for 16-chip DP=4 cluster benchmark!")

        if idx == 0:
            t_w = time.time()
            while time.time() - t_w < 2400:
                if all(
                    gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/dp4_ready_pod_{i}.txt") is not None
                    for i in range(4)
                ):
                    break
                time.sleep(5)

            endpoints = [f"http://{nodes[i]['ip']}:{30100 + i}" for i in range(4)]
            cluster_rows: list[dict[str, Any]] = []

            cluster_suites = [
                # 1. Exact Cloud TPU i2v 768P 15s (4-step) at c=4 and c=8 across 16 v7x chips
                {"label": "768p 16:9 15s i2v (4-step)", "mode": "fixed", "dur": 15.0, "steps": 5, "conc": 4, "n": 8},
                {"label": "768p 16:9 15s i2v (4-step)", "mode": "fixed", "dur": 15.0, "steps": 5, "conc": 8, "n": 12},
                # 2. 768p 16:9 15s i2v (25-step) at c=4
                {"label": "768p 16:9 15s i2v (25-step)", "mode": "fixed", "dur": 15.0, "steps": 26, "conc": 4, "n": 8},
                # 3. 768p 16:9 5s i2v (4-step) high-RPM burst at c=4 and c=8
                {"label": "768p 16:9 5s i2v (4-step)", "mode": "fixed", "dur": 5.0, "steps": 5, "conc": 4, "n": 12},
                {"label": "768p 16:9 5s i2v (4-step)", "mode": "fixed", "dur": 5.0, "steps": 5, "conc": 8, "n": 16},
                # 4. Mixed Production Distribution (50% 5s, 30% 10s, 20% 15s i2v @ 768p 16:9)
                {"label": "Mixed Dist 50/30/20 (5s/10s/15s 768p i2v, 4-step)", "mode": "mixed", "steps": 5, "conc": 4, "n": 10},
                {"label": "Mixed Dist 50/30/20 (5s/10s/15s 768p i2v, 4-step)", "mode": "mixed", "steps": 5, "conc": 8, "n": 16},
            ]

            mixed_durs = [5.0, 5.0, 10.0, 5.0, 15.0, 5.0, 10.0, 5.0, 10.0, 15.0]

            for s_i, cs in enumerate(cluster_suites):
                n_reqs = cs["n"]
                conc = cs["conc"]
                steps_grid = cs["steps"]
                denoise_steps = steps_grid - 1
                reqs_list = []
                for r_i in range(n_reqs):
                    dur = cs["dur"] if cs["mode"] == "fixed" else mixed_durs[r_i % len(mixed_durs)]
                    ep = endpoints[r_i % 4]
                    p = make_minimax_payload(
                        prompt=PROMPTS[r_i % len(PROMPTS)],
                        task="fl2va",
                        short_edge=768,
                        aspect_ratio="16:9",
                        duration_seconds=dur,
                        num_inference_steps=steps_grid,
                        seed=5000 + s_i * 100 + r_i,
                        keyframe_uri=keyframes["16:9_768"],
                    )
                    reqs_list.append((ep, p, dur))

                log(0, f"[Cluster 16-chip] Running {cs['label']} (c={conc}, n={n_reqs})...")
                t_c0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(lambda item: submit_and_poll_video(item[0], item[1]), reqs_list))
                wall_s = time.perf_counter() - t_c0

                oks = [o for o in outs if o.get("ok")]
                lats = sorted(o["e2e_s"] for o in oks)
                med_e2e = round(statistics.median(lats), 3) if lats else None
                p90_e2e = round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
                rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
                ok_vid_sec = sum(reqs_list[i][2] for i, o in enumerate(outs) if o.get("ok"))
                rtf = round(ok_vid_sec / wall_s, 3) if wall_s > 0 else 0.0

                crow = {
                    "cluster_config": dp_label,
                    "workload_label": cs["label"],
                    "denoise_steps": denoise_steps,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": len(oks),
                    "wall_s": round(wall_s, 3),
                    "median_e2e_s": med_e2e,
                    "p90_e2e_s": p90_e2e,
                    "rpm_videos_per_min": rpm,
                    "video_sec_per_wall_sec": rtf,
                }
                log(0, f"[Cluster 16-chip] DONE {cs['label']} c={conc}: ok={len(oks)}/{n_reqs} RPM={rpm}/min med={med_e2e}s RTF={rtf}x")
                cluster_rows.append(crow)

                all_rows = []
                for i in range(4):
                    raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                    if raw:
                        all_rows.extend(json.loads(raw.decode("utf-8")))
                summary_obj = {
                    "run_id": args.run_id,
                    "updated_utc": utc_now(),
                    "nodes": nodes,
                    "single_host_results": all_rows,
                    "cluster_dp4_results": cluster_rows,
                }
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/summary.json", json.dumps(summary_obj, indent=2).encode("utf-8"))
                md_txt = format_markdown_summary(args.run_id, nodes, all_rows, cluster_rows)
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")

            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt", b"ok\n", content_type="text/plain")
            log(0, "ALL MiniMax-H3 & FastH3 serving benchmarks complete! Signaled ALL_DONE.txt.")
        else:
            # Worker pods 1..3 keep their DP replica alive until pod-0 writes ALL_DONE.txt
            t_hold = time.time()
            while time.time() - t_hold < 7200:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    log(idx, "Received ALL_DONE.txt from pod-0. Shutting down cleanly.")
                    break
                time.sleep(5)
    finally:
        kill_sglang_processes(proc_dp)


if __name__ == "__main__":
    main()
