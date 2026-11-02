#!/usr/bin/env python3
"""Priority #3: Wan 2.2 A14B (`Wan2.2-I2V-A14B-Diffusers` & `Wan2.2-T2V-A14B-Diffusers` 28.6B Total / 14.3B Active Two-Stage MoE DiT) Serving Benchmark Suite on TPU v7x.

Runs across the 4 steady Flex TPU v7x hosts (`tpu-v7x-16-pool`, 4 chips / 8 TensorCores / 768 GiB HBM3e per host =
16 physical v7x chips / 32 TensorCores / 3,072 GiB HBM3e slice-wide):

Phase 3A — 4-Host Simultaneous Single-Host Configuration & Input-Distribution Matrix (`sglang-torchtpu`):
  - Pod 0 (`Wan2.2-I2V-A14B`): `wan22_i2v_bf16_u8` — BF16, 4 chips (`Ulysses=8`), Pallas FlashAttention
  - Pod 1 (`Wan2.2-I2V-A14B`):
      * `wan22_i2v_fp8_u8_all_hbm` — Online FP8 (`--quantization fp8`), 4 chips (`Ulysses=8`), All-HBM-Resident (`96 GiB/core`)
      * `wan22_i2v_fp8_cfg2_u4_all_hbm` — Online FP8, `CFG=2 x Ulysses=4`, All-HBM-Resident (2-pass CFG parallel scaling)
      * `wan22_i2v_fp8_u8_cpu_offload` — Online FP8, `Ulysses=8`, CPU-Offload Enc/VAE (ablation vs All-HBM-Resident)
  - Pod 2 (`Wan2.2-T2V-A14B`):
      * `wan22_t2v_bf16_u8` — BF16, 4 chips (`Ulysses=8`), Pallas FlashAttention
      * `wan22_t2v_fp8_2chip_u4_all_hbm` — Online FP8, 2 physical chips (`4` TensorCores, `Ulysses=4`), All-HBM-Resident
  - Pod 3 (`Wan2.2-T2V-A14B`):
      * `wan22_t2v_fp8_u8_all_hbm` — Online FP8, 4 chips (`Ulysses=8`), All-HBM-Resident (`96 GiB/core`)
      * `wan22_t2v_fp8_cfg2_u4_all_hbm` — Online FP8, `CFG=2 x Ulysses=4`, All-HBM-Resident
      * `wan22_t2v_bf16_u8_sdpa` — BF16, 4 chips (`Ulysses=8`), Stock SDPA (`SGLANG_TPU_DISABLE_SPLASH=1`)

  Input Distributions Swept (`16 fps`):
    * `832x480` (480p 16:9) `81` frames (`5.0s`) — `4`-step fast (`guidance=1.0`), `20`-step & `40`-step (`guidance=3.5`)
    * `1280x720` (720p 16:9) `81` frames (`5.0s`) — `4`-step fast (`guidance=1.0`), `20`-step & `40`-step (`guidance=3.5`)
    * `1280x720` (720p 16:9) `161` frames (`10.0s`) — `20`-step (`guidance=3.5`)
    * `720x1280` (720p 9:16 Portrait) `81` frames (`5.0s`) — `20`-step (`guidance=3.5`)
    * Single-host concurrency `c = 1, 2`

Phase 3B — Full 16-Chip `v7x-16` Slice (`DP=4` Replicas of `Wan2.2-I2V-A14B-Diffusers`) Throughput & `30 RPM` Sizing:
  - Swaps `/dev/shm` on Pods 2–3 to `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (strictly 1 model in `/dev/shm` at a time).
  - Load-balances `c = 4, 8` across all 4 hosts (`16` v7x chips) on `480p 5s`, `720p 5s` (`4`-step & `20`-step), and `720p 10s`.
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
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic spray and sunlight.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, reflections on wet asphalt.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, crisp fur detail.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist at sunrise.",
]

_PORT_SEED_COUNTER = 29810


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


def apply_sglang_wan22_tpu_patches(idx: int) -> None:
    """Apply verified SGLang v0.2.0 TPU v7x patches for Wan 2.2 (MRoPE, FP8 MXU, Splash toggle)."""
    sp = Path("/usr/local/lib/python3.12/site-packages")

    # 1. srt/layers/rotary_embedding/mrope.py: XLA-safe torch.where
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

    # 4. runtime/utils/hf_diffusers_utils.py: bypass HF download for local model directories
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

    # 5. runtime/loader/component_loaders/transformer_loader.py: allow native fallback on distributed without raising
    p_tl = sp / "sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py"
    if p_tl.exists():
        t_tl = p_tl.read_text()
        old_tl = '        if requested_distributed_execution:\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        new_tl = '        if requested_distributed_execution:\n            return\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        if old_tl in t_tl and new_tl not in t_tl:
            t_tl = t_tl.replace(old_tl, new_tl, 1)
            p_tl.write_text(t_tl)

    # 6. runtime/layers/parallel_conv.py: bypass halo exchange on TPU
    p_pc = sp / "sglang/multimodal_gen/runtime/layers/parallel_conv.py"
    if p_pc.exists():
        t_pc = p_pc.read_text()
        old_pc = """def _spatial_parallel_conv_forward(
    module: nn.Module,
    x: torch.Tensor,
    conv_forward,
    *,
    height_pad_mode: str,
    match_conv3d_format: bool = False,
) -> torch.Tensor:
    # send and recv halo"""
        new_pc = """def _spatial_parallel_conv_forward(
    module: nn.Module,
    x: torch.Tensor,
    conv_forward,
    *,
    height_pad_mode: str,
    match_conv3d_format: bool = False,
) -> torch.Tensor:
    if x.device.type == "tpu":
        if match_conv3d_format:
            x = _match_conv3d_input_format(x, module.weight)
        return conv_forward(x)
    # send and recv halo"""
        if old_pc in t_pc and new_pc not in t_pc:
            p_pc.write_text(t_pc.replace(old_pc, new_pc, 1))

    # 7. configs/models/vaes/wanvae.py & runtime/models/vaes/wanvae.py: disable use_parallel_encode on TPU
    p_wvc = sp / "sglang/multimodal_gen/configs/models/vaes/wanvae.py"
    if p_wvc.exists():
        t_wvc = p_wvc.read_text()
        if "use_parallel_encode: bool = True" in t_wvc:
            p_wvc.write_text(t_wvc.replace("use_parallel_encode: bool = True", "use_parallel_encode: bool = False", 1))

    p_wvr = sp / "sglang/multimodal_gen/runtime/models/vaes/wanvae.py"
    if p_wvr.exists():
        t_wvr = p_wvr.read_text()
        old_wvr = 'self.use_parallel_encode = getattr(config, "use_parallel_encode", False)'
        new_wvr = "self.use_parallel_encode = False"
        if old_wvr in t_wvr:
            p_wvr.write_text(t_wvr.replace(old_wvr, new_wvr, 1))

    log(idx, "All SGLang TPU v7x Wan 2.2 runtime patches (MRoPE, FP8 MXU, Splash toggle, LocalHFVerify, FallbackBypass, VAEParallelConvBypass, DisableParallelEncode) applied!")


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
    for sz in ("832x480", "1280x720", "720x1280"):
        w, h = (int(x) for x in sz.split("x"))
        p = input_dir / f"wan_ref_{sz}.png"
        img = Image.new("RGB", (w, h), (22, 42, 68))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(28 + 150 * (y / max(h, 1))))
            g = min(255, int(52 + 100 * (y / max(h, 1))))
            b = min(255, int(92 + 80 * (1.0 - y / max(h, 1))))
            d.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        d.ellipse([w // 3, h // 4, w // 3 + 120, h // 4 + 120], fill=(248, 208, 120))
        img.save(p, format="PNG")
        out[sz] = str(p)
    return out


def stage_wan_repo_from_gcs(idx: int, bucket: str, repo_id: str) -> str:
    """Strictly stage ONE Wan 2.2 repo at a time under /dev/shm/Wan-AI/<name> (~125 GiB)."""
    for stale in (
        "/dev/shm/MiniMax-H3",
        "/dev/shm/FastH3",
        "/dev/shm/LTX-2.5",
        "/dev/shm/LTX-2.5-Diffusers",
        "/dev/shm/hf_cache",
        "/dev/shm/sglang_diffusion_cache",
    ):
        shutil.rmtree(stale, ignore_errors=True)

    wan_root = Path("/dev/shm/Wan-AI")
    target_dir = Path("/dev/shm") / repo_id
    if wan_root.exists():
        for child in wan_root.iterdir():
            if child != target_dir:
                shutil.rmtree(child, ignore_errors=True)

    fast_gcs_sync_dir(
        idx,
        bucket,
        f"models/{repo_id}/",
        target_dir,
        exclude_prefixes=("assets/", "examples/"),
    )
    assert (target_dir / "model_index.json").exists(), f"Missing {target_dir / 'model_index.json'}"
    log(idx, f"Staged {repo_id} in {target_dir} (RAM: {host_ram_summary()})")
    return str(target_dir)


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
            "rm -f /tmp/libtpu_lockfile /dev/shm/torch_tpu_pg_store_*; rm -rf /dev/shm/wan22_out_*; sleep 2",
        ],
        check=False,
    )


def start_sglang_wan22(
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

    out_dir = Path(f"/dev/shm/wan22_out_{config_name}")
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
    raise TimeoutError(f"Timed out waiting for sglang Wan 2.2 [{config_name}]:\n{tail}")


def call_sglang_video(endpoint: str, payload: dict[str, Any], timeout_s: float = 2400.0) -> dict[str, Any]:
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
            }
        if st.get("status") == "failed":
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))[:800]}
    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}


def run_wan_sweep(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    cfg_meta: dict[str, Any],
    endpoint: str,
    workloads: list[dict[str, Any]],
    is_i2v: bool,
    refs: dict[str, str],
    pod_rows: list[dict[str, Any]],
) -> None:
    warmed_shapes: set[tuple[str, int, bool]] = set()
    for w_i, wl in enumerate(workloads):
        sz = wl["size"]
        nf = int(wl["num_frames"])
        dur = float(wl["dur"])
        steps = int(wl["steps"])
        guidance = float(wl["guidance"])
        conc = int(wl.get("conc", 1))
        n_reqs = int(wl.get("n", max(2, conc)))
        is_guided = guidance > 1.0

        shape_key = (sz, nf, is_guided)
        cold_s = None
        if shape_key not in warmed_shapes:
            p_warm: dict[str, Any] = {
                "prompt": PROMPTS[0],
                "size": sz,
                "num_frames": nf,
                "fps": 16,
                "num_inference_steps": 4,
                "guidance_scale": guidance,
                "guidance_scale_2": guidance,
                "generator_device": "cpu",
                "seed": 100 + w_i,
            }
            if is_i2v:
                p_warm["input_reference"] = refs[sz]
            log(idx, f"[{cfg_meta['config']}] Warming shape {shape_key} ({wl['label']})...")
            w_res = call_sglang_video(endpoint, p_warm)
            cold_s = w_res.get("e2e_s")
            log(idx, f"[{cfg_meta['config']}] Warmup {wl['label']}: ok={w_res['ok']} e2e={cold_s}s err={w_res.get('error', '')[:160]}")
            if w_res["ok"]:
                warmed_shapes.add(shape_key)
            else:
                pod_rows.append(
                    {
                        **cfg_meta,
                        "workload": wl["label"],
                        "task": "i2v" if is_i2v else "t2v",
                        "size": sz,
                        "num_frames": nf,
                        "duration_s": dur,
                        "steps": steps,
                        "guidance_scale": guidance,
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
                "fps": 16,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "guidance_scale_2": guidance,
                "generator_device": "cpu",
                "seed": 2000 + w_i * 100 + r_i,
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
            "task": "i2v" if is_i2v else "t2v",
            "size": sz,
            "num_frames": nf,
            "duration_s": dur,
            "steps": steps,
            "guidance_scale": guidance,
            "concurrency": conc,
            "cold_compile_s": cold_s,
            "warm_median_e2e_s": med_s,
            "mean_inference_s": mean_inf,
            "ms_per_step": ms_step,
            "rpm": rpm,
            "video_sec_per_wall_sec": rtf,
            "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
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
        f"# Priority #3 — Wan 2.2 A14B (I2V-A14B & T2V-A14B MoE DiT) Serving on TPU v7x (`{run_id}`)",
        "",
        f"- **Updated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `tpu-v7x-cluster` / `tpu-v7x-16-pool` (Steady Flex `v7x-16`, `16` physical chips / `32` TensorCores / `3,072 GiB` HBM3e)",
        f"- **Registered Hosts:** `{json.dumps({k: v['hostname'] for k, v in sorted(nodes.items())})}`",
        "",
        "## 1. Single-Host (`4x` & `2x` TPU v7x Chips) Configuration & Input-Distribution Matrix",
        "",
        "| Model | Config | Prec. | Chips (Cores) | Mode | Workload | Canvas | Frames | Steps | Conc. | Cold Compile (s) | Warm Med E2E (s) | ms/step | Host RPM | Video-s / Wall-s | OK |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_host_rows:
        lines.append(
            f"| `{r['model']}` | `{r['config']}` | `{r['precision']}` | {r['chips']} ({r['num_gpus']}) | `{r['mode']}` | "
            f"**{r['workload']}** (`{r['task']}`) | `{r['size']}` | {r['num_frames']} | {r['steps']} | {r['concurrency']} | "
            f"{r.get('cold_compile_s') or '-'} | **{r.get('warm_median_e2e_s') or '-'}** | {r.get('ms_per_step') or '-'} | "
            f"**{r.get('rpm') or 0:.2f}** | {r.get('video_sec_per_wall_sec') or 0:.2f}x | `{r['ok']}` |"
        )
    if cluster_rows:
        lines.extend(
            [
                "",
                "## 2. Full `v7x-16` Slice (`16x TPU v7x Chips`, `DP=4` Replicas) Throughput & `30 RPM` Sizing",
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
    ap.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
    ap.add_argument("--run-id", default="v7x_flex_wan22_serve_p3")
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

    apply_sglang_wan22_tpu_patches(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    refs = generate_i2v_keyframes(Path("/dev/shm/wan22_inputs"))

    repo_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers" if idx in (0, 1) else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    is_i2v = idx in (0, 1)
    model_dir = stage_wan_repo_from_gcs(idx, args.gcs_bucket, repo_id)

    full_workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 3},
        {"label": "480p (832x480) 5s 81f (20-step)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "480p (832x480) 5s 81f (40-step std)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 40, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 5s 81f (4-step fast)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 3},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 5s 81f (40-step std)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 40, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 10s 161f (20-step)", "size": "1280x720", "num_frames": 161, "dur": 10.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "720p Portrait (720x1280) 5s 81f (20-step)", "size": "720x1280", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
    ]

    fp8_i2v_workloads = full_workloads + [
        {"label": "480p (832x480) 5s 81f (4-step, c=2)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "conc": 2, "n": 4},
    ]

    cfg2_workloads = [
        {"label": "480p (832x480) 5s 81f (20-step CFG=2xU4)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step CFG=2xU4)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 5s 81f (40-step CFG=2xU4)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 40, "guidance": 3.5, "n": 2},
    ]

    offload_ablation_workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x720) 5s 81f (4-step fast)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
    ]

    subhost_2chip_workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 3},
        {"label": "480p (832x480) 5s 81f (20-step)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
        {"label": "720p (1280x720) 5s 81f (4-step fast)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
    ]

    sdpa_workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "n": 2},
    ]

    port = 30130 + idx
    pod_rows: list[dict[str, Any]] = []

    # (cfg_name, prec, ngpus, cfg_par, uly, quant, disable_splash, all_hbm, mode_lbl, wl_list)
    configs_by_pod = {
        0: [
            ("wan22_i2v_bf16_u8", "BF16", 8, 1, 8, None, False, False, "Ulysses=8 Pallas (CPU-Offload Enc/VAE)", full_workloads),
        ],
        1: [
            ("wan22_i2v_fp8_u8_all_hbm", "FP8 (online)", 8, 1, 8, "fp8", False, True, "Ulysses=8 Pallas All-HBM-Resident (96GiB/core)", fp8_i2v_workloads),
            ("wan22_i2v_fp8_cfg2_u4_all_hbm", "FP8 (online)", 8, 2, 4, "fp8", False, True, "CFG=2 x Ulysses=4 Pallas All-HBM-Resident", cfg2_workloads),
            ("wan22_i2v_fp8_u8_cpu_offload", "FP8 (online)", 8, 1, 8, "fp8", False, False, "Ulysses=8 Pallas (CPU-Offload Enc/VAE)", offload_ablation_workloads),
        ],
        2: [
            ("wan22_t2v_bf16_u8", "BF16", 8, 1, 8, None, False, False, "Ulysses=8 Pallas (CPU-Offload Enc/VAE)", full_workloads),
            ("wan22_t2v_fp8_2chip_u4_all_hbm", "FP8 (online)", 4, 1, 4, "fp8", False, True, "2-Chip Ulysses=4 Pallas All-HBM-Resident", subhost_2chip_workloads),
        ],
        3: [
            ("wan22_t2v_fp8_u8_all_hbm", "FP8 (online)", 8, 1, 8, "fp8", False, True, "Ulysses=8 Pallas All-HBM-Resident (96GiB/core)", full_workloads),
            ("wan22_t2v_fp8_cfg2_u4_all_hbm", "FP8 (online)", 8, 2, 4, "fp8", False, True, "CFG=2 x Ulysses=4 Pallas All-HBM-Resident", cfg2_workloads),
            ("wan22_t2v_bf16_u8_sdpa", "BF16", 8, 1, 8, None, True, False, "Ulysses=8 Stock SDPA (Splash disabled)", sdpa_workloads),
        ],
    }

    for cfg_name, prec, ngpus, cfg_par, uly, quant, dis_sp, all_hbm, mode_lbl, wl_list in configs_by_pod[idx]:
        try:
            proc, boot_s, _ = start_sglang_wan22(
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
                    "model": repo_id.split("/")[-1],
                    "config": cfg_name,
                    "engine": "sglang-torchtpu",
                    "precision": prec,
                    "chips": ngpus // 2,
                    "num_gpus": ngpus,
                    "mode": mode_lbl,
                    "boot_s": boot_s,
                }
                run_wan_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_prefix=gcs_prefix,
                    cfg_meta=meta,
                    endpoint=f"http://127.0.0.1:{port}",
                    workloads=wl_list,
                    is_i2v=is_i2v,
                    refs=refs,
                    pod_rows=pod_rows,
                )
            finally:
                kill_sglang_processes(proc)
        except Exception as exc:
            log(idx, f"[{cfg_name}] error: {exc}")

    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase3a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, "Phase 3A complete! Waiting for all 4 pods before starting Phase 3B (16-chip DP=4 cluster)...")
    t_s = time.time()
    while time.time() - t_s < 7200:
        ready_3a = sum(
            1 for i in range(4) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase3a_pod_{i}.txt") is not None
        )
        if idx == 0:
            all_host: list[dict[str, Any]] = []
            for i in range(4):
                r = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if r:
                    all_host.extend(json.loads(r.decode("utf-8")))
            md_txt = format_markdown_summary(args.run_id, nodes, all_host, [])
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")
        if ready_3a == 4:
            break
        time.sleep(10)

    # Phase 3B: Swap Pods 2-3 to Wan2.2-I2V-A14B-Diffusers (strictly 1 repo in /dev/shm) and run 16-chip DP=4
    dp_model_dir = stage_wan_repo_from_gcs(idx, args.gcs_bucket, "Wan-AI/Wan2.2-I2V-A14B-Diffusers")

    p1_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_1_results.json")
    p1_rows = json.loads(p1_raw.decode("utf-8")) if p1_raw else []
    hbm_ok = [
        r["warm_median_e2e_s"]
        for r in p1_rows
        if r.get("config") == "wan22_i2v_fp8_u8_all_hbm" and r.get("warm_median_e2e_s") and not r.get("error")
    ]
    off_ok = [
        r["warm_median_e2e_s"]
        for r in p1_rows
        if r.get("config") == "wan22_i2v_fp8_u8_cpu_offload" and r.get("warm_median_e2e_s") and not r.get("error")
    ]
    cfg2_720 = next(
        (
            r["warm_median_e2e_s"]
            for r in p1_rows
            if r.get("config") == "wan22_i2v_fp8_cfg2_u4_all_hbm"
            and r.get("size") == "1280x720"
            and r.get("steps") == 20
            and r.get("warm_median_e2e_s")
        ),
        None,
    )
    u8_720 = next(
        (
            r["warm_median_e2e_s"]
            for r in p1_rows
            if r.get("config") == "wan22_i2v_fp8_u8_all_hbm"
            and r.get("size") == "1280x720"
            and r.get("steps") == 20
            and r.get("warm_median_e2e_s")
        ),
        None,
    )
    use_all_hbm = len(hbm_ok) >= 4
    use_fp8 = use_all_hbm or len(off_ok) >= 2
    use_cfg2 = bool(cfg2_720 and u8_720 and cfg2_720 < u8_720 * 0.95)
    dp_cfg_par = 2 if use_cfg2 else 1
    dp_uly = 4 if use_cfg2 else 8
    dp_quant = "fp8" if use_fp8 else None
    dp_label = (
        f"DP=4 × {'CFG=2×Ulysses=4' if use_cfg2 else 'Ulysses=8'} "
        f"(16 v7x chips, {'FP8' if use_fp8 else 'BF16'}, {'All-HBM-Resident' if use_all_hbm else 'Pallas'})"
    )

    dp_port = 30230 + idx
    proc_dp, _, _ = start_sglang_wan22(
        idx,
        config_name=f"wan22_dp4_rep{idx}",
        model_path=dp_model_dir,
        port=dp_port,
        num_gpus=8,
        ulysses_degree=dp_uly,
        cfg_parallel_size=dp_cfg_par,
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
                ("480p (832x480) 5s 81f I2V (4-step fast)", "832x480", 81, 5.0, 4, 1.0, 4, 12),
                ("480p (832x480) 5s 81f I2V (4-step fast)", "832x480", 81, 5.0, 4, 1.0, 8, 16),
                ("720p (1280x720) 5s 81f I2V (4-step fast)", "1280x720", 81, 5.0, 4, 1.0, 4, 12),
                ("720p (1280x720) 5s 81f I2V (4-step fast)", "1280x720", 81, 5.0, 4, 1.0, 8, 16),
                ("720p (1280x720) 5s 81f I2V (20-step)", "1280x720", 81, 5.0, 20, 3.5, 4, 8),
                ("720p (1280x720) 5s 81f I2V (20-step)", "1280x720", 81, 5.0, 20, 3.5, 8, 8),
                ("720p (1280x720) 10s 161f I2V (20-step)", "1280x720", 161, 10.0, 20, 3.5, 4, 4),
            ]
            warmed_dp: set[tuple[str, int, bool]] = set()
            for lbl, sz, nf, dur, steps, guid, conc, n_req in dp_suites:
                is_g = guid > 1.0
                if (sz, nf, is_g) not in warmed_dp:
                    log(0, f"[Cluster 16-chip] Warming all 4 replicas on {(sz, nf, is_g)}...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
                        w_outs = list(
                            p.map(
                                lambda i: call_sglang_video(
                                    eps[i],
                                    {
                                        "prompt": PROMPTS[0],
                                        "size": sz,
                                        "num_frames": nf,
                                        "fps": 16,
                                        "num_inference_steps": 4,
                                        "guidance_scale": guid,
                                        "guidance_scale_2": guid,
                                        "input_reference": refs[sz],
                                        "generator_device": "cpu",
                                        "seed": 42,
                                    },
                                ),
                                range(4),
                            )
                        )
                    if all(o.get("ok") for o in w_outs):
                        warmed_dp.add((sz, nf, is_g))

                log(0, f"[Cluster 16-chip] Running {lbl} (c={conc}, n={n_req})...")
                t_c = time.perf_counter()
                reqs = [
                    (
                        eps[r % 4],
                        {
                            "prompt": PROMPTS[r % 4],
                            "size": sz,
                            "num_frames": nf,
                            "fps": 16,
                            "num_inference_steps": steps,
                            "guidance_scale": guid,
                            "guidance_scale_2": guid,
                            "input_reference": refs[sz],
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
            log(0, "ALL Wan 2.2 A14B benchmarks complete! Signaled ALL_DONE.txt.")
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
