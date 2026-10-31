#!/usr/bin/env python3
"""Priority #1: MiniMax-H3 & FastH3 (4-step VSA Distilled) Video+Audio Serving Benchmark Suite on TPU v7x.

Runs across the 4 steady Flex TPU v7x hosts (`ank-tpu-v7x-16-pool`, 4 chips / 8 TensorCores / 768 GB HBM per host =
16 physical v7x chips / 32 TensorCores / 3,072 GB HBM slice-wide):

Phase 1A — 4-Host Simultaneous Single-Host (4x TPU v7x chips = 8 TensorCores) Configuration & Input-Distribution Matrix:
  - Pod 0: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), BF16, Ulysses=8, `tpu_splash_attn` + Pallas Mosaic FlashAttn varlen (`SGLANG_TPU_SPLASH_VARLEN=1`)
  - Pod 1: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), Online FP8 (`--quantization fp8`), Ulysses=8, `tpu_splash_attn` + Pallas Mosaic FlashAttn varlen (`SGLANG_TPU_SPLASH_VARLEN=1`)
  - Pod 2: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition), Stock `tpu_splash_attn` (`SGLANG_TPU_SPLASH_VARLEN=0`, SDPA varlen) + Online AdaLN (`--minimax-h3-adaln-online`), Ulysses=8 (followed by 2-chip `Ulysses=4` ablation)
  - Pod 3: `FastH3` (`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` — 4-step VSA distilled MiniMax-H3), Ulysses=8, BF16 then Online FP8 (`--quantization fp8`)

  Input Distributions Swept (`bs=1` + concurrency `c=1, 2`):
    * Exact Atlas Cloud Spec: `i2v 768P 15s` (`fl2va`, 1344x768, 15.0s @ 24fps = 360 frames + 32kHz stereo audio) & `t2va 768P 15s`
      across 4 denoising steps (`num_inference_steps=5`), 12 steps (`13`), 25 steps (`26`), and 49 steps (`50`)
    * Standard 768p 16:9 Durations: `768p 5s` (124 frames), `768p 10s` (244 frames), `768p 15s` (360 frames)
    * Multi-Aspect & Preview Tiers: `768p 9:16 Portrait 5s` (768x1344), `512p 16:9 5s & 15s` (896x512)

Phase 1B — Full 16-Chip `v7x-16` Slice (`4x` Data-Parallel Replicas, `DP=4 x Ulysses=8`) Throughput & 30-RPM Sizing:
  - Load-balances concurrent requests (`c = 4, 8, 12`) across all 4 hosts (`16` v7x chips) on `768p 15s`, `768p 5s`,
    and a realistic mixed video production distribution (`50%` 5s, `30%` 10s, `20%` 15s).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
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

PROMPTS = [
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic sea spray and Gull calls.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, crisp tire hiss and city ambience.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, soft footsteps crunching in snow.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist, gentle mountain wind and distant stream.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][pod-{idx}] {msg}", flush=True)


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
        except Exception as exc:
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


def fast_gcs_sync_dir(idx: int, bucket: str, gcs_prefix: str, local_dir: Path) -> int:
    """Multi-threaded chunked download from regional GCS into /dev/shm."""
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs = [
        (it["name"][len(gcs_prefix) :], int(it["size"]))
        for it in items
        if not it["name"].endswith("/") and "/.cache/" not in it["name"]
    ]
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


def ensure_sglang_tpu_installed(idx: int) -> None:
    check = subprocess.run(
        ["python3", "-c", "import shutil, sglang, sglang_tpu, av, cv2, diffusers; assert shutil.which('ffmpeg') and shutil.which('ffprobe')"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        log(idx, "sglang + sglang_tpu + ffmpeg already pre-installed in container.")
        return

    log(idx, "Bootstrapping ffmpeg + sglang-torchtpu in container (~25s)...")
    bootstrap_sh = r"""
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 gcc g++ make git curl netcat-openbsd >/dev/null
rm -rf /workspace/sglang
git clone --depth 1 https://github.com/sgl-project/sglang-torchtpu.git /workspace/sglang
python3 - <<'PY'
import importlib.metadata as md
from pathlib import Path

keep_pkgs = ["torch", "torchvision", "torch-tpu", "libtpu", "jax", "jaxlib", "tpu-sync-torch", "transformers"]
constraints = []
for p in keep_pkgs:
    try:
        constraints.append(f"{p}=={md.version(p)}")
    except Exception:
        pass
Path("/tmp/tpu_abi_constraints.txt").write_text("\n".join(constraints) + "\n")

import tomllib

txt = Path("/workspace/sglang/python/pyproject_tpu.toml").read_text()
for marker in ("[[tool.uv.index]]", "[tool.uv.index]", "[tool.uv.sources]"):
    txt = txt.split(marker)[0]
drop_prefixes = ('"torch==', '"torchvision==', '"torch_tpu==', '"tpu-sync-torch==', '"libtpu==', '"jax==', '"jaxlib==', '"torchao==')
lines = []
for line in txt.splitlines():
    s = line.strip()
    if any(s.startswith(dp) for dp in drop_prefixes):
        continue
    if s.startswith('"transformers=='):
        line = line.replace('"transformers==5.17.0"', '"transformers>=5.17.0"')
    lines.append(line)
rendered = "\n".join(lines) + "\n"
tomllib.loads(rendered)
Path("/workspace/sglang/python/pyproject.toml").write_text(rendered)

splash_p = Path("/workspace/sglang/sglang_tpu/multimodal_gen/tpu_splash_attn.py")
stxt = splash_p.read_text()
target = "        return self._sdpa.forward_varlen("
replacement = '''        import os
        if self.use_splash and os.environ.get("SGLANG_TPU_SPLASH_VARLEN", "0") == "1":
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return self._sdpa.forward_varlen('''
if target in stxt and "SGLANG_TPU_SPLASH_VARLEN" not in stxt:
    splash_p.write_text(stxt.replace(target, replacement, 1))
PY
uv pip install --system -c /tmp/tpu_abi_constraints.txt "/workspace/sglang/python[diffusion_tpu,all_tpu,ray,runai]"
uv pip install --system --no-deps /workspace/sglang
"""
    subprocess.run(["bash", "-lc", bootstrap_sh], check=True)
    log(idx, "sglang-torchtpu bootstrap complete!")


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
pgrep -f mock_metadata.py >/dev/null || (LD_PRELOAD="" nohup python3 /workspace/mock_metadata.py >/workspace/mock_metadata.log 2>&1 &)
sleep 1
"""
    subprocess.run(["bash", "-lc", setup_sh], check=True)


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


def stage_minimax_models(idx: int, bucket: str, need_base: bool, need_fasth3: bool) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN") or None
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    hf_cache = Path("/dev/shm/hf_cache")
    hf_cache.mkdir(parents=True, exist_ok=True)

    base_dir = Path("/dev/shm/MiniMax-H3")
    if need_base:
        marker = base_dir / ".ready_fl2va"
        if not marker.exists():
            fast_gcs_sync_dir(idx, bucket, "models/MiniMaxAI/MiniMax-H3/", base_dir)
            # Ensure all FL2VA shards are present (self-heal from HF Xet at multi-GB/s if any shard still streaming)
            t0 = time.time()
            log(idx, "Verifying / completing MiniMaxAI/MiniMax-H3 FL2VA partition via huggingface_hub...")
            snapshot_download(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=["FL2VA/*", "model_index.json", "modular_model_index.json"],
                ignore_patterns=["*.mp4", "*.mov", "*.gif", "docs/*"],
                local_dir=str(base_dir),
                token=hf_token,
                max_workers=16,
            )
            assert (base_dir / "FL2VA" / "model_index.json").exists(), "Missing FL2VA/model_index.json!"
            marker.write_text("ok\n")
            log(idx, f"MiniMaxAI/MiniMax-H3 FL2VA ready in /dev/shm/MiniMax-H3 ({time.time() - t0:.1f}s)")

    fasth3_repo = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
    if need_fasth3:
        t0 = time.time()
        log(idx, "Pre-caching FastH3 4-step VSA checkpoint + overlay into /dev/shm/hf_cache...")
        snapshot_download(
            repo_id="kevin-mi/FastH3-4step-Preview-overlay",
            revision="f769cb8001dae335089de7b250364335bc7cb183",
            cache_dir=str(hf_cache / "hub"),
            token=hf_token,
            max_workers=16,
        )
        snapshot_download(
            repo_id=fasth3_repo,
            cache_dir=str(hf_cache / "hub"),
            ignore_patterns=["*.mp4", "*.mov", "*.gif"],
            token=hf_token,
            max_workers=16,
        )
        log(idx, f"FastH3 + overlay cached in /dev/shm/hf_cache ({time.time() - t0:.1f}s)")

    return str(base_dir), fasth3_repo


def build_tpu_server_env(
    *,
    num_gpus: int,
    splash_varlen: bool,
    port_seed: int = 29610,
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
    ):
        env.pop(k, None)

    if num_gpus == 4:
        # 2 physical v7x chips = 4 TensorCores (1x2x1 sub-host slice)
        chips_bounds = "1,2,1"
        topo = "1x2x1"
        accel = "tpu7x-4"
        vis_devs = "0,1"
        metrics_ports = "8431,8432,8433,8434"
    else:
        # Full 4 physical v7x chips = 8 TensorCores (2x2x1 single host)
        chips_bounds = "2,2,1"
        topo = "2x2x1"
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
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": topo,
            "TOPOLOGY": topo,
            "TPU_ACCELERATOR_TYPE": accel,
            "TPU_VISIBLE_DEVICES": vis_devs,
            "TPU_RUNTIME_METRICS_PORTS": metrics_ports,
            "GCE_METADATA_HOST": "127.0.0.1:80",
            "GCE_METADATA_IP": "127.0.0.1",
            "GOOGLE_CLOUD_PROJECT": "tpu-prod-env-one-vm",
            "GCLOUD_PROJECT": "tpu-prod-env-one-vm",
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "HF_HOME": "/dev/shm/hf_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TMPDIR": str(cache_root / "tmp"),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_TPU_SPLASH_VARLEN": "1" if splash_varlen else "0",
        }
    )
    if Path("/var/run/tpu-plugin/vbar.sock").is_socket():
        env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
    return env


def kill_sglang_processes() -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; rm -f /tmp/libtpu_lockfile; sleep 3",
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
    splash_varlen: bool,
    adaln_online: bool = False,
    extra_args: list[str] | None = None,
    ready_timeout_s: int = 1500,
) -> tuple[subprocess.Popen, float, Path]:
    kill_sglang_processes()
    out_dir = Path(f"/dev/shm/sglang_out_{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/workspace/sglang_{config_name}.log")

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--modality",
        "video",
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
        "--output-path",
        str(out_dir),
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])
    if adaln_online:
        cmd.append("--minimax-h3-adaln-online")
    if extra_args:
        cmd.extend(extra_args)

    env = build_tpu_server_env(num_gpus=num_gpus, splash_varlen=splash_varlen)
    log(idx, f"Starting [{config_name}]: {' '.join(cmd)} (splash_varlen={splash_varlen})")
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
                        log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s!")
                        return proc, boot_s, log_path
            except Exception:
                pass
        time.sleep(3)

    proc.kill()
    tail = log_path.read_text(errors="replace")[-4000:]
    raise TimeoutError(f"Server [{config_name}] timed out after {ready_timeout_s}s:\n{tail}")


def submit_and_poll_video(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 2400.0,
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
        if shape_key not in warmed_shapes:
            # Compile/warmup pass (use 5 sigma points = 4 steps so compile is fast!)
            warm_steps = 5 if config_meta["is_fasth3"] else min(steps_grid, 5)
            log(
                idx,
                f"[{config_meta['name']}] Warming XLA shape {shape_key} (grid={warm_steps})...",
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
            w_res = submit_and_poll_video(endpoint, w_payload)
            cold_compile_s = w_res["e2e_s"]
            log(
                idx,
                f"[{config_meta['name']}] Warmup {shape_key}: ok={w_res['ok']} e2e={cold_compile_s}s err={w_res.get('error', '')[:160]}",
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
        f"# Priority #1 — MiniMax-H3 & FastH3 Video+Audio Serving on TPU v7x (`{run_id}`)",
        "",
        f"- **Updated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `ank-tpu-v7x` / `ank-tpu-v7x-16-pool` (Steady Flex `v7x-16`, `4` hosts × `4` physical `v7x` chips = `16` physical chips / `32` TensorCores / `3,072 GiB` HBM3e)",
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
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
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

    # 2. Install sglang-torchtpu if needed, compile 2x2x1 metadata hook, create keyframes, stage checkpoints
    ensure_sglang_tpu_installed(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    keyframes = generate_synthetic_keyframes(Path("/dev/shm/inputs"))
    base_dir, fasth3_repo = stage_minimax_models(
        idx,
        args.gcs_bucket,
        need_base=True,  # All 4 pods participate in Phase 1B 4x DP cluster too!
        need_fasth3=(idx == 3),
    )

    # Define input-distribution workloads
    base_workloads = [
        # Exact Atlas Cloud target: i2v 768P 15s (4-step, 12-step, 25-step, 49-step) + t2va 768P 15s
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
        proc, boot_s, _ = start_sglang_server(
            idx,
            config_name="mh3_bf16_u8_pallas",
            model_path=base_dir,
            port=port,
            num_gpus=8,
            ulysses_degree=8,
            quantization=None,
            splash_varlen=True,
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
                "attn_mode": "Pallas FlashAttn varlen",
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
            proc.terminate()
            kill_sglang_processes()

    elif idx == 1:
        proc, boot_s, _ = start_sglang_server(
            idx,
            config_name="mh3_fp8_u8_pallas",
            model_path=base_dir,
            port=port,
            num_gpus=8,
            ulysses_degree=8,
            quantization="fp8",
            splash_varlen=True,
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
                "attn_mode": "Pallas FlashAttn varlen",
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
            proc.terminate()
            kill_sglang_processes()

    elif idx == 2:
        # Sub-pass 2A: Stock SDPA varlen + AdaLN online on 4 chips (Ulysses=8)
        ablation_workloads = [
            {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        ]
        proc, boot_s, _ = start_sglang_server(
            idx,
            config_name="mh3_bf16_u8_sdpa_adaln",
            model_path=base_dir,
            port=port,
            num_gpus=8,
            ulysses_degree=8,
            quantization=None,
            splash_varlen=False,
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
                "attn_mode": "SDPA varlen + AdaLN-online",
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
            proc.terminate()
            kill_sglang_processes()

        # Sub-pass 2B: 2-chip (`num_gpus=4, ulysses_degree=4`, FP8 + Pallas FlashAttn) scaling test!
        try:
            proc2, boot2_s, _ = start_sglang_server(
                idx,
                config_name="mh3_fp8_u4_2chips",
                model_path=base_dir,
                port=port,
                num_gpus=4,
                ulysses_degree=4,
                quantization="fp8",
                splash_varlen=True,
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
                    "attn_mode": "Pallas FlashAttn varlen",
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
                proc2.terminate()
                kill_sglang_processes()
        except Exception as exc:
            log(idx, f"2-chip ablation note: {exc}")

    elif idx == 3:
        # Sub-pass 3A: FastH3 (4-step VSA distilled) BF16
        for q_mode, cfg_tag, prec_lbl in [
            (None, "fasth3_bf16_u8_pallas", "BF16"),
            ("fp8", "fasth3_fp8_u8_pallas", "FP8 (online)"),
        ]:
            try:
                proc, boot_s, _ = start_sglang_server(
                    idx,
                    config_name=cfg_tag,
                    model_path=fasth3_repo,
                    port=port,
                    num_gpus=8,
                    ulysses_degree=8,
                    quantization=q_mode,
                    splash_varlen=True,
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
                        "attn_mode": "Pallas FlashAttn varlen",
                        "boot_s": boot_s,
                    }
                    run_host_sweep(
                        idx,
                        bucket=args.gcs_bucket,
                        gcs_out_prefix=gcs_prefix,
                        config_meta=meta,
                        endpoint=f"http://127.0.0.1:{port}",
                        workloads=fasth3_workloads,
                        keyframes=keyframes,
                        existing_results=pod_results,
                    )
                finally:
                    proc.terminate()
                    kill_sglang_processes()
            except Exception as exc:
                log(idx, f"[{cfg_tag}] error: {exc}")

    # Mark Phase 1A complete for this pod
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase1a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, "Phase 1A complete! Waiting for all 4 pods before starting Phase 1B (16-chip DP=4 cluster)...")

    t_sync = time.time()
    while time.time() - t_sync < 5400:
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
    # Pick best precision (FP8 if pod-1 succeeded cleanly on 768p 15s, else BF16)
    p1_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_1_results.json")
    p1_rows = json.loads(p1_raw.decode("utf-8")) if p1_raw else []
    use_fp8 = any(r.get("ok_count", 0) > 0 and r.get("duration_seconds") == 15.0 for r in p1_rows)
    dp_quant = "fp8" if use_fp8 else None
    dp_label = f"DP=4 × Ulysses=8 (16 v7x chips, {'FP8' if use_fp8 else 'BF16'}, Pallas)"

    dp_port = 30100 + idx
    proc_dp, _, _ = start_sglang_server(
        idx,
        config_name=f"dp4_rep{idx}",
        model_path=base_dir,
        port=dp_port,
        num_gpus=8,
        ulysses_degree=8,
        quantization=dp_quant,
        splash_varlen=True,
    )

    try:
        # Warm 768p 16:9 5s, 10s, 15s i2v & t2va on every replica in parallel!
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
            while time.time() - t_w < 1800:
                if all(
                    gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/dp4_ready_pod_{i}.txt") is not None
                    for i in range(4)
                ):
                    break
                time.sleep(5)

            endpoints = [f"http://{nodes[i]['ip']}:{30100 + i}" for i in range(4)]
            cluster_rows: list[dict[str, Any]] = []

            cluster_suites = [
                # 1. Exact Atlas Cloud i2v 768P 15s (4-step) at c=4 and c=8 across 16 v7x chips
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
                total_vid_sec = 0.0
                for r_i in range(n_reqs):
                    dur = cs["dur"] if cs["mode"] == "fixed" else mixed_durs[r_i % len(mixed_durs)]
                    total_vid_sec += dur
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
            while time.time() - t_hold < 5400:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    log(idx, "Received ALL_DONE.txt from pod-0. Shutting down cleanly.")
                    break
                time.sleep(5)
    finally:
        proc_dp.terminate()
        kill_sglang_processes()


if __name__ == "__main__":
    main()
