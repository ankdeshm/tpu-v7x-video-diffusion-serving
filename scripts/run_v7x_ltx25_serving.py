#!/usr/bin/env python3
"""Priority #2: LTX-2.5 (Lightricks/LTX-2.5-Diffusers) Video+Audio Serving Benchmark Suite on TPU v7x.

Covers single-host (4x TPU v7x chips = 8 TensorCores / 768 GiB HBM) and 16-chip Flex slice (`DP=4 x Ulysses=8`)
across configurations & input distributions, including Atlas Cloud's Kling-V4 Proxy (`1080p` 1920x1088, 20s @ 24fps):
  - Pod 0: `sglang-torchtpu` — BF16, Ulysses=8 (`4` chips), `tpu_splash_attn` Pallas FlashAttention
  - Pod 1: `sglang-torchtpu` — Online FP8 (`--quantization fp8`), Ulysses=8 (`4` chips), `tpu_splash_attn` Pallas FlashAttention
  - Pod 2: `sglang-torchtpu` — BF16, `CFG=2 x Ulysses=4` (`4` chips) & `2`-chip (`Ulysses=4`) scaling ablation
  - Pod 3: `vllm-torchtpu` (`vLLM-Omni` v0.32.0) — BF16, Ulysses=8 (`4` chips), engine comparison on identical shapes

Input Distributions Swept:
  - Standard Distilled (8-step, unguided `guidance_scale=1.0`, convolutional VAE decode):
      * `960x544` 5s (`121` frames @ 24fps) T2VA & I2VA
      * `960x544` 10s (`241` frames @ 24fps) T2VA
      * `960x544` 20s (`481` frames @ 24fps) T2VA
      * `1280x704` (720p) 5s (`121` frames) & 10s (`241` frames) T2VA
  - Two-Stage Diffusion Decoder Upscaled (`use_diffusion_decoder=True` -> `1920x1088` 1080p):
      * `1920x1088` (1080p) 5s (`121` frames @ 24fps)
      * `1920x1088` (1080p) 20s (`481` frames @ 24fps, `bs=1`) — **Kling V4 Proxy Video Benchmark**
  - Quality Step Sweep: `8`-step distilled vs `20`-step & `40`-step dev schedule
  - Full 16-Chip Slice (`DP=4 x Ulysses=8`): `c=4, 8` throughput & 30-RPM slice sizing (no cost columns).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
import shutil
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


def fast_gcs_sync_dir(idx: int, bucket: str, gcs_prefix: str, local_dir: Path) -> int:
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
pgrep -f mock_metadata.py >/dev/null || (LD_PRELOAD="" nohup python3 /workspace/mock_metadata.py >/workspace/mock_metadata.log 2>&1 &)
sleep 1
"""
    subprocess.run(["bash", "-lc", setup_sh], check=True)


def ensure_engine_installed(idx: int, engine: str) -> None:
    if engine == "sglang":
        chk = subprocess.run(
            ["python3", "-c", "import shutil, sglang, sglang_tpu; assert shutil.which('ffmpeg')"],
            capture_output=True,
        )
        if chk.returncode == 0:
            return
        log(idx, "Bootstrapping sglang-torchtpu...")
        sh = r"""
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 gcc g++ make git curl netcat-openbsd >/dev/null
rm -rf /workspace/sglang
git clone --depth 1 https://github.com/sgl-project/sglang-torchtpu.git /workspace/sglang
python3 - <<'PY'
import importlib.metadata as md, tomllib
from pathlib import Path
keep_pkgs = ["torch", "torchvision", "torch-tpu", "libtpu", "jax", "jaxlib", "tpu-sync-torch", "transformers"]
Path("/tmp/tpu_abi_constraints.txt").write_text("\n".join(f"{p}=={md.version(p)}" for p in keep_pkgs if p) + "\n")
txt = Path("/workspace/sglang/python/pyproject_tpu.toml").read_text()
for marker in ("[[tool.uv.index]]", "[tool.uv.index]", "[tool.uv.sources]"):
    txt = txt.split(marker)[0]
drop = ('"torch==', '"torchvision==', '"torch_tpu==', '"tpu-sync-torch==', '"libtpu==', '"jax==', '"jaxlib==', '"torchao==')
lines = [l.replace('"transformers==5.17.0"', '"transformers>=5.17.0"') for l in txt.splitlines() if not any(l.strip().startswith(d) for d in drop)]
rendered = "\n".join(lines) + "\n"
tomllib.loads(rendered)
Path("/workspace/sglang/python/pyproject.toml").write_text(rendered)
PY
uv pip install --system -c /tmp/tpu_abi_constraints.txt "/workspace/sglang/python[diffusion_tpu,all_tpu,ray,runai]"
uv pip install --system --no-deps /workspace/sglang
"""
        subprocess.run(["bash", "-lc", sh], check=True)
    else:
        chk = subprocess.run(
            ["python3", "-c", "import shutil, vllm_omni, vllm_torchtpu; assert shutil.which('ffmpeg')"],
            capture_output=True,
        )
        if chk.returncode == 0:
            return
        log(idx, "Bootstrapping vllm-omni on vllm-torchtpu...")
        sh = r"""
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 gcc g++ make git curl netcat-openbsd >/dev/null
TORCH_BASE_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
(uv pip install --system --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-deps "torchaudio==${TORCH_BASE_VER}" || uv pip install --system --index-url https://download.pytorch.org/whl/cpu --no-deps torchaudio)
rm -rf /workspace/vllm-omni
git clone --branch v0.32.0rc1 --depth 1 https://github.com/vllm-project/vllm-omni.git /workspace/vllm-omni
python3 - <<'PY'
import importlib.metadata as md
from pathlib import Path
keep = ["torch", "torchvision", "torchaudio", "torch-tpu", "libtpu", "jax", "jaxlib", "tpu-sync-torch", "transformers", "vllm"]
cs = []
for p in keep:
    try: cs.append(f"{p}=={md.version(p)}")
    except Exception: pass
Path("/tmp/vllm_omni_constraints.txt").write_text("\n".join(cs) + "\n")
req = Path("/workspace/vllm-omni/requirements/common.txt")
req.write_text("\n".join(l for l in req.read_text().splitlines() if not l.strip().startswith(("kernels==", "cosmos-guardrail"))) + "\n")
PY
uv pip install --system "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8.0"
VLLM_OMNI_TARGET_DEVICE=cpu VLLM_OMNI_VERSION_OVERRIDE=0.32.0rc1 SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0rc1 \
  uv pip install --system --no-build-isolation -c /tmp/vllm_omni_constraints.txt /workspace/vllm-omni
uv pip install --system -c /tmp/vllm_omni_constraints.txt imageio imageio-ffmpeg opencv-python-headless av soundfile diffusers accelerate
"""
        subprocess.run(["bash", "-lc", sh], check=True)


def stage_ltx25_model(idx: int, bucket: str) -> str:
    from huggingface_hub import snapshot_download

    model_dir = Path("/dev/shm/LTX-2.5-Diffusers")
    marker = model_dir / ".ready"
    if not marker.exists():
        fast_gcs_sync_dir(idx, bucket, "models/Lightricks/LTX-2.5-Diffusers/", model_dir)
        log(idx, "Verifying Lightricks/LTX-2.5-Diffusers via huggingface_hub...")
        snapshot_download(
            repo_id="Lightricks/LTX-2.5-Diffusers",
            ignore_patterns=["*.mp4", "*.mov", "*.gif", "*.md"],
            local_dir=str(model_dir),
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )
        marker.write_text(utc_now())
    return str(model_dir)


def kill_servers() -> None:
    subprocess.run(
        ["bash", "-lc", "pkill -9 -f 'sglang.cli.main' || true; pkill -9 -f 'vllm_omni' || true; pkill -9 -f 'vllm serve' || true; rm -f /tmp/libtpu_lockfile || true"],
        check=False,
    )
    time.sleep(2)


def start_sglang_ltx25(
    idx: int,
    model_path: str,
    port: int,
    num_gpus: int = 8,
    ulysses_degree: int = 8,
    cfg_parallel: bool = False,
    quantization: str | None = None,
    log_path: Path = Path("/tmp/sglang_ltx25.log"),
) -> tuple[subprocess.Popen[Any], float]:
    kill_servers()
    env = os.environ.copy()
    env.update(
        {
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "PJRT_DEVICE": "TPU",
            "TPU_ML_PLATFORM": "PyTorch",
            "LIBTPU_INIT_ARGS": "--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "TPU_VISIBLE_CHIPS": ",".join(str(c) for c in range(max(1, num_gpus // 2))),
        }
    )
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
        "--save-output",
        "--output-path",
        "/dev/shm/ltx25_outputs",
        "--log-level",
        "info",
    ]
    if cfg_parallel:
        cmd.append("--enable-cfg-parallel")
    if quantization:
        cmd.extend(["--quantization", quantization])

    t0 = time.time()
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"sglang exited ({proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, round(time.time() - t0, 2)
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    raise TimeoutError("Timed out waiting for sglang LTX-2.5 server")


def start_vllm_omni_ltx25(
    idx: int,
    model_path: str,
    port: int,
    ulysses_degree: int = 8,
    log_path: Path = Path("/tmp/vllm_omni_ltx25.log"),
) -> tuple[subprocess.Popen[Any], float]:
    kill_servers()
    env = os.environ.copy()
    env.update(
        {
            "LD_PRELOAD": "/workspace/metadata_hook.so",
            "PJRT_DEVICE": "TPU",
            "TPU_ML_PLATFORM": "PyTorch",
            "TPU_VISIBLE_CHIPS": "0,1,2,3",
        }
    )
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--usp",
        str(ulysses_degree),
    ]
    t0 = time.time()
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm-omni exited ({proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, round(time.time() - t0, 2)
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    raise TimeoutError("Timed out waiting for vllm-omni LTX-2.5 server")


def call_sglang_video(endpoint: str, payload: dict[str, Any], timeout_s: float = 2400.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)}
    vid_id = job["id"]
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(0.5)
        with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
            st = json.loads(r.read().decode("utf-8"))
        if st.get("status") == "completed":
            e2e = time.perf_counter() - t0
            return {
                "ok": True,
                "e2e_s": round(e2e, 3),
                "inference_s": st.get("inference_time_s") or round(e2e, 3),
            }
        if st.get("status") == "failed":
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))}
    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}


def call_vllm_omni_video(endpoint: str, payload: dict[str, Any], timeout_s: float = 2400.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    body = {
        "prompt": payload["prompt"],
        "width": int(payload["size"].split("x")[0]),
        "height": int(payload["size"].split("x")[1]),
        "num_frames": payload["num_frames"],
        "fps": payload.get("fps", 24),
        "num_inference_steps": payload["num_inference_steps"],
        "guidance_scale": payload.get("guidance_scale", 1.0),
        "seed": payload.get("seed", 42),
    }
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        # Handle either sync base64 or async job ID
        if "id" in raw and raw.get("status") in ("queued", "in_progress"):
            vid_id = raw["id"]
            while time.perf_counter() - t0 < timeout_s:
                time.sleep(0.5)
                with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
                    st = json.loads(r.read().decode("utf-8"))
                if st.get("status") == "completed":
                    break
                if st.get("status") == "failed":
                    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))}
        e2e = round(time.perf_counter() - t0, 3)
        return {"ok": True, "e2e_s": e2e, "inference_s": e2e}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)}


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
    kill_servers()

    gcs_upload_bytes(
        args.gcs_bucket,
        f"{gcs_prefix}/nodes/node_{idx}.json",
        json.dumps({"idx": idx, "ip": ip, "hostname": hostname}).encode("utf-8"),
    )
    nodes: dict[int, dict[str, Any]] = {}
    t_w = time.time()
    while len(nodes) < 4 and time.time() - t_w < 600:
        for i in range(4):
            raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
            if raw:
                nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < 4:
            time.sleep(3)

    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    ensure_engine_installed(idx, "vllm_omni" if idx == 3 else "sglang")
    if idx == 3:
        ensure_engine_installed(idx, "sglang")  # Needed for Phase 2B 4-replica DP cluster
    model_dir = stage_ltx25_model(idx, args.gcs_bucket)

    workloads = [
        {"label": "544p (960x544) 5s 121f (8-step distill)", "size": "960x544", "out_res": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "n": 3},
        {"label": "544p (960x544) 10s 241f (8-step distill)", "size": "960x544", "out_res": "960x544", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "upscale": False, "n": 2},
        {"label": "544p (960x544) 20s 481f (8-step distill)", "size": "960x544", "out_res": "960x544", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": False, "n": 2},
        {"label": "720p (1280x704) 5s 121f (8-step distill)", "size": "1280x704", "out_res": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": False, "n": 2},
        {"label": "720p (1280x704) 10s 241f (8-step distill)", "size": "1280x704", "out_res": "1280x704", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "upscale": False, "n": 2},
        {"label": "1080p Two-Stage (1920x1088) 5s 121f", "size": "960x544", "out_res": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "upscale": True, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f", "size": "960x544", "out_res": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "upscale": True, "n": 2},
        {"label": "720p (1280x704) 5s 121f (20-step dev)", "size": "1280x704", "out_res": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "upscale": False, "n": 2},
    ]

    port = 30110 + idx
    pod_rows: list[dict[str, Any]] = []

    configs_by_pod = {
        0: [("sglang_bf16_u8", "sglang", "BF16", 8, 8, False, None)],
        1: [("sglang_fp8_u8", "sglang", "FP8", 8, 8, False, "fp8")],
        2: [
            ("sglang_bf16_cfg2_u4", "sglang", "BF16", 8, 4, True, None),
            ("sglang_bf16_2chip_u4", "sglang", "BF16", 4, 4, False, None),
        ],
        3: [("vllm_omni_bf16_u8", "vllm_omni", "BF16", 8, 8, False, None)],
    }

    for cfg_name, eng, prec, ngpus, uly, cfg_p, quant in configs_by_pod[idx]:
        try:
            if eng == "sglang":
                proc, boot_s = start_sglang_ltx25(idx, model_dir, port, ngpus, uly, cfg_p, quant)
                caller = call_sglang_video
            else:
                proc, boot_s = start_vllm_omni_ltx25(idx, model_dir, port, uly)
                caller = call_vllm_omni_video
            try:
                ep = f"http://127.0.0.1:{port}"
                for w_i, wl in enumerate(workloads):
                    if eng == "vllm_omni" and wl["upscale"]:
                        continue
                    if ngpus == 4 and wl["num_frames"] > 241:
                        continue
                    p_warm = {
                        "prompt": PROMPTS[0],
                        "size": wl["size"],
                        "num_frames": wl["num_frames"],
                        "fps": 24,
                        "num_inference_steps": wl["steps"],
                        "guidance_scale": wl["guidance"],
                        "use_diffusion_decoder": wl["upscale"],
                        "generator_device": "cpu",
                        "seed": 100 + w_i,
                    }
                    warm = caller(ep, p_warm)
                    outs = []
                    t0 = time.perf_counter()
                    for r_i in range(wl["n"]):
                        p_r = dict(p_warm, prompt=PROMPTS[r_i % len(PROMPTS)], seed=1000 + w_i * 10 + r_i)
                        outs.append(caller(ep, p_r))
                    wall_s = time.perf_counter() - t0
                    oks = [o for o in outs if o.get("ok")]
                    med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
                    ms_step = round((med_s * 1000.0) / wl["steps"], 1) if med_s else None
                    rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
                    pod_rows.append(
                        {
                            "config": cfg_name,
                            "engine": eng,
                            "precision": prec,
                            "chips": ngpus // 2,
                            "workload": wl["label"],
                            "output_res": wl["out_res"],
                            "duration_s": wl["dur"],
                            "steps": wl["steps"],
                            "cold_compile_s": warm.get("e2e_s"),
                            "warm_median_e2e_s": med_s,
                            "ms_per_step": ms_step,
                            "rpm": rpm,
                            "video_sec_per_wall_sec": round((len(oks) * wl["dur"]) / wall_s, 3) if oks else 0.0,
                            "ok": f"{len(oks)}/{wl['n']}",
                            "error": None if oks else warm.get("error"),
                        }
                    )
                    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))
            finally:
                proc.terminate()
                kill_servers()
        except Exception as exc:
            log(idx, f"[{cfg_name}] failed: {exc}")

    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase2a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    t_s = time.time()
    while time.time() - t_s < 5400:
        if sum(1 for i in range(4) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase2a_pod_{i}.txt") is not None) == 4:
            break
        time.sleep(10)

    # Phase 2B: Full 16-Chip Slice DP=4 Cluster (including 30-RPM Kling-V4 Proxy sizing)
    dp_port = 30210 + idx
    proc_dp, _ = start_sglang_ltx25(idx, model_dir, dp_port, 8, 8, False, None)
    try:
        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_pod_{idx}.txt", f"http://{ip}:{dp_port}".encode("utf-8"), content_type="text/plain")
        if idx == 0:
            eps: dict[int, str] = {}
            while len(eps) < 4:
                for i in range(4):
                    b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_pod_{i}.txt")
                    if b:
                        eps[i] = b.decode("utf-8").strip()
                if len(eps) < 4:
                    time.sleep(3)
            cluster_rows = []
            for lbl, sz, nf, dur, upsc, conc, n_req in [
                ("544p 5s 121f (8-step)", "960x544", 121, 5.0, False, 4, 12),
                ("544p 5s 121f (8-step)", "960x544", 121, 5.0, False, 8, 16),
                ("1080p Kling-V4 Proxy 20s 481f", "960x544", 481, 20.0, True, 4, 8),
                ("1080p Kling-V4 Proxy 20s 481f", "960x544", 481, 20.0, True, 8, 12),
            ]:
                # Warm all 4 replicas on shape
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
                    list(p.map(lambda i: call_sglang_video(eps[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": 8, "guidance_scale": 1.0, "use_diffusion_decoder": upsc, "generator_device": "cpu", "seed": 42}), range(4)))
                t_c = time.perf_counter()
                reqs = [(eps[r % 4], {"prompt": PROMPTS[r % 4], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": 8, "guidance_scale": 1.0, "use_diffusion_decoder": upsc, "generator_device": "cpu", "seed": 900 + r}) for r in range(n_req)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_sglang_video(it[0], it[1]), reqs))
                ws = time.perf_counter() - t_c
                oks = [o for o in outs if o.get("ok")]
                rpm = round((len(oks) / ws) * 60.0, 3) if oks else 0.0
                cluster_rows.append(
                    {
                        "workload": lbl,
                        "concurrency": conc,
                        "median_e2e_s": round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None,
                        "slice_rpm": rpm,
                        "slices_for_30_rpm": math.ceil(30.0 / rpm) if rpm > 0 else None,
                        "video_sec_per_wall_sec": round((len(oks) * dur) / ws, 3) if oks else 0.0,
                        "ok": f"{len(oks)}/{n_req}",
                    }
                )
            all_host = []
            for i in range(4):
                r = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if r:
                    all_host.extend(json.loads(r.decode("utf-8")))
            gcs_upload_bytes(
                args.gcs_bucket,
                f"{gcs_prefix}/summary.json",
                json.dumps({"run_id": args.run_id, "single_host_results": all_host, "cluster_dp4_results": cluster_rows}, indent=2).encode("utf-8"),
            )
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt", b"ok\n", content_type="text/plain")
        else:
            t_h = time.time()
            while time.time() - t_h < 5400:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    break
                time.sleep(5)
    finally:
        proc_dp.terminate()
        kill_servers()


if __name__ == "__main__":
    main()
