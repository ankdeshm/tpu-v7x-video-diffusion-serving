#!/usr/bin/env python3
"""Suite 2 — SGLang-TorchTPU Video + Stereo-Audio Diffusion Serving Optimization on TPU v7x (`tpu-v7x-cluster-8-pool-b`).

Target Container:
  `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0`

Implements & Ablates 5 Production Serving Optimizations across 2 Indexed Pods (4 physical v7x chips = 8 TensorCores = 768 GiB HBM3e per pod):
  - Opt 1: True `v7x` FP8 MXU (`torch._scaled_mm(..., out_dtype=torch.bfloat16)` + bf16 activation quant & row-wise epilogue = 1,592.3 TFLOP/s/core vs 588.0 TFLOP/s fp32 path) + TPU `WeightOnlyFP8Linear` W8A8 enablement.
  - Opt 2: `--enable-torch-compile true` on DiT (`DenoisingStage._maybe_torch_compile`) AND VAE Decoder (`TpuOOTPlatform.supports_vae_torch_compile` + `LTX2AVDecodingStage` / `DecodingStage` compiled `vae.decode` with automatic eager fallback + Dynamo cache limit 64).
  - Opt 3: All-HBM-Resident (`768 GiB` HBM3e/host — zero CPU offload for DiT, Gemma/UMT5 text encoder, image encoder, and VAE).
  - Opt 4: Disaggregated Post-VAE MP4 Video+Audio Offload (`SGLANG_REMOTE_ENCODE_URL` streaming zero-copy RGB24 + f32le stereo PCM over HTTP to `video-encode-svc` on `c3-standard-44` AVX-512 CPU pool).
  - Opt 5: Sub-Host Multi-Replica Isolation (`DP=2 x Ulysses=4` = 2 isolated 2-chip servers/host; `DP=4 x Ulysses=2` = 4 isolated 1-chip servers/host) via `SGLANG_TPU_LOCAL_RANKS` VFIO pinning + mock GCE metadata interception + least-inflight load balancing.

Pod Assignments:
  - Pod 0 (`JOB_COMPLETION_INDEX == 0`): `Lightricks/LTX-2.5` (22B Joint Video + 24 kHz Stereo Audio) — 6-config step-by-step ablation & concurrency scaling (`544p 5s/10s/20s`, `720p 5s/10s/20-step`, `1080p 5s`, and `1080p 20s 481f` Kling-V4 Proxy).
  - Pod 1 (`JOB_COMPLETION_INDEX == 1`): `Wan-AI/Wan2.2-T2V-A14B-Diffusers` -> `/dev/shm` clean swap -> `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (27B total / 14B active MoE) across `FP8` vs `BF16` and `DP=1xU=8`, `DP=2xU=4`, `DP=4xU=2` (`480p`, `720p 5s`, `720p 10s 161f`, `720p Portrait`).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf_cache/hub")
os.environ.setdefault("SGLANG_DIFFUSION_CACHE_ROOT", "/dev/shm/sglang_diffusion_cache")

import argparse
import concurrent.futures
import io
import json
import math
import re
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
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

HOSTNAME = socket.gethostname()

PROMPTS = [
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic sea spray and Gull calls.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, crisp tire hiss and city ambience.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, soft footsteps crunching in snow.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist, gentle mountain wind and distant stream.",
    "Studio macro shot of golden espresso pouring into a warm ceramic cup in morning window light, rich crema swirling with subtle cafe soundscape.",
    "High-altitude tracking shot of an alpine glacier express train crossing a stone viaduct above a turquoise canyon river in crisp autumn sunlight.",
]

METADATA_HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const struct sockaddr_in *addr_in = (const struct sockaddr_in *)addr;
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip, INET_ADDRSTRLEN);
        if (strcmp(ip, "169.254.169.254") == 0) {
            struct sockaddr_in redirected = *addr_in;
            redirected.sin_addr.s_addr = inet_addr("127.0.0.1");
            redirected.sin_port = htons(80);
            return real_connect(sockfd, (const struct sockaddr *)&redirected, sizeof(redirected));
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""

MOCK_METADATA_PY = r"""#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


class MockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def forward_to_real_metadata(self):
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        response_body = None
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "CONSUMER_PROJECT_ID": "mock-tpu-project",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": "1,1,1",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": "2x2x1",
            "TPU_ACCELERATOR_TYPE": "tpu7x-8",
            "WRAP": "false,false,false",
            "ZONE": "us-central1-c",
        }
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            response_body = (
                "TPU_WORKER_ID: '0'\n"
                "ALT: 'false'\n"
                "CHIPS_PER_HOST_BOUNDS: '2,2,1'\n"
                "CONSUMER_PROJECT_ID: 'mock-tpu-project'\n"
                "ENABLE_ICI_RESILIENCY: 'false'\n"
                "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\n"
                "HOST_BOUNDS: '1,1,1'\n"
                "INJECT_SLICE_BUILDER_FAULT: ''\n"
                "NODE_ID: '0'\n"
                "TOPOLOGY: '2x2x1'\n"
                "TPU_ACCELERATOR_TYPE: 'tpu7x-8'\n"
                "WRAP: 'false,false,false'\n"
                "ZONE: 'us-central1-c'\n"
            ).encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints") or path.endswith(
            "/computeMetadata/v1/instance/attributes/instance-id"
        ):
            req = urllib.request.Request(
                "http://169.254.169.254/computeMetadata/v1/instance/hostname",
                headers={"Metadata-Flavor": "Google"},
            )
            hn = urllib.request.urlopen(req, timeout=2).read().decode().strip().split(".")[0]
            response_body = hn.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = b"tpu7x-8"
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"mock-tpu-project"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("127.0.0.1", 80), MockMetadataHandler)
    print("[mock_metadata] Serving on 127.0.0.1:80", file=sys.stderr, flush=True)
    httpd.serve_forever()
"""

_MOCK_META_PROC: subprocess.Popen[Any] | None = None
_PORT_SEQ = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][suite2-pod-{idx}] {msg}", flush=True)


def host_ram_summary() -> str:
    try:
        lines = subprocess.check_output(["free", "-h"], text=True).splitlines()
        return lines[1] if len(lines) > 1 else lines[0]
    except Exception:
        return "unknown"


def shm_df_summary() -> str:
    try:
        lines = subprocess.check_output(["df", "-h", "/dev/shm"], text=True).splitlines()
        return lines[-1] if lines else "unknown"
    except Exception:
        return "unknown"


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ---------------------------------------------------------------------------
# GCS Checkpointing & High-Throughput Parallel Sync
# ---------------------------------------------------------------------------


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(
    idx: int,
    bucket: str,
    obj: str,
    data: bytes,
    content_type: str = "application/json",
) -> None:
    for attempt in range(5):
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
            with urllib.request.urlopen(req, timeout=120):
                return
        except Exception as exc:
            if attempt == 4:
                log(idx, f"WARNING: GCS upload failed for gs://{bucket}/{obj}: {exc}")
                return
            time.sleep(2 * (attempt + 1))


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 3:
                return None
            time.sleep(2 * (attempt + 1))
    return None


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
        if name.endswith("/") or "/.cache/" in name or name.endswith(".ready"):
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


def load_existing_checkpoint(
    idx: int,
    bucket: str,
    jsonl_obj: str,
    local_jsonl: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int]] = set()
    if not resume:
        return rows, done_keys
    raw = gcs_download_bytes(bucket, jsonl_obj)
    if not raw:
        return rows, done_keys
    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    local_jsonl.write_bytes(raw)
    for ln in raw.decode("utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            rows.append(r)
            if r.get("ok_count", 0) > 0:
                done_keys.add((str(r.get("config")), str(r.get("workload")), int(r.get("concurrency", 1))))
        except Exception:
            pass
    log(idx, f"Resumed {len(rows)} rows ({len(done_keys)} completed keys) from gs://{bucket}/{jsonl_obj}")
    return rows, done_keys


def checkpoint_row(
    idx: int,
    bucket: str,
    gcs_prefix: str,
    local_jsonl: Path,
    local_summary: Path,
    pod_rows: list[dict[str, Any]],
    new_row: dict[str, Any],
) -> None:
    pod_rows.append(new_row)
    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(local_jsonl, "w", encoding="utf-8") as f:
        for r in pod_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_payload = {
        "suite": "suite2_sglang_opt",
        "pod_index": idx,
        "hostname": HOSTNAME,
        "updated_at": utc_now(),
        "total_rows": len(pod_rows),
        "completed_rows": sum(1 for r in pod_rows if r.get("ok_count", 0) > 0),
        "rows": pod_rows,
    }
    summary_bytes = json.dumps(summary_payload, indent=2, ensure_ascii=False).encode("utf-8")
    local_summary.write_bytes(summary_bytes)
    gcs_upload_bytes(
        idx,
        bucket,
        f"{gcs_prefix}/suite2_pod{idx}.jsonl",
        local_jsonl.read_bytes(),
        content_type="application/x-ndjson",
    )
    gcs_upload_bytes(
        idx,
        bucket,
        f"{gcs_prefix}/suite2_pod{idx}_summary.json",
        summary_bytes,
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Sub-Host GCE Metadata Interception & SGLang TPU v0.2.0 Runtime Patches
# ---------------------------------------------------------------------------


def ensure_subhost_isolation_ready(idx: int) -> None:
    """Compile /tmp/metadata_hook.so, start mock_metadata.py on 127.0.0.1:80, and add /etc/hosts override."""
    global _MOCK_META_PROC
    hook_c = Path("/tmp/metadata_hook.c")
    hook_so = Path("/tmp/metadata_hook.so")
    if not hook_so.exists():
        hook_c.write_text(METADATA_HOOK_C)
        subprocess.run(
            ["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"],
            check=True,
        )
        log(idx, "Compiled /tmp/metadata_hook.so for GCE metadata interception")

    try:
        hosts_txt = Path("/etc/hosts").read_text()
        if "metadata.google.internal" not in hosts_txt:
            with open("/etc/hosts", "a") as hf:
                hf.write("\n127.0.0.1 metadata.google.internal\n")
    except Exception:
        pass

    port80_up = False
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.5):
            port80_up = True
    except Exception:
        port80_up = False

    if not port80_up:
        mock_py = Path("/tmp/mock_metadata.py")
        mock_py.write_text(MOCK_METADATA_PY)
        env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        _MOCK_META_PROC = subprocess.Popen(
            ["python3", "-u", str(mock_py)],
            env=env_clean,
            stdout=open("/tmp/mock_metadata.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(25):
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:80/computeMetadata/v1/instance/attributes/tpu-env",
                    headers={"Metadata-Flavor": "Google"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as r:
                    if r.status == 200:
                        log(idx, "Started mock_metadata.py proxy on 127.0.0.1:80")
                        break
            except Exception:
                time.sleep(0.2)


def apply_suite2_sglang_tpu_patches(idx: int) -> None:
    """Idempotently apply all baseline LTX-2.5 / Wan 2.2 patches + 5 Suite 2 TPU v7x optimizations."""
    sp = Path("/usr/local/lib/python3.12/site-packages")
    rt = sp / "sglang/multimodal_gen/runtime"

    # --- Baseline 1: mrope.py XLA-safe torch.where ---
    p_mrope = sp / "sglang/srt/layers/rotary_embedding/mrope.py"
    t_mrope = p_mrope.read_text()
    old_mrope = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t"""
    new_mrope = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    num_pairs = x.shape[-1]
    lim1 = min(3 * mrope_section[1], num_pairs)
    lim2 = min(3 * mrope_section[2], num_pairs)
    lanes = torch.arange(num_pairs, device=x.device, dtype=torch.int32)
    m1 = ((lanes % 3) == 1) & (lanes < lim1)
    m2 = ((lanes % 3) == 2) & (lanes < lim2)
    out = torch.where(m1, x[1], x[0])
    out = torch.where(m2, x[2], out)
    return out"""
    if new_mrope not in t_mrope:
        assert old_mrope in t_mrope, "Missing mrope.py target"
        p_mrope.write_text(t_mrope.replace(old_mrope, new_mrope, 1))

    # --- Opt 1A: fp8_utils.py True v7x FP8 MXU (bf16 out_dtype = 1,592.3 TFLOP/s vs 588.0 TFLOP/s fp32) ---
    p_fp8 = sp / "sglang/srt/layers/quantization/fp8_utils.py"
    t_fp8 = p_fp8.read_text()
    if "_SGLANG_TPU_FP8_BF16_FAST" not in t_fp8:
        old_fp8_const = "TORCH_DEVICE_IDENTITY = None"
        new_fp8_const = (
            "TORCH_DEVICE_IDENTITY = None\n"
            "import os as _sgl_os\n"
            '_SGLANG_TPU_FP8_BF16_FAST = _sgl_os.environ.get("SGLANG_TPU_FP8_BF16_FAST", "1") == "1"'
        )
        assert old_fp8_const in t_fp8, "Missing TORCH_DEVICE_IDENTITY in fp8_utils.py"
        t_fp8 = t_fp8.replace(old_fp8_const, new_fp8_const, 1)

        old_fp8 = """    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
        new_fp8 = """    if input.device.type == "tpu":
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        if _SGLANG_TPU_FP8_BF16_FAST:
            target_dtype = (
                pre_quant_output_dtype
                if input_2d.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                else input.dtype
            ) or torch.bfloat16
            if input_2d.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                qinput = input_2d
                x_scale_bf16 = (
                    input_scale.to(torch.bfloat16)
                    if input_scale is not None
                    else torch.ones(1, device=input.device, dtype=torch.bfloat16)
                )
            elif input_scale is not None:
                x_bf16 = input_2d.to(torch.bfloat16)
                x_scale_bf16 = input_scale.to(torch.bfloat16).clamp(min=1e-12)
                qinput = (x_bf16 / x_scale_bf16).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            elif weight_scale.numel() == 1:
                x_bf16 = input_2d.to(torch.bfloat16)
                x_scale_bf16 = x_bf16.abs().amax().clamp(min=1e-12) * (1.0 / 448.0)
                qinput = (x_bf16 / x_scale_bf16).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            else:
                x_bf16 = input_2d.to(torch.bfloat16)
                x_scale_bf16 = x_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) * (1.0 / 448.0)
                qinput = (x_bf16 / x_scale_bf16).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            if weight_scale.numel() == 1 and x_scale_bf16.numel() == 1:
                out = torch._scaled_mm(
                    qinput,
                    weight,
                    scale_a=x_scale_bf16.float().reshape(1),
                    scale_b=weight_scale.float().reshape(1),
                    out_dtype=torch.bfloat16,
                )
                if bias is not None:
                    out = out + bias.to(torch.bfloat16)
                return out.to(target_dtype).view(*output_shape)
            unit_scale = torch.ones(1, device=input.device, dtype=torch.float32)
            out = torch._scaled_mm(
                qinput,
                weight,
                scale_a=unit_scale,
                scale_b=unit_scale,
                out_dtype=torch.bfloat16,
            )
            out = (out * x_scale_bf16.reshape(-1, 1)) * weight_scale.to(torch.bfloat16).reshape(1, -1)
            if bias is not None:
                out = out + bias.to(torch.bfloat16)
            return out.to(target_dtype).view(*output_shape)
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
        assert old_fp8 in t_fp8, "Missing apply_fp8_linear target in fp8_utils.py"
        p_fp8.write_text(t_fp8.replace(old_fp8, new_fp8, 1))

    # --- Opt 1B: weight_only_fp8.py allow TPU W8A8 FP8 linear ---
    p_wofp8 = rt / "layers/quantization/weight_only_fp8.py"
    if p_wofp8.exists():
        t_wo = p_wofp8.read_text()
        old_wo = """        x.device.type == "cuda"
        and weight.device.type == "cuda"
        and weight_scale.device.type == "cuda" """.rstrip()
        old_wo_exact = (
            '        x.device.type == "cuda"\n'
            '        and weight.device.type == "cuda"\n'
            '        and weight_scale.device.type == "cuda"'
        )
        new_wo_exact = (
            '        x.device.type in ("cuda", "tpu")\n'
            "        and weight.device.type == x.device.type\n"
            "        and weight_scale.device.type == x.device.type"
        )
        if new_wo_exact not in t_wo:
            assert old_wo_exact in t_wo, "Missing _can_apply_fused_w8a8_fp8_linear target in weight_only_fp8.py"
            p_wofp8.write_text(t_wo.replace(old_wo_exact, new_wo_exact, 1))

    # --- Baseline 2: tpu_splash_attn.py SGLANG_TPU_DISABLE_SPLASH gate ---
    p_splash = sp / "sglang_tpu/multimodal_gen/tpu_splash_attn.py"
    if p_splash.exists():
        stxt = p_splash.read_text()
        old_sp = "        self.use_splash = self.dropout == 0.0 and _check_splash_available()"
        new_sp = (
            "        import os\n"
            "        self.use_splash = (\n"
            '            os.environ.get("SGLANG_TPU_DISABLE_SPLASH", "0") != "1"\n'
            "            and self.dropout == 0.0\n"
            "            and _check_splash_available()\n"
            "        )"
        )
        if old_sp in stxt and "SGLANG_TPU_DISABLE_SPLASH" not in stxt:
            p_splash.write_text(stxt.replace(old_sp, new_sp, 1))

    # --- Opt 2A: sglang_tpu/multimodal_gen/platform.py Dynamo cache size 64 + supports_vae_torch_compile ---
    p_oot = sp / "sglang_tpu/multimodal_gen/platform.py"
    if p_oot.exists():
        t_oot = p_oot.read_text()
        old_dyn = "        torch._dynamo.config.automatic_dynamic_shapes = False"
        new_dyn = (
            "        torch._dynamo.config.automatic_dynamic_shapes = False\n"
            "        torch._dynamo.config.cache_size_limit = 64\n"
            "        torch._dynamo.config.accumulated_cache_size_limit = 256"
        )
        if "cache_size_limit = 64" not in t_oot:
            assert old_dyn in t_oot, "Missing automatic_dynamic_shapes in sglang_tpu/multimodal_gen/platform.py"
            t_oot = t_oot.replace(old_dyn, new_dyn, 1)

        old_vae_sup = """    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        # Off for every VAE: the torch_tpu backend fails to compile the diffusers
        # AutoencoderKL decoder (torch_tpu 0.2.0rc4, libtpu 0.0.50: view() cannot
        # turn a channels-last [1, 512, 128, 128] activation into [1, 32, 262144]),
        # and the other VAEs are untested under it.
        return False"""
        new_vae_sup = """    @classmethod
    def supports_vae_torch_compile(cls) -> bool:
        import os

        return os.environ.get("SGLANG_TPU_COMPILE_VAE", "0") == "1\""""
        if "SGLANG_TPU_COMPILE_VAE" not in t_oot:
            assert old_vae_sup in t_oot, "Missing supports_vae_torch_compile in sglang_tpu/multimodal_gen/platform.py"
            t_oot = t_oot.replace(old_vae_sup, new_vae_sup, 1)
        p_oot.write_text(t_oot)

    # --- Opt 2B: DecodingStage.decode safe compiled VAE fallback (WanVAE & generic VAEs) ---
    p_dec_stage = rt / "pipelines_core/stages/decoding.py"
    t_dec_stage = p_dec_stage.read_text()
    old_dec_call = """                try:
                    decode_output = self._get_vae_decode_fn(vae, server_args)(latents)
                except Exception as error:"""
    new_dec_call = """                try:
                    try:
                        decode_output = self._get_vae_decode_fn(vae, server_args)(latents)
                    except Exception as _vae_compile_exc:
                        if "out of memory" in str(_vae_compile_exc).lower() or not server_args.enable_torch_compile:
                            raise
                        import os as _os

                        _os.environ["SGLANG_TPU_COMPILE_VAE"] = "0"
                        self._compiled_vae_decode = ActiveTargetCompiledCallable()
                        logger.warning(
                            "Compiled VAE decode failed (%s); falling back to eager vae.decode",
                            _vae_compile_exc,
                        )
                        decode_output = vae.decode(latents)
                except Exception as error:"""
    if "_vae_compile_exc" not in t_dec_stage:
        assert old_dec_call in t_dec_stage, "Missing decode_output target in stages/decoding.py"
        p_dec_stage.write_text(t_dec_stage.replace(old_dec_call, new_dec_call, 1))

    # --- Baseline 3: ltx_2_pipeline.py honor explicit 20/40-step schedules ---
    p_pipe = rt / "pipelines/ltx_2_pipeline.py"
    t_pipe = p_pipe.read_text()
    old_sig = "        pinned_sigmas = server_args.pipeline_config.default_sigmas\n        if pinned_sigmas:"
    new_sig = "        pinned_sigmas = server_args.pipeline_config.default_sigmas\n        if pinned_sigmas and int(batch.num_inference_steps) == len(pinned_sigmas):"
    if new_sig not in t_pipe:
        assert old_sig in t_pipe, "Missing pinned_sigmas target in ltx_2_pipeline.py"
        p_pipe.write_text(t_pipe.replace(old_sig, new_sig, 1))

    # --- Baseline 4 + Opt 2C: ltx_2/decoding_av.py CPU Generator on TPU + compiled VAE decode ---
    p_dec_av = rt / "pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py"
    t_dec_av = p_dec_av.read_text()
    old_gen = "        generator = torch.Generator(device=latents.device).manual_seed(int(batch.seed))"
    new_gen = '        generator = torch.Generator(device="cpu" if latents.device.type == "tpu" else latents.device).manual_seed(int(batch.seed))'
    if new_gen not in t_dec_av:
        assert old_gen in t_dec_av, "Missing Generator target in ltx_2/decoding_av.py"
        t_dec_av = t_dec_av.replace(old_gen, new_gen, 1)

    old_ltx_vae = """                    with temporary_module_dtype(
                        self.vae, vae_dtype, enabled=should_cast_vae
                    ) as vae:
                        decode_output = vae.decode(latents)"""
    new_ltx_vae = """                    with temporary_module_dtype(
                        self.vae, vae_dtype, enabled=should_cast_vae
                    ) as vae:
                        try:
                            decode_output = self._get_vae_decode_fn(vae, server_args)(latents)
                        except Exception as _ltx_vae_exc:
                            if "out of memory" in str(_ltx_vae_exc).lower() or not server_args.enable_torch_compile:
                                raise
                            import os as _os

                            _os.environ["SGLANG_TPU_COMPILE_VAE"] = "0"
                            self._compiled_vae_decode = type(self._compiled_vae_decode)()
                            logger.warning(
                                "Compiled LTX-2.5 VAE decode failed (%s); falling back to eager vae.decode",
                                _ltx_vae_exc,
                            )
                            decode_output = vae.decode(latents)"""
    if "_ltx_vae_exc" not in t_dec_av:
        assert old_ltx_vae in t_dec_av, "Missing vae.decode(latents) target in ltx_2/decoding_av.py"
        t_dec_av = t_dec_av.replace(old_ltx_vae, new_ltx_vae, 1)
    p_dec_av.write_text(t_dec_av)

    # --- Baseline 5: ltx_2_5_diffusion_decoder.py CPU torch.randn on TPU ---
    p_ddec = rt / "models/decoders/ltx_2_5_diffusion_decoder.py"
    if p_ddec.exists():
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
        if new_dd1 not in t_ddec and old_dd1 in t_ddec:
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
        if new_dd2 not in t_ddec and old_dd2 in t_ddec:
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
        if new_dd3 not in t_ddec and old_dd3 in t_ddec:
            t_ddec = t_ddec.replace(old_dd3, new_dd3, 1)
        p_ddec.write_text(t_ddec)

    # --- Baseline 6: ltx_2/denoising_av.py CPU stage2 renoise generator on TPU ---
    p_den_av = rt / "pipelines_core/stages/model_specific_stages/ltx_2/denoising_av.py"
    if p_den_av.exists():
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
            p_den_av.write_text(t_den_av.replace(old_den, new_den, 1))

    # --- Baseline 7: hf_diffusers_utils.py local model directory fast-path ---
    p_hfu = rt / "utils/hf_diffusers_utils.py"
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
        new3 = 'def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:\n    return True, ""'
        if new3 not in t_hfu and old3 in t_hfu:
            t_hfu = t_hfu.replace(old3, new3, 1)
        old4 = "    if os.path.isdir(model_path):\n        if force_diffusers_model:"
        new4 = "    if os.path.isdir(model_path):\n        return model_path\n        if force_diffusers_model:"
        if new4 not in t_hfu and old4 in t_hfu:
            t_hfu = t_hfu.replace(old4, new4, 1)
        p_hfu.write_text(t_hfu)

    # --- Baseline 8..13: LTX-2.5 registry, loader, fsdp, duration, connectors ---
    p_reg = rt / "models/registry.py"
    if p_reg.exists():
        t_reg = p_reg.read_text()
        old_reg = "_ALIAS_TO_MODEL: dict[str, str] = {}"
        new_reg = (
            "_ALIAS_TO_MODEL: dict[str, str] = {\n"
            '    "LTXVideoTransformer3DModel": "LTX2VideoTransformer3DModel",\n'
            '    "LTXVideoTransformer3D": "LTX2VideoTransformer3DModel",\n'
            "}"
        )
        if old_reg in t_reg and "LTXVideoTransformer3DModel" not in t_reg:
            p_reg.write_text(t_reg.replace(old_reg, new_reg, 1))

    p_tl = rt / "loader/component_loaders/transformer_loader.py"
    if p_tl.exists():
        t_tl = p_tl.read_text()
        old_tl = '        if requested_distributed_execution:\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        new_tl = '        if requested_distributed_execution:\n            return\n            raise RuntimeError(\n                f"Native Diffusers fallback for transformer component "'
        if old_tl in t_tl and new_tl not in t_tl:
            p_tl.write_text(t_tl.replace(old_tl, new_tl, 1))

    p_fsdp = rt / "loader/fsdp_load.py"
    if p_fsdp.exists():
        t_fsdp = p_fsdp.read_text()
        old_fsdp = "    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [\n"
        new_fsdp = '    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [\n        "",\n'
        if old_fsdp in t_fsdp and '        "",\n' not in t_fsdp:
            p_fsdp.write_text(t_fsdp.replace(old_fsdp, new_fsdp, 1))

    p_dur = rt / "pipelines_core/stages/model_specific_stages/ltx_2/duration.py"
    if p_dur.exists():
        t_dur = p_dur.read_text()
        old_dur = "        if not batch.auto_duration:\n"
        new_dur = "        if not getattr(batch, 'auto_duration', False):\n"
        if old_dur in t_dur and new_dur not in t_dur:
            p_dur.write_text(t_dur.replace(old_dur, new_dur, 1))

    p_loc = rt / "managers/memory_managers/layerwise_offload_components.py"
    if p_loc.exists():
        t_loc = p_loc.read_text()
        old_loc = '    return component_name.startswith("text_encoder") or component_name.endswith(\n        "text_encoder"\n    )'
        new_loc = '    return component_name.startswith("text_encoder") or component_name.endswith(\n        "text_encoder"\n    ) or component_name in ("duration_head",)'
        if old_loc in t_loc and new_loc not in t_loc:
            p_loc.write_text(t_loc.replace(old_loc, new_loc, 1))

    p_tc = rt / "pipelines_core/stages/model_specific_stages/ltx_2/text_connector.py"
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

            _tpu = torch.device("tpu")
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

            _tpu = torch.device("tpu")
            batch.prompt_embeds = [connector_prompt_embeds.to(_tpu)]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds.to(_tpu)]
            batch.prompt_attention_mask = connector_mask.to(_tpu) if connector_mask is not None else None

        return batch
'''
        p_tc.write_text(clean_tc)

    # --- Baseline 14: Wan 2.2 parallel_conv halo bypass & WanVAE use_parallel_encode=False ---
    p_pc = rt / "layers/parallel_conv.py"
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

    p_wvc = sp / "sglang/multimodal_gen/configs/models/vaes/wanvae.py"
    if p_wvc.exists():
        t_wvc = p_wvc.read_text()
        if "use_parallel_encode: bool = True" in t_wvc:
            p_wvc.write_text(t_wvc.replace("use_parallel_encode: bool = True", "use_parallel_encode: bool = False", 1))

    p_wvr = rt / "models/vaes/wanvae.py"
    if p_wvr.exists():
        t_wvr = p_wvr.read_text()
        old_wvr = 'self.use_parallel_encode = getattr(config, "use_parallel_encode", False)'
        new_wvr = "self.use_parallel_encode = False"
        if old_wvr in t_wvr:
            p_wvr.write_text(t_wvr.replace(old_wvr, new_wvr, 1))

    # --- Opt 4: entrypoints/utils.py Disaggregated Remote MP4 Video+Audio Offload ---
    p_ep_utils = rt / "entrypoints/utils.py"
    t_ep = p_ep_utils.read_text()
    if "_try_save_remote_video_direct" not in t_ep:
        remote_fn_code = '''

def _try_save_remote_video_direct(
    *,
    save_file_path: str,
    sample: Any,
    fps: int,
    audio_sample_rate: Optional[int],
    x264_preset: Optional[str] = None,
) -> bool:
    """Stream raw uint8 RGB24 frames + optional f32le audio to disaggregated CPU video-encode-svc."""
    remote_url = os.environ.get("SGLANG_REMOTE_ENCODE_URL", "").strip()
    if not remote_url or os.path.splitext(save_file_path)[1].lower() != ".mp4":
        return False
    try:
        import http.client
        import struct
        import urllib.parse

        raw_video, sample_audio = _split_sample_audio(sample)
        if isinstance(raw_video, np.ndarray):
            vid_t = torch.from_numpy(raw_video)
        elif isinstance(raw_video, torch.Tensor):
            vid_t = raw_video.detach()
        else:
            return False

        if vid_t.ndim == 3 and int(vid_t.shape[0]) == 3:
            vid_t = vid_t.unsqueeze(1)
        if vid_t.ndim != 4:
            return False

        if int(vid_t.shape[-1]) == 3:
            vid_thwc = vid_t
        elif int(vid_t.shape[0]) == 3:
            vid_thwc = vid_t.permute(1, 2, 3, 0)
        elif int(vid_t.shape[1]) == 3:
            vid_thwc = vid_t.permute(0, 2, 3, 1)
        else:
            return False

        num_frames = int(vid_thwc.shape[0])
        height = int(vid_thwc.shape[1])
        width = int(vid_thwc.shape[2])
        if num_frames <= 0 or height <= 0 or width <= 0:
            return False

        if (
            vid_thwc.device.type == "cpu"
            and vid_thwc.dtype == torch.uint8
            and vid_thwc.is_contiguous()
        ):
            rgb_np = vid_thwc.numpy()
        else:
            rgb_np = np.empty((num_frames, height, width, 3), dtype=np.uint8)
            out_u8 = torch.from_numpy(rgb_np)
            chunk_f = 32
            for s in range(0, num_frames, chunk_f):
                e = min(s + chunk_f, num_frames)
                sl = vid_thwc[s:e]
                if sl.device.type != "cpu":
                    sl = sl.cpu()
                if sl.dtype == torch.uint8:
                    out_u8[s:e].copy_(sl)
                else:
                    scaled = (sl.to(torch.float32) * 255.0).clamp_(0.0, 255.0).to(torch.uint8)
                    out_u8[s:e].copy_(scaled)

        pcm_bytes = b""
        selected_sr = 0
        audio_channels = 2
        audio_np = _normalize_audio_to_numpy(sample_audio)
        if audio_np is not None and audio_np.size > 0:
            selected_sr = int(
                _pick_audio_sample_rate(
                    audio_np=audio_np,
                    audio_sample_rate=audio_sample_rate,
                    fps=fps,
                    num_frames=num_frames,
                )
            )
            if audio_np.ndim == 1:
                audio_channels = 1
                pcm_bytes = np.ascontiguousarray(audio_np, dtype="<f4").tobytes()
            elif audio_np.ndim == 2:
                audio_channels = int(audio_np.shape[1])
                pcm_bytes = np.ascontiguousarray(audio_np, dtype="<f4").tobytes()

        meta = {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "pix_fmt": "rgb24",
            "preset": x264_preset or os.environ.get("SGLANG_REMOTE_ENCODE_PRESET", "ultrafast"),
            "crf": 20,
            "threads": 16,
            "audio_sr": selected_sr,
            "audio_channels": audio_channels,
            "audio_bytes": len(pcm_bytes),
            "return_mp4": True,
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        hdr_prefix = struct.pack(">I", len(meta_bytes)) + meta_bytes
        rgb_mv = memoryview(rgb_np).cast("B")
        total_len = len(hdr_prefix) + len(rgb_mv) + len(pcm_bytes)

        parsed = urllib.parse.urlsplit(remote_url)
        host = parsed.hostname or "video-encode-svc.default.svc.cluster.local"
        port = parsed.port or 8090
        path = parsed.path or "/encode"

        conn = http.client.HTTPConnection(host, port, timeout=300)
        try:
            conn.putrequest("POST", path)
            conn.putheader("Content-Type", "application/octet-stream")
            conn.putheader("Content-Length", str(total_len))
            conn.end_headers()
            conn.send(hdr_prefix)
            step_b = 64 * 1024 * 1024
            for off in range(0, len(rgb_mv), step_b):
                conn.send(rgb_mv[off : off + step_b])
            if pcm_bytes:
                conn.send(pcm_bytes)
            resp = conn.getresponse()
            mp4_bytes = resp.read()
            status = resp.status
            enc_host = resp.getheader("X-Encoder-Host", "unknown")
            enc_fps = resp.getheader("X-Encode-FPS", "?")
        finally:
            conn.close()

        if status != 200 or len(mp4_bytes) < 256:
            logger.warning(
                "Remote video-encode-svc returned status=%s bytes=%s; falling back to local encode",
                status,
                len(mp4_bytes),
            )
            return False

        with open(save_file_path, "wb") as f:
            f.write(mp4_bytes)
        logger.info(
            "Disaggregated MP4 saved via %s (%s fps, %.2f MB) -> %s",
            enc_host,
            enc_fps,
            len(mp4_bytes) / (1024.0 * 1024.0),
            save_file_path,
        )
        return True
    except Exception as exc:
        logger.warning("Remote video encode offload failed (%s); falling back to local save_outputs", exc)
        return False


def save_outputs('''
        old_save_def = "\n\ndef save_outputs("
        assert old_save_def in t_ep, "Missing def save_outputs in entrypoints/utils.py"
        t_ep = t_ep.replace(old_save_def, remote_fn_code, 1)

        old_save_hook = """                direct_saved = parallel_results is not None and parallel_results[idx]
                if not direct_saved:
                    direct_saved = _try_save_cuda_video_direct("""
        new_save_hook = """                direct_saved = parallel_results is not None and parallel_results[idx]
                if not direct_saved and os.environ.get("SGLANG_REMOTE_ENCODE_URL"):
                    direct_saved = _try_save_remote_video_direct(
                        save_file_path=save_file_path,
                        sample=sample,
                        fps=fps,
                        audio_sample_rate=audio_sample_rate,
                        x264_preset=x264_preset,
                    )
                if not direct_saved:
                    direct_saved = _try_save_cuda_video_direct("""
        assert old_save_hook in t_ep, "Missing direct_saved hook in entrypoints/utils.py"
        t_ep = t_ep.replace(old_save_hook, new_save_hook, 1)
        p_ep_utils.write_text(t_ep)

    # --- Opt 5: Sub-host worker_bootstrap.py & platforms/tpu.py LOCAL_RANK VFIO pinning ---
    wb_path = rt / "managers/worker_bootstrap.py"
    if wb_path.exists():
        wb_txt = wb_path.read_text()
        if "SGLANG_TPU_LOCAL_RANKS" not in wb_txt:
            anchor = "def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:\n"
            inject = (
                anchor
                + "    import os as _os\n"
                + "    _os.environ['RANK'] = str(spec.rank)\n"
                + "    if _os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                + "        _lrs = [x.strip() for x in _os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                + "        _os.environ['LOCAL_RANK'] = _lrs[spec.rank]\n"
            )
            assert anchor in wb_txt, "Missing bootstrap_scheduler_process in worker_bootstrap.py"
            wb_path.write_text(wb_txt.replace(anchor, inject, 1))

    tpu_plat_path = rt / "platforms/tpu.py"
    if tpu_plat_path.exists():
        tpu_txt = tpu_plat_path.read_text()
        if "SGLANG_TPU_LOCAL_RANKS" not in tpu_txt:
            anchor = "_prepare_tpu_env(world_size=_world_size(), resolve_single_host_topology=True)"
            inject = (
                "if os.environ.get('SGLANG_TPU_LOCAL_RANKS'):\n"
                "            _lrs = [x.strip() for x in os.environ['SGLANG_TPU_LOCAL_RANKS'].split(',') if x.strip()]\n"
                "            os.environ['LOCAL_RANK'] = _lrs[int(os.environ.get('RANK', '0'))]\n"
                "            for _k in ('TPU_CHIPS_PER_HOST_BOUNDS', 'TPU_HOST_BOUNDS', 'TPU_WORKER_ID', 'TPU_WORKER_HOSTNAMES', 'CLOUD_TPU_TASK_ID'):\n"
                "                os.environ.pop(_k, None)\n"
                "        " + anchor
            )
            assert anchor in tpu_txt, "Missing _prepare_tpu_env in platforms/tpu.py"
            tpu_plat_path.write_text(tpu_txt.replace(anchor, inject, 1))

    log(
        idx,
        "Applied all Suite 2 SGLang TPU v7x patches (Opt1 FP8-BF16 MXU + W8A8, Opt2 DiT+VAE torch.compile, Opt4 Disagg HTTP MP4/Audio Offload, Opt5 Sub-Host LOCAL_RANK Pinning)!",
    )


# ---------------------------------------------------------------------------
# Model Staging into /dev/shm & Synthetic Keyframes
# ---------------------------------------------------------------------------


def generate_i2v_keyframes(input_dir: Path, sizes: tuple[str, ...], prefix: str) -> dict[str, str]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for sz in sizes:
        w, h = (int(x) for x in sz.split("x"))
        p = input_dir / f"{prefix}_{sz}.png"
        img = Image.new("RGB", (w, h), (24, 44, 72))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(28 + 150 * (y / max(h, 1))))
            g = min(255, int(54 + 102 * (y / max(h, 1))))
            b = min(255, int(94 + 78 * (1.0 - y / max(h, 1))))
            d.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        d.ellipse([w // 3, h // 4, w // 3 + 124, h // 4 + 124], fill=(250, 210, 122))
        img.save(p, format="PNG")
        out[sz] = str(p)
    return out


def stage_ltx25_model(idx: int, bucket: str) -> str:
    for stale in (
        "/dev/shm/MiniMax-H3",
        "/dev/shm/FastH3",
        "/dev/shm/Wan-AI",
        "/dev/shm/Wan2.2-I2V-A14B-Diffusers",
        "/dev/shm/Wan2.2-T2V-A14B-Diffusers",
    ):
        shutil.rmtree(stale, ignore_errors=True)

    model_dir = Path("/dev/shm/LTX-2.5")
    n_blobs = fast_gcs_sync_dir(idx, bucket, "models/Lightricks/LTX-2.5-Diffusers/", model_dir)
    if n_blobs == 0:
        n_blobs = fast_gcs_sync_dir(idx, bucket, "models/Lightricks/LTX-2.5/", model_dir)
    if n_blobs == 0 and not (model_dir / "model_index.json").exists():
        log(idx, "GCS prefix empty; falling back to huggingface_hub.snapshot_download('Lightricks/LTX-2.5')...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Lightricks/LTX-2.5",
            local_dir=str(model_dir),
            ignore_patterns=["*.ckpt", "prompt_enhancer/*", "diffusion_decoder/*"],
        )

    idx_path = model_dir / "model_index.json"
    assert idx_path.exists(), f"Missing {idx_path}"
    midx = json.loads(idx_path.read_text())
    midx.pop("prompt_enhancer", None)
    midx.pop("diffusion_decoder", None)
    idx_path.write_text(json.dumps(midx, indent=2))

    tr_cfg = model_dir / "transformer" / "config.json"
    if tr_cfg.exists():
        cdata = json.loads(tr_cfg.read_text())
        cdata["_class_name"] = "LTX2VideoTransformer3DModel"
        cdata["apply_gated_attention"] = True
        cdata["cross_attention_adaln"] = True
        cdata["caption_proj_before_connector"] = True
        cdata["use_keyframes_abs_pos_embedding"] = True
        cdata["ff_bias"] = False
        cdata["audio_ff_bias"] = True
        tr_cfg.write_text(json.dumps(cdata, indent=2))

    for stray_idx in model_dir.glob("*/*.index.json"):
        try:
            stray_idx.unlink()
        except OSError:
            pass

    log(idx, f"Staged Lightricks/LTX-2.5 in {model_dir} (shm: {shm_df_summary()}, RAM: {host_ram_summary()})")
    return str(model_dir)


def stage_wan22_repo(idx: int, bucket: str, repo_id: str) -> str:
    """Strictly stage ONE Wan 2.2 A14B repo in /dev/shm at a time (~125 GiB)."""
    for stale in (
        "/dev/shm/MiniMax-H3",
        "/dev/shm/FastH3",
        "/dev/shm/LTX-2.5",
        "/dev/shm/LTX-2.5-Diffusers",
    ):
        shutil.rmtree(stale, ignore_errors=True)

    wan_root = Path("/dev/shm/Wan-AI")
    target_dir = Path("/dev/shm") / repo_id
    if wan_root.exists():
        for child in wan_root.iterdir():
            if child != target_dir:
                log(idx, f"Removing previous /dev/shm model {child} before staging {repo_id}...")
                shutil.rmtree(child, ignore_errors=True)

    n_blobs = fast_gcs_sync_dir(
        idx,
        bucket,
        f"models/{repo_id}/",
        target_dir,
        exclude_prefixes=("assets/", "examples/"),
    )
    if n_blobs == 0 and not (target_dir / "model_index.json").exists():
        log(idx, f"GCS prefix models/{repo_id}/ empty; falling back to snapshot_download('{repo_id}')...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            ignore_patterns=["assets/*", "examples/*"],
        )

    assert (target_dir / "model_index.json").exists(), f"Missing {target_dir / 'model_index.json'}"
    log(idx, f"Staged {repo_id} in {target_dir} (shm: {shm_df_summary()}, RAM: {host_ram_summary()})")
    return str(target_dir)


# ---------------------------------------------------------------------------
# Process & VFIO Cleanup + Sub-Host Multi-Replica Cluster Manager
# ---------------------------------------------------------------------------


def find_vfio_holder_pids() -> list[int]:
    my_pid = os.getpid()
    mock_pid = _MOCK_META_PROC.pid if _MOCK_META_PROC is not None else -1
    holders: set[int] = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in (my_pid, mock_pid, 1):
            continue
        fd_dir = p / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                    if "/dev/vfio/" in target:
                        holders.add(pid)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(holders)


def stop_cluster_cleanly(idx: int, procs: list[subprocess.Popen[Any]] | None = None) -> None:
    if procs:
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        t_wait = time.time()
        while any(p is not None and p.poll() is None for p in procs) and time.time() - t_wait < 6.0:
            time.sleep(0.3)
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    subprocess.run(
        [
            "bash",
            "-lc",
            "pkill -15 -f '[s]glang serve' 2>/dev/null || true; "
            "pkill -15 -f '[s]glang.multimodal_gen' 2>/dev/null || true; "
            "sleep 1.0; "
            "pkill -9 -f '[s]glang serve' 2>/dev/null || true; "
            "pkill -9 -f '[s]glang.multimodal_gen' 2>/dev/null || true; "
            "pkill -9 -f '[m]ultiprocessing.spawn' 2>/dev/null || true; "
            "pkill -9 -f '[m]ultiprocessing.resource_tracker' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/torch_tpu_pg_store_* 2>/dev/null || true; "
            "rm -rf /dev/shm/sglang_outputs_* 2>/dev/null || true",
        ],
        check=False,
    )

    t_vfio = time.time()
    while time.time() - t_vfio < 20.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        log(idx, f"Killing lingering VFIO holder PIDs: {holders}")
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)
    time.sleep(1.5)


def extract_clean_error_tail(log_path: Path, max_chars: int = 3500) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="ignore").splitlines()
    filtered = [
        ln
        for ln in lines
        if "coredump_hook.cc" not in ln
        and "Coroner client" not in ln
        and "remote_coredump.socket" not in ln
        and "process_state.cc:794" not in ln
    ]
    return "\n".join(filtered)[-max_chars:]


def parse_hbm_from_logs(log_paths: list[Path]) -> float | None:
    vals_gib: list[float] = []
    for lp in log_paths:
        if not lp.exists():
            continue
        try:
            txt = lp.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(
            r"(?:memory|hbm|allocated|used)[^\n]{0,40}?([0-9]+\.[0-9]+)\s*(GiB|GB|MiB|MB)",
            txt,
            re.IGNORECASE,
        ):
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit in ("MIB", "MB"):
                val /= 1024.0
            if 1.0 <= val <= 96.0:
                vals_gib.append(val)
    return round(max(vals_gib), 2) if vals_gib else None


def build_clean_tpu_base_env(
    *,
    fp8_bf16_fast: bool,
    compile_vae: bool,
    remote_encode_url: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    # Strip GKE 2x2x2 multi-host slice variables so each host & sub-host replica initializes cleanly
    for k in (
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "HOST_BOUNDS",
        "TPU_TOPOLOGY",
        "TOPOLOGY",
        "TPU_ACCELERATOR_TYPE",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "CLOUD_TPU_TASK_ID",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "LIBTPU_INIT_ARGS",
        "TPU_ML_PLATFORM",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "SGLANG_TPU_LOCAL_RANKS",
        "SGLANG_REMOTE_ENCODE_URL",
    ):
        env.pop(k, None)

    cache_root = Path("/dev/shm/sglang_xla_cache")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "INJECT_SLICE_BUILDER_FAULT": "",
            "GCE_METADATA_HOST": "127.0.0.1:80",
            "GCE_METADATA_IP": "127.0.0.1",
            "GOOGLE_CLOUD_PROJECT": "mock-tpu-project",
            "GCLOUD_PROJECT": "mock-tpu-project",
            "LD_PRELOAD": "/tmp/metadata_hook.so",
            "HF_HOME": "/dev/shm/hf_cache",
            "HF_HUB_CACHE": "/dev/shm/hf_cache/hub",
            "SGLANG_DIFFUSION_CACHE_ROOT": "/dev/shm/sglang_diffusion_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "JAX_COMPILATION_CACHE_DIR": str(cache_root / "xla"),
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_TPU_SPLASH_VARLEN": "0",
            "SGLANG_TPU_DISABLE_SPLASH": "0",
            "SGLANG_TPU_FP8_BF16_FAST": "1" if fp8_bf16_fast else "0",
            "SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM": "1" if fp8_bf16_fast else "0",
            "SGLANG_TPU_COMPILE_VAE": "1" if compile_vae else "0",
        }
    )
    if remote_encode_url:
        env["SGLANG_REMOTE_ENCODE_URL"] = remote_encode_url
    return env


def _launch_one_sglang_instance(
    idx: int,
    *,
    model_path: str,
    port: int,
    scheduler_port: int,
    master_port: int,
    num_gpus: int,
    dp_size: int,
    ulysses_degree: int,
    quantization: str | None,
    enable_compile: bool,
    all_hbm_resident: bool,
    base_env: dict[str, str],
    extra_env: dict[str, str],
    log_path: Path,
    boot_timeout_s: float = 900.0,
) -> tuple[subprocess.Popen[Any], float]:
    out_dir = Path(f"/dev/shm/sglang_outputs_{port}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env.update(extra_env)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)

    offload_str = "false" if all_hbm_resident else "true"
    cmd = [
        "sglang",
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--scheduler-port",
        str(scheduler_port),
        "--master-port",
        str(master_port),
        "--num-gpus",
        str(num_gpus),
        "--dp-size",
        str(dp_size),
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        "tpu_splash_attn",
        "--enable-torch-compile",
        "true" if enable_compile else "false",
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

    lrs_tag = env.get("SGLANG_TPU_LOCAL_RANKS", "all-8")
    topo_tag = env.get("TORCH_TPU_TOPOLOGY", "auto")
    log(
        idx,
        f"Launching SGLang :{port} (local_ranks={lrs_tag}, topo={topo_tag}, compile={enable_compile}, "
        f"fp8_fast={env.get('SGLANG_TPU_FP8_BF16_FAST')}, disagg={bool(env.get('SGLANG_REMOTE_ENCODE_URL'))}): {' '.join(cmd)}",
    )
    lf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    urls = (f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/videos")
    while time.time() - t0 < boot_timeout_s:
        if proc.poll() is not None:
            lf.flush()
            tail = extract_clean_error_tail(log_path)
            raise RuntimeError(f"sglang serve :{port} exited early (rc={proc.returncode}):\n{tail}")
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=4.0) as r:
                    if r.status == 200:
                        boot_s = round(time.time() - t0, 2)
                        log(idx, f"SGLang READY on :{port} in {boot_s}s (local_ranks={lrs_tag}, RAM={host_ram_summary()})")
                        return proc, boot_s
            except Exception:
                pass
        time.sleep(2.5)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    tail = extract_clean_error_tail(log_path)
    raise TimeoutError(f"Timed out waiting for SGLang server on :{port}:\n{tail}")


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
            "error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:800]}",
        }
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)[:800]}

    vid_id = job["id"]
    while time.perf_counter() - t0 < timeout_s:
        time.sleep(0.4)
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
                st = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
        status = st.get("status")
        if status == "completed":
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
            valid = fsize is None or fsize > 256
            return {
                "ok": valid,
                "e2e_s": round(e2e, 3),
                "inference_s": st.get("inference_time_s"),
                "peak_memory_mb": st.get("peak_memory_mb"),
                "file_bytes": fsize,
                "ffprobe_streams": ffprobe_streams,
                "error": None if valid else f"MP4 too small ({fsize} bytes)",
            }
        if status == "failed":
            return {
                "ok": False,
                "e2e_s": round(time.perf_counter() - t0, 3),
                "error": str(st.get("error"))[:800],
            }
    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}


class SGLangVideoCluster:
    """Manages 1..4 isolated SGLang video servers on the 8 v7x TensorCores with least-inflight load balancing."""

    def __init__(
        self,
        idx: int,
        procs: list[subprocess.Popen[Any]],
        endpoints: list[str],
        log_paths: list[Path],
        boot_s: float,
        mode: str,
        lockstep_dp: int = 1,
    ) -> None:
        self.idx = idx
        self.procs = procs
        self.endpoints = endpoints
        self.log_paths = log_paths
        self.boot_s = boot_s
        self.mode = mode
        self.lockstep_dp = lockstep_dp
        self._lock = threading.Lock()
        self._inflight = [0] * len(endpoints)
        self._rr = 0

    def _acquire_endpoint(self) -> tuple[int, str]:
        with self._lock:
            n = len(self.endpoints)
            best_i = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[best_i] += 1
            self._rr = (best_i + 1) % n
            return best_i, self.endpoints[best_i]

    def _release_endpoint(self, ep_i: int) -> None:
        with self._lock:
            self._inflight[ep_i] = max(0, self._inflight[ep_i] - 1)

    def call_single(self, payload: dict[str, Any], *, probe_streams: bool = False) -> dict[str, Any]:
        if self.lockstep_dp > 1:
            ep = self.endpoints[0]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.lockstep_dp) as pool:
                f_main = pool.submit(call_sglang_video, ep, payload, 2400.0, probe_streams)
                helpers = []
                for j in range(1, self.lockstep_dp):
                    hp = dict(payload)
                    hp["seed"] = int(payload.get("seed", 100)) + 10000 + j
                    helpers.append(pool.submit(call_sglang_video, ep, hp, 2400.0, False))
                res = f_main.result()
                for fh in helpers:
                    try:
                        fh.result()
                    except Exception:
                        pass
                return res

        ep_i, ep = self._acquire_endpoint()
        try:
            return call_sglang_video(ep, payload, probe_streams=probe_streams)
        finally:
            self._release_endpoint(ep_i)

    def warm_all_replicas(self, base_payload: dict[str, Any], *, probe_streams: bool = False) -> dict[str, Any]:
        """Warm every sub-host replica concurrently so all replicas hold compiled XLA graphs before timed waves."""
        if len(self.endpoints) == 1:
            return self.call_single(base_payload, probe_streams=probe_streams)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.endpoints)) as pool:
            futs = []
            for r_i, ep in enumerate(self.endpoints):
                p_r = dict(base_payload)
                p_r["seed"] = int(base_payload.get("seed", 100)) + r_i
                futs.append(pool.submit(call_sglang_video, ep, p_r, 2400.0, (probe_streams and r_i == 0)))
            results = [f.result() for f in futs]
        failed = [r for r in results if not r.get("ok")]
        if failed:
            return failed[0]
        res0 = dict(results[0])
        res0["e2e_s"] = round(max(r["e2e_s"] for r in results), 3)
        return res0

    def run_wave(self, conc: int, payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        if self.lockstep_dp > 1 and conc > 1:
            rem = len(payloads) % self.lockstep_dp
            eff_payloads = list(payloads)
            for k in range(0 if rem == 0 else (self.lockstep_dp - rem)):
                pad_p = dict(payloads[k % len(payloads)])
                pad_p["seed"] = int(pad_p.get("seed", 1000)) + 50000 + k
                eff_payloads.append(pad_p)
            eff_conc = max(self.lockstep_dp, (conc // self.lockstep_dp) * self.lockstep_dp)
            ep = self.endpoints[0]
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=eff_conc) as pool:
                outs = list(pool.map(lambda p: call_sglang_video(ep, p), eff_payloads))
            return outs[: len(payloads)], time.perf_counter() - t0

        t0 = time.perf_counter()
        if conc <= 1:
            outs = [self.call_single(p) for p in payloads]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(self.call_single, payloads))
        return outs, time.perf_counter() - t0

    def stop(self) -> None:
        stop_cluster_cleanly(self.idx, self.procs)


def start_sglang_video_cluster(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    dp_size: int,
    ulysses_degree: int,
    quantization: str | None,
    fp8_bf16_fast: bool,
    enable_compile: bool,
    compile_vae: bool,
    all_hbm_resident: bool,
    remote_encode_url: str | None,
) -> SGLangVideoCluster:
    global _PORT_SEQ
    stop_cluster_cleanly(idx, None)
    _PORT_SEQ += 1
    seq_off = (_PORT_SEQ % 20) * 100

    base_port = 30100 + (seq_off // 10)
    base_env = build_clean_tpu_base_env(
        fp8_bf16_fast=fp8_bf16_fast,
        compile_vae=compile_vae,
        remote_encode_url=remote_encode_url,
    )
    log_dir = Path("/workspace/suite2_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    t_cluster0 = time.time()

    # Case 1: DP=4 x Ulysses=2 -> 4 isolated 1-chip (2-TensorCore) servers on [0,1], [2,3], [4,5], [6,7]
    if dp_size == 4 and ulysses_degree == 2:
        for use_vis_devs in (False, True):
            procs: list[subprocess.Popen[Any]] = []
            endpoints: list[str] = []
            log_paths: list[Path] = []
            try:
                for r in range(4):
                    p_http = base_port + r * 20
                    p_sched = 6400 + seq_off + r * 20
                    p_master = 31400 + seq_off + r * 20
                    sb0 = 28400 + seq_off + r * 10
                    tmp_r = Path(f"/tmp/sgl_suite2_u2_rep{r}")
                    tmp_r.mkdir(parents=True, exist_ok=True)
                    lp = log_dir / f"{config_name}_rep{r}.log"
                    extra_env = {
                        "SGLANG_TPU_LOCAL_RANKS": "0,1" if use_vis_devs else f"{2 * r},{2 * r + 1}",
                        "TORCH_TPU_TOPOLOGY": "1,1,1,2",
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES": f"localhost:{sb0},localhost:{sb0 + 1}",
                        "TPU_RUNTIME_METRICS_PORTS": f"{8431 + 2 * r},{8432 + 2 * r}",
                        "TMPDIR": str(tmp_r),
                    }
                    if use_vis_devs:
                        extra_env["TPU_VISIBLE_DEVICES"] = str(r)
                    proc, _ = _launch_one_sglang_instance(
                        idx,
                        model_path=model_path,
                        port=p_http,
                        scheduler_port=p_sched,
                        master_port=p_master,
                        num_gpus=2,
                        dp_size=1,
                        ulysses_degree=2,
                        quantization=quantization,
                        enable_compile=enable_compile,
                        all_hbm_resident=all_hbm_resident,
                        base_env=base_env,
                        extra_env=extra_env,
                        log_path=lp,
                        boot_timeout_s=720.0,
                    )
                    procs.append(proc)
                    endpoints.append(f"http://127.0.0.1:{p_http}")
                    log_paths.append(lp)
                total_boot = round(time.time() - t_cluster0, 2)
                mode = "subhost_4x_u2_visdev" if use_vis_devs else "subhost_4x_u2"
                log(idx, f"[{config_name}] DP=4 x U=2 cluster ready in {total_boot}s ({mode}): {endpoints}")
                return SGLangVideoCluster(idx, procs, endpoints, log_paths, total_boot, mode=mode, lockstep_dp=1)
            except Exception as sub_exc:
                log(idx, f"[{config_name}] DP=4 x U=2 attempt (vis_dev={use_vis_devs}) failed: {str(sub_exc)[:260]}")
                stop_cluster_cleanly(idx, procs)
                if use_vis_devs:
                    raise

    # Case 2: DP=2 x Ulysses=4 -> 2 isolated 2-chip (4-TensorCore) servers on [0..3], [4..7] with graceful fallbacks
    if dp_size == 2 and ulysses_degree == 4:
        attempts: list[tuple[str, bool]] = [
            ("1,2,1,2", False),
            ("2,1,1,2", False),
            ("1,2,1,2", True),
        ]
        for topo_try, use_vis_devs in attempts:
            procs = []
            endpoints = []
            log_paths = []
            try:
                for r, lrs in enumerate(("0,1,2,3", "4,5,6,7")):
                    p_http = base_port + r * 20
                    p_sched = 6800 + seq_off + r * 20
                    p_master = 31800 + seq_off + r * 20
                    sb0 = 28800 + seq_off + r * 10
                    tmp_r = Path(f"/tmp/sgl_suite2_u4_rep{r}")
                    tmp_r.mkdir(parents=True, exist_ok=True)
                    lp = log_dir / f"{config_name}_rep{r}.log"
                    extra_env = {
                        "SGLANG_TPU_LOCAL_RANKS": "0,1,2,3" if use_vis_devs else lrs,
                        "TORCH_TPU_TOPOLOGY": topo_try,
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{sb0 + k}" for k in range(4)),
                        "TPU_RUNTIME_METRICS_PORTS": ",".join(str(8431 + 4 * r + k) for k in range(4)),
                        "TMPDIR": str(tmp_r),
                    }
                    if use_vis_devs:
                        extra_env["TPU_VISIBLE_DEVICES"] = "0,1" if r == 0 else "2,3"
                    proc, _ = _launch_one_sglang_instance(
                        idx,
                        model_path=model_path,
                        port=p_http,
                        scheduler_port=p_sched,
                        master_port=p_master,
                        num_gpus=4,
                        dp_size=1,
                        ulysses_degree=4,
                        quantization=quantization,
                        enable_compile=enable_compile,
                        all_hbm_resident=all_hbm_resident,
                        base_env=base_env,
                        extra_env=extra_env,
                        log_path=lp,
                        boot_timeout_s=360.0,
                    )
                    procs.append(proc)
                    endpoints.append(f"http://127.0.0.1:{p_http}")
                    log_paths.append(lp)
                total_boot = round(time.time() - t_cluster0, 2)
                mode = f"subhost_2x_u4_{topo_try}{'_visdev' if use_vis_devs else ''}"
                log(idx, f"[{config_name}] DP=2 x U=4 cluster ready in {total_boot}s ({mode}): {endpoints}")
                return SGLangVideoCluster(idx, procs, endpoints, log_paths, total_boot, mode=mode, lockstep_dp=1)
            except Exception as sub_exc:
                log(idx, f"[{config_name}] Sub-host 2x U=4 ({topo_try}, vis={use_vis_devs}) did not bind: {str(sub_exc)[:240]}")
                stop_cluster_cleanly(idx, procs)

        # Fallback: single 8-TensorCore server with dp_size=2, ulysses_degree=4 lockstep wave pairing
        lp = log_dir / f"{config_name}_lockstep.log"
        sb0 = 29200 + seq_off
        tmp_r = Path("/tmp/sgl_suite2_u8")
        tmp_r.mkdir(parents=True, exist_ok=True)
        extra_env = {
            "SGLANG_TPU_LOCAL_RANKS": "0,1,2,3,4,5,6,7",
            "TORCH_TPU_TOPOLOGY": "2,2,1,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{sb0 + k}" for k in range(8)),
            "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438",
            "TMPDIR": str(tmp_r),
        }
        proc, boot_s = _launch_one_sglang_instance(
            idx,
            model_path=model_path,
            port=base_port,
            scheduler_port=7200 + seq_off,
            master_port=32200 + seq_off,
            num_gpus=8,
            dp_size=2,
            ulysses_degree=4,
            quantization=quantization,
            enable_compile=enable_compile,
            all_hbm_resident=all_hbm_resident,
            base_env=base_env,
            extra_env=extra_env,
            log_path=lp,
            boot_timeout_s=900.0,
        )
        return SGLangVideoCluster(
            idx,
            [proc],
            [f"http://127.0.0.1:{base_port}"],
            [lp],
            boot_s,
            mode="single_8core_lockstep_dp2_u4",
            lockstep_dp=2,
        )

    # Case 3: DP=1 x Ulysses=8 -> Single 8-TensorCore server across all 4 host v7x chips
    lp = log_dir / f"{config_name}.log"
    sb0 = 29500 + seq_off
    tmp_r = Path("/tmp/sgl_suite2_u8")
    tmp_r.mkdir(parents=True, exist_ok=True)
    extra_env = {
        "SGLANG_TPU_LOCAL_RANKS": "0,1,2,3,4,5,6,7",
        "TORCH_TPU_TOPOLOGY": "2,2,1,2",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{sb0 + k}" for k in range(8)),
        "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438",
        "TMPDIR": str(tmp_r),
    }
    proc, boot_s = _launch_one_sglang_instance(
        idx,
        model_path=model_path,
        port=base_port,
        scheduler_port=7500 + seq_off,
        master_port=32500 + seq_off,
        num_gpus=8,
        dp_size=dp_size,
        ulysses_degree=ulysses_degree,
        quantization=quantization,
        enable_compile=enable_compile,
        all_hbm_resident=all_hbm_resident,
        base_env=base_env,
        extra_env=extra_env,
        log_path=lp,
        boot_timeout_s=1200.0,
    )
    return SGLangVideoCluster(
        idx,
        [proc],
        [f"http://127.0.0.1:{base_port}"],
        [lp],
        boot_s,
        mode=f"single_8core_dp{dp_size}_u{ulysses_degree}",
        lockstep_dp=1,
    )


# ---------------------------------------------------------------------------
# Benchmark Execution Helpers (Pod 0 LTX-2.5 & Pod 1 Wan 2.2 A14B)
# ---------------------------------------------------------------------------


def check_remote_encoder_health(idx: int, encode_url: str) -> dict[str, Any] | None:
    try:
        parsed = urllib.parse.urlsplit(encode_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        with urllib.request.urlopen(health_url, timeout=5.0) as r:
            info = json.loads(r.read().decode("utf-8"))
        log(idx, f"Remote video-encode-svc HEALTHY at {health_url}: {info}")
        return info
    except Exception as exc:
        log(idx, f"WARNING: Remote video-encode-svc health check failed ({encode_url}): {exc}")
        return None


def run_config_workloads(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    local_jsonl: Path,
    local_summary: Path,
    pod_rows: list[dict[str, Any]],
    done_keys: set[tuple[str, str, int]],
    model_name: str,
    model_family: str,
    model_path: str,
    cfg: dict[str, Any],
    workloads: list[dict[str, Any]],
    refs: dict[str, str],
    encode_url: str,
) -> None:
    config_name = str(cfg["name"])
    pending = [
        wl
        for wl in workloads
        if (config_name, str(wl["label"]), int(wl.get("conc", 1))) not in done_keys
    ]
    if not pending:
        log(idx, f"[{config_name}] All {len(workloads)} workload rows already in checkpoint — skipping!")
        return

    cluster: SGLangVideoCluster | None = None
    try:
        cluster = start_sglang_video_cluster(
            idx,
            config_name=config_name,
            model_path=model_path,
            dp_size=int(cfg["dp_size"]),
            ulysses_degree=int(cfg["ulysses_degree"]),
            quantization=cfg.get("quantization"),
            fp8_bf16_fast=bool(cfg.get("fp8_bf16_fast", True)),
            enable_compile=bool(cfg.get("enable_compile", True)),
            compile_vae=bool(cfg.get("compile_vae", True)),
            all_hbm_resident=bool(cfg.get("all_hbm_resident", True)),
            remote_encode_url=encode_url if cfg.get("disagg_encode") else None,
        )

        probed_once = False
        warmed_shapes: dict[tuple[Any, ...], float] = {}

        for w_i, wl in enumerate(workloads):
            w_label = str(wl["label"])
            conc = int(wl.get("conc", 1))
            if (config_name, w_label, conc) in done_keys:
                continue

            sz = str(wl["size"])
            nf = int(wl["num_frames"])
            dur = float(wl["dur"])
            steps = int(wl["steps"])
            guidance = float(wl["guidance"])
            is_i2v = bool(wl.get("i2v", False))
            fps = 24 if model_family == "ltx25" else 16
            n_reqs = int(wl.get("n", max(2, conc)))
            is_guided = guidance > 1.0

            if model_family == "ltx25":
                task = "i2va" if is_i2v else "t2va"
                shape_key: tuple[Any, ...] = ("ltx25", sz, nf, steps == 8, is_guided, is_i2v)
                warm_steps = 8 if steps == 8 else steps
            else:
                task = "i2v" if is_i2v else "t2v"
                shape_key = ("wan22", sz, nf, is_guided, is_i2v)
                warm_steps = min(steps, 4)

            cold_s = warmed_shapes.get(shape_key)
            probed_streams = None

            if shape_key not in warmed_shapes:
                p_warm: dict[str, Any] = {
                    "prompt": PROMPTS[0],
                    "size": sz,
                    "num_frames": nf,
                    "fps": fps,
                    "num_inference_steps": warm_steps,
                    "guidance_scale": guidance,
                    "generator_device": "cpu",
                    "seed": 100 + w_i * 10,
                }
                if model_family == "ltx25":
                    p_warm["use_diffusion_decoder"] = False
                    p_warm["auto_duration"] = False
                else:
                    p_warm["guidance_scale_2"] = guidance
                if is_i2v:
                    p_warm["input_reference"] = refs[sz]

                log(
                    idx,
                    f"[{config_name}] Warming {len(cluster.endpoints)} replica(s) on shape={shape_key} ({w_label})...",
                )
                w_res = cluster.warm_all_replicas(p_warm, probe_streams=(not probed_once))
                cold_s = w_res.get("e2e_s")
                if w_res.get("ffprobe_streams"):
                    probed_streams = w_res["ffprobe_streams"]
                    probed_once = True
                log(
                    idx,
                    f"[{config_name}] Warmup {w_label}: ok={w_res['ok']} cold_e2e={cold_s}s err={str(w_res.get('error') or '')[:180]}",
                )
                if not w_res["ok"]:
                    err_row = _build_result_row(
                        idx=idx,
                        model_name=model_name,
                        cfg=cfg,
                        cluster=cluster,
                        w_label=w_label,
                        task=task,
                        sz=sz,
                        nf=nf,
                        fps=fps,
                        dur=dur,
                        steps=steps,
                        guidance=guidance,
                        conc=conc,
                        n_reqs=n_reqs,
                        cold_s=cold_s,
                        outs=[w_res],
                        wall_s=0.0,
                        probed_streams=probed_streams,
                    )
                    checkpoint_row(idx, bucket, gcs_prefix, local_jsonl, local_summary, pod_rows, err_row)
                    continue
                warmed_shapes[shape_key] = float(cold_s or 0.0)

            payloads: list[dict[str, Any]] = []
            for r_i in range(n_reqs):
                pr: dict[str, Any] = {
                    "prompt": PROMPTS[(w_i + r_i) % len(PROMPTS)],
                    "size": sz,
                    "num_frames": nf,
                    "fps": fps,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "generator_device": "cpu",
                    "seed": 2000 + w_i * 100 + r_i,
                }
                if model_family == "ltx25":
                    pr["use_diffusion_decoder"] = False
                    pr["auto_duration"] = False
                else:
                    pr["guidance_scale_2"] = guidance
                if is_i2v:
                    pr["input_reference"] = refs[sz]
                payloads.append(pr)

            log(idx, f"[{config_name}] Running timed wave {w_label} (c={conc}, n={n_reqs}, steps={steps})...")
            outs, wall_s = cluster.run_wave(conc, payloads)
            row = _build_result_row(
                idx=idx,
                model_name=model_name,
                cfg=cfg,
                cluster=cluster,
                w_label=w_label,
                task=task,
                sz=sz,
                nf=nf,
                fps=fps,
                dur=dur,
                steps=steps,
                guidance=guidance,
                conc=conc,
                n_reqs=n_reqs,
                cold_s=cold_s,
                outs=outs,
                wall_s=wall_s,
                probed_streams=probed_streams,
            )
            log(
                idx,
                f"[{config_name}] DONE {w_label} c={conc}: ok={row['ok']} med={row['warm_median_e2e_s']}s "
                f"inf={row['server_inference_median_s']}s save={row['post_vae_encode_save_s']}s "
                f"rpm={row['rpm']} vid_s/s={row['video_sec_per_wall_sec']} chips@30rpm={row['chips_for_30_rpm']}",
            )
            checkpoint_row(idx, bucket, gcs_prefix, local_jsonl, local_summary, pod_rows, row)
            if row["ok_count"] > 0:
                done_keys.add((config_name, w_label, conc))

    except Exception as exc:
        tb = traceback.format_exc()
        log(idx, f"ERROR in config [{config_name}]: {exc}\n{tb}")
        if cluster is not None:
            for lp in cluster.log_paths:
                if lp.exists():
                    gcs_upload_bytes(
                        idx,
                        bucket,
                        f"{gcs_prefix}/logs/pod{idx}_{lp.name}",
                        lp.read_bytes()[-250_000:],
                        content_type="text/plain",
                    )
        fail_row = {
            "suite": "suite2_sglang_opt",
            "pod_index": idx,
            "hostname": HOSTNAME,
            "engine": "sglang-torchtpu:v0.2.0",
            "model": model_name,
            "config": config_name,
            "cluster_mode": cluster.mode if cluster else "boot_failed",
            "precision": "FP8" if cfg.get("quantization") == "fp8" else "BF16",
            "fp8_bf16_fast": bool(cfg.get("fp8_bf16_fast", True)),
            "enable_torch_compile": bool(cfg.get("enable_compile", True)),
            "compile_vae": bool(cfg.get("compile_vae", True)),
            "all_hbm_resident": bool(cfg.get("all_hbm_resident", True)),
            "disagg_encode": bool(cfg.get("disagg_encode", False)),
            "dp_size": int(cfg["dp_size"]),
            "ulysses_degree": int(cfg["ulysses_degree"]),
            "workload": "CONFIG_BOOT_OR_RUNTIME_ERROR",
            "concurrency": 0,
            "ok": "0/0",
            "ok_count": 0,
            "rpm": 0.0,
            "error": f"{exc}",
            "timestamp": utc_now(),
        }
        checkpoint_row(idx, bucket, gcs_prefix, local_jsonl, local_summary, pod_rows, fail_row)
    finally:
        if cluster is not None:
            for lp in cluster.log_paths:
                if lp.exists():
                    gcs_upload_bytes(
                        idx,
                        bucket,
                        f"{gcs_prefix}/logs/pod{idx}_{lp.name}",
                        lp.read_bytes()[-200_000:],
                        content_type="text/plain",
                    )
            cluster.stop()
        else:
            stop_cluster_cleanly(idx, None)


def _build_result_row(
    *,
    idx: int,
    model_name: str,
    cfg: dict[str, Any],
    cluster: SGLangVideoCluster,
    w_label: str,
    task: str,
    sz: str,
    nf: int,
    fps: int,
    dur: float,
    steps: int,
    guidance: float,
    conc: int,
    n_reqs: int,
    cold_s: float | None,
    outs: list[dict[str, Any]],
    wall_s: float,
    probed_streams: Any,
) -> dict[str, Any]:
    oks = [o for o in outs if o.get("ok")]
    errs = [o.get("error") for o in outs if not o.get("ok") and o.get("error")]
    e2e_vals = [float(o["e2e_s"]) for o in oks]
    inf_vals = [float(o["inference_s"]) for o in oks if o.get("inference_s") is not None]
    mem_mb_vals = [float(o["peak_memory_mb"]) for o in oks if o.get("peak_memory_mb")]
    fbytes_vals = [int(o["file_bytes"]) for o in oks if o.get("file_bytes")]

    med_e2e = round(statistics.median(e2e_vals), 3) if e2e_vals else None
    mean_e2e = round(statistics.mean(e2e_vals), 3) if e2e_vals else None
    p95_e2e = round(percentile(e2e_vals, 95.0), 3) if e2e_vals else None
    med_inf = round(statistics.median(inf_vals), 3) if inf_vals else None
    save_overhead = (
        round(max(0.0, med_e2e - med_inf), 3)
        if (med_e2e is not None and med_inf is not None)
        else None
    )
    step_basis = med_inf if med_inf is not None else med_e2e
    ms_step = round((step_basis * 1000.0) / max(steps, 1), 1) if step_basis is not None else None
    rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
    vid_s_per_s = round((len(oks) * dur) / wall_s, 4) if oks and wall_s > 0 else 0.0
    fps_tput = round((len(oks) * nf) / wall_s, 2) if oks and wall_s > 0 else 0.0
    chips_30rpm = int(math.ceil(30.0 / rpm) * 4) if rpm > 0 else None

    peak_hbm_gib = (
        round(max(mem_mb_vals) / 1024.0, 2)
        if mem_mb_vals
        else parse_hbm_from_logs(cluster.log_paths)
    )

    return {
        "suite": "suite2_sglang_opt",
        "pod_index": idx,
        "hostname": HOSTNAME,
        "engine": "sglang-torchtpu:v0.2.0",
        "model": model_name,
        "config": str(cfg["name"]),
        "cluster_mode": cluster.mode,
        "precision": "FP8" if cfg.get("quantization") == "fp8" else "BF16",
        "fp8_bf16_fast": bool(cfg.get("fp8_bf16_fast", True)),
        "enable_torch_compile": bool(cfg.get("enable_compile", True)),
        "compile_vae": bool(cfg.get("compile_vae", True)),
        "all_hbm_resident": bool(cfg.get("all_hbm_resident", True)),
        "disagg_encode": bool(cfg.get("disagg_encode", False)),
        "dp_size": int(cfg["dp_size"]),
        "ulysses_degree": int(cfg["ulysses_degree"]),
        "num_replicas": len(cluster.endpoints),
        "workload": w_label,
        "task": task,
        "size": sz,
        "num_frames": nf,
        "fps": fps,
        "duration_s": dur,
        "steps": steps,
        "guidance_scale": guidance,
        "concurrency": conc,
        "num_requests": n_reqs,
        "ok": f"{len(oks)}/{len(outs)}",
        "ok_count": len(oks),
        "boot_s": cluster.boot_s,
        "cold_compile_s": cold_s,
        "warm_e2e_list_s": e2e_vals,
        "warm_median_e2e_s": med_e2e,
        "warm_mean_e2e_s": mean_e2e,
        "warm_p95_e2e_s": p95_e2e,
        "server_inference_median_s": med_inf,
        "post_vae_encode_save_s": save_overhead,
        "ms_per_step": ms_step,
        "wall_s": round(wall_s, 3),
        "rpm": rpm,
        "video_sec_per_wall_sec": vid_s_per_s,
        "frames_per_sec": fps_tput,
        "chips_for_30_rpm": chips_30rpm,
        "peak_hbm_gib": peak_hbm_gib,
        "sample_mp4_bytes": fbytes_vals[0] if fbytes_vals else None,
        "ffprobe_streams": probed_streams,
        "error": errs[0] if errs else None,
        "timestamp": utc_now(),
    }


# ---------------------------------------------------------------------------
# Pod 0 (`Lightricks/LTX-2.5`) & Pod 1 (`Wan 2.2 A14B` T2V -> I2V) Plans
# ---------------------------------------------------------------------------


def run_pod0_ltx25(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    encode_url: str,
    resume: bool,
) -> None:
    local_jsonl = Path(f"/workspace/suite2_pod{idx}.jsonl")
    local_summary = Path(f"/workspace/suite2_pod{idx}_summary.json")
    pod_rows, done_keys = load_existing_checkpoint(
        idx, bucket, f"{gcs_prefix}/suite2_pod{idx}.jsonl", local_jsonl, resume
    )

    model_path = stage_ltx25_model(idx, bucket)
    refs = generate_i2v_keyframes(
        Path("/dev/shm/suite2_ltx_inputs"),
        ("960x544", "1280x704", "1920x1088"),
        "ltx_ref",
    )

    # 1. Unoptimized Baseline (prior fp32 _scaled_mm + eager + local FFmpeg)
    w_unopt = [
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
    ]

    # 2. + FP8 bf16 MXU + torch.compile (DiT + VAE), still local FFmpeg (isolates compute speedup vs encode bottleneck)
    w_compiled_localenc = [
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_5s_121f_t2va", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
    ]

    # 3. + Disaggregated CPU video-encode-svc — FULL Input Distribution Matrix
    w_compiled_disagg_full = [
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "544p_5s_121f_i2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "i2v": True, "conc": 1, "n": 2},
        {"label": "544p_10s_241f_t2va", "size": "960x544", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "544p_20s_481f_t2va", "size": "960x544", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_i2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "i2v": True, "conc": 1, "n": 2},
        {"label": "720p_10s_241f_t2va", "size": "1280x704", "num_frames": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_20step_cfg3", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "conc": 1, "n": 2},
        {"label": "1080p_5s_121f_t2va", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 2, "n": 2},
    ]

    # 4. BF16 + torch.compile + Disagg (isolates true 2x FP8 MXU gain vs BF16 MXU)
    w_bf16_disagg = [
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "conc": 1, "n": 2},
    ]

    # 5. Sub-host DP=2 x Ulysses=4 (2 isolated 2-chip servers/host) across c = 1, 2, 4
    w_dp2_u4 = []
    for base_wl in (
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
        {"label": "1080p_5s_121f_t2va", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
        {"label": "1080p_20s_481f_kling_v4_proxy", "size": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "guidance": 1.0},
    ):
        for c in (1, 2, 4):
            w_dp2_u4.append({**base_wl, "conc": c, "n": max(2, c)})

    # 6. Sub-host DP=4 x Ulysses=2 (4 isolated 1-chip servers/host) across c = 1, 4, 8
    w_dp4_u2 = []
    for base_wl in (
        {"label": "544p_5s_121f_t2va", "size": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
        {"label": "720p_5s_121f_t2va", "size": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
        {"label": "1080p_5s_121f_t2va", "size": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "guidance": 1.0},
    ):
        for c in (1, 4, 8):
            w_dp4_u2.append({**base_wl, "conc": c, "n": max(2, c)})

    configs_and_workloads: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (
            {
                "name": "sglang_ltx25_fp8_u8_unopt_base",
                "quantization": "fp8",
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": False,
                "enable_compile": False,
                "compile_vae": False,
                "disagg_encode": False,
                "all_hbm_resident": True,
            },
            w_unopt,
        ),
        (
            {
                "name": "sglang_ltx25_fp8_u8_compiled_localenc",
                "quantization": "fp8",
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": False,
                "all_hbm_resident": True,
            },
            w_compiled_localenc,
        ),
        (
            {
                "name": "sglang_ltx25_fp8_u8_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_compiled_disagg_full,
        ),
        (
            {
                "name": "sglang_ltx25_bf16_u8_compiled_disagg",
                "quantization": None,
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_bf16_disagg,
        ),
        (
            {
                "name": "sglang_ltx25_fp8_dp2_u4_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 2,
                "ulysses_degree": 4,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_dp2_u4,
        ),
        (
            {
                "name": "sglang_ltx25_fp8_dp4_u2_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 4,
                "ulysses_degree": 2,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_dp4_u2,
        ),
    ]

    for cfg, wls in configs_and_workloads:
        run_config_workloads(
            idx,
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            local_jsonl=local_jsonl,
            local_summary=local_summary,
            pod_rows=pod_rows,
            done_keys=done_keys,
            model_name="Lightricks/LTX-2.5",
            model_family="ltx25",
            model_path=model_path,
            cfg=cfg,
            workloads=wls,
            refs=refs,
            encode_url=encode_url,
        )


def _wan_base_workloads(*, is_i2v: bool) -> list[dict[str, Any]]:
    return [
        {"label": "480p_5s_81f_4step", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "i2v": is_i2v},
        {"label": "480p_5s_81f_20step", "size": "832x480", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "i2v": is_i2v},
        {"label": "720p_5s_81f_4step", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 4, "guidance": 1.0, "i2v": is_i2v},
        {"label": "720p_5s_81f_20step", "size": "1280x720", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "i2v": is_i2v},
        {"label": "720p_10s_161f_20step", "size": "1280x720", "num_frames": 161, "dur": 10.0, "steps": 20, "guidance": 3.5, "i2v": is_i2v},
        {"label": "720p_portrait_5s_81f_20step", "size": "720x1280", "num_frames": 81, "dur": 5.0, "steps": 20, "guidance": 3.5, "i2v": is_i2v},
    ]


def run_pod1_wan22(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    encode_url: str,
    resume: bool,
) -> None:
    local_jsonl = Path(f"/workspace/suite2_pod{idx}.jsonl")
    local_summary = Path(f"/workspace/suite2_pod{idx}_summary.json")
    pod_rows, done_keys = load_existing_checkpoint(
        idx, bucket, f"{gcs_prefix}/suite2_pod{idx}.jsonl", local_jsonl, resume
    )

    refs = generate_i2v_keyframes(
        Path("/dev/shm/suite2_wan_inputs"),
        ("832x480", "1280x720", "720x1280"),
        "wan_ref",
    )

    # --- Phase A: Wan-AI/Wan2.2-T2V-A14B-Diffusers ---
    t2v_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    t2v_base = _wan_base_workloads(is_i2v=False)

    w_t2v_u8 = [{**w, "conc": 1, "n": 2} for w in t2v_base]
    w_t2v_dp2 = [
        {**w, "conc": c, "n": max(2, c)}
        for w in t2v_base
        for c in (1, 2, 4)
    ]
    w_t2v_dp4 = [
        {**w, "conc": c, "n": max(2, c)}
        for w in t2v_base[:4]  # 480p & 720p 5s (4-step & 20-step) across c=1,4,8
        for c in (1, 4, 8)
    ] + [
        {**w, "conc": c, "n": max(2, c)}
        for w in t2v_base[4:]  # 720p 10s 161f & 720p Portrait across c=1,4
        for c in (1, 4)
    ]

    t2v_configs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (
            {
                "name": "sglang_wan22_t2v_fp8_u8_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_t2v_u8,
        ),
        (
            {
                "name": "sglang_wan22_t2v_bf16_u8_compiled_disagg",
                "quantization": None,
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_t2v_u8,
        ),
        (
            {
                "name": "sglang_wan22_t2v_fp8_dp2_u4_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 2,
                "ulysses_degree": 4,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_t2v_dp2,
        ),
        (
            {
                "name": "sglang_wan22_t2v_fp8_dp4_u2_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 4,
                "ulysses_degree": 2,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_t2v_dp4,
        ),
    ]

    t2v_needed = any(
        (str(cfg["name"]), str(w["label"]), int(w["conc"])) not in done_keys
        for cfg, wls in t2v_configs
        for w in wls
    )
    if t2v_needed:
        t2v_path = stage_wan22_repo(idx, bucket, t2v_repo)
        for cfg, wls in t2v_configs:
            run_config_workloads(
                idx,
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                local_summary=local_summary,
                pod_rows=pod_rows,
                done_keys=done_keys,
                model_name=t2v_repo,
                model_family="wan22",
                model_path=t2v_path,
                cfg=cfg,
                workloads=wls,
                refs=refs,
                encode_url=encode_url,
            )

    # --- Phase B: Clean /dev/shm Swap -> Wan-AI/Wan2.2-I2V-A14B-Diffusers ---
    i2v_repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    i2v_base = _wan_base_workloads(is_i2v=True)

    w_i2v_u8 = [{**w, "conc": 1, "n": 2} for w in i2v_base]
    w_i2v_dp2 = [
        {**w, "conc": c, "n": max(2, c)}
        for w in i2v_base
        for c in (1, 2, 4)
    ]

    i2v_configs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (
            {
                "name": "sglang_wan22_i2v_fp8_u8_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_i2v_u8,
        ),
        (
            {
                "name": "sglang_wan22_i2v_bf16_u8_compiled_disagg",
                "quantization": None,
                "dp_size": 1,
                "ulysses_degree": 8,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_i2v_u8,
        ),
        (
            {
                "name": "sglang_wan22_i2v_fp8_dp2_u4_compiled_disagg",
                "quantization": "fp8",
                "dp_size": 2,
                "ulysses_degree": 4,
                "fp8_bf16_fast": True,
                "enable_compile": True,
                "compile_vae": True,
                "disagg_encode": True,
                "all_hbm_resident": True,
            },
            w_i2v_dp2,
        ),
    ]

    i2v_needed = any(
        (str(cfg["name"]), str(w["label"]), int(w["conc"])) not in done_keys
        for cfg, wls in i2v_configs
        for w in wls
    )
    if i2v_needed:
        i2v_path = stage_wan22_repo(idx, bucket, i2v_repo)
        for cfg, wls in i2v_configs:
            run_config_workloads(
                idx,
                bucket=bucket,
                gcs_prefix=gcs_prefix,
                local_jsonl=local_jsonl,
                local_summary=local_summary,
                pod_rows=pod_rows,
                done_keys=done_keys,
                model_name=i2v_repo,
                model_family="wan22",
                model_path=i2v_path,
                cfg=cfg,
                workloads=wls,
                refs=refs,
                encode_url=encode_url,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Suite 2 SGLang-TorchTPU Video+Audio Serving Optimization")
    parser.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--gcs-prefix", default="v7x-video-opt/results")
    parser.add_argument(
        "--encode-url",
        default=os.environ.get("SGLANG_ENCODE_URL", "http://video-encode-svc.default.svc.cluster.local:8090/encode"),
    )
    parser.add_argument("--pod-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    idx = args.pod_index if args.pod_index is not None else int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    log(
        idx,
        f"Starting Suite 2 SGLang-TorchTPU Optimization on host={HOSTNAME} "
        f"(bucket={args.gcs_bucket}, prefix={args.gcs_prefix}, encode_url={args.encode_url})",
    )

    ensure_subhost_isolation_ready(idx)
    apply_suite2_sglang_tpu_patches(idx)
    check_remote_encoder_health(idx, args.encode_url)

    if idx == 0:
        run_pod0_ltx25(
            idx,
            bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            encode_url=args.encode_url,
            resume=not args.no_resume,
        )
    elif idx == 1:
        run_pod1_wan22(
            idx,
            bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            encode_url=args.encode_url,
            resume=not args.no_resume,
        )
    else:
        raise ValueError(f"Unexpected JOB_COMPLETION_INDEX={idx} (expected 0 or 1)")

    log(idx, f"Suite 2 Pod {idx} completed all configurations cleanly!")


if __name__ == "__main__":
    main()
