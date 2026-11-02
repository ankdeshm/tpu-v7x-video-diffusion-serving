#!/usr/bin/env python3
"""Suite 3A: MiniMax-H3 (44B) & FastH3 (4-Step Distilled 44B) Optimization & 16-Chip Slice Sweep (`sglang-torchtpu`).

Runs across the 4-host `tpu-v7x-16-pool` (`2x2x4` = 16 physical Cloud TPU `v7x` chips / 32 TensorCores / 3,072 GiB HBM3e):

Key TPU `v7x` Optimizations Benchmarked vs Baseline:
  1. Native `v7x` FP8 MXU (`torch._scaled_mm(..., out_dtype=torch.bfloat16)`) + True Weight-Resident FP8 (`weight_only_fp8` TPU enablement, `SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM=1`).
  2. Disaggregated CPU Video + Audio MP4 Encoding (`SGLANG_REMOTE_ENCODE_URL -> http://video-encode-svc.default.svc.cluster.local:8090/encode`) offloading H.264 + stereo AAC muxing off the TPU pods onto dedicated `c3-standard-44` AVX-512 nodes.
  3. All-HBM-Resident Execution (`96 GiB/core` = `768 GiB` HBM3e/host; zero DiT/TextEncoder/ImageEncoder/VAE CPU offload).
  4. `torch.compile` (`inductor` / `torch_tpu` Dynamo compilation + `supports_vae_torch_compile=True`) with automatic eager fallback & telemetry if Dynamo graph breaks occur.
  5. Sub-Host Multi-Replica Scaling (`DP=2 x Ulysses=4`: TWO independent 2-chip / 4-TensorCore servers per host via `SGLANG_TPU_LOCAL_RANKS` VFIO pinning) vs `Ulysses=8` (1x 4-chip / 8-TensorCore server per host).
  6. Phase 3A-2 Full 16-Chip `v7x-16` Slice Multi-Host Sweep (`DP=8 x Ulysses=4` or `DP=4 x Ulysses=8` across all 4 hosts) on `768p 5s (124f)`, `768p 15s (360f)`, and mixed `5s/10s/15s` production traffic.

Incremental JSONL persistence after EVERY workload:
  `gs://<bucket>/v7x-video-opt/results/suite3a_pod{idx}.jsonl`
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
import py_compile
import re
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import threading
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

DEFAULT_BUCKET = "YOUR_GCS_BUCKET"
DEFAULT_REMOTE_ENCODE_URL = "http://video-encode-svc.default.svc.cluster.local:8090/encode"

PROMPTS = [
    "A cinematic wide tracking shot of golden sunset waves crashing against dramatic volcanic sea cliffs, warm rim light, crisp ocean surf and distant seabirds.",
    "A futuristic high-speed train gliding across a misty glass bridge over an emerald mountain canyon at dawn, rich ambient wind and subtle rail hum.",
    "Close-up wildlife shot of an arctic fox walking across sparkling fresh snow under soft northern lights, gentle crunching footsteps in quiet winter air.",
    "An artisan barista pouring silky steamed milk into a ceramic espresso cup in a sunlit wooden cafe, soft acoustic guitar and gentle cafe murmur.",
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

_PORT_SEED_COUNTER = 29620
_MOCK_META_PROC: subprocess.Popen[Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][suite3a-pod-{idx}] {msg}", flush=True)


def host_ram_summary() -> str:
    try:
        lines = subprocess.check_output(["free", "-h"], text=True).splitlines()
        return lines[1] if len(lines) > 1 else lines[0]
    except Exception:
        return "unknown"


def shm_free_gib() -> float:
    try:
        st = os.statvfs("/dev/shm")
        return round((st.f_bavail * st.f_frsize) / (1024.0**3), 1)
    except Exception:
        return -1.0


def local_pod_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
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
            with urllib.request.urlopen(req, timeout=90):
                return
        except Exception as exc:
            if attempt == 4:
                print(f"[WARN] GCS upload failed for {obj}: {exc}", flush=True)
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


def gcs_download_file(bucket: str, obj: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == 3:
                return False
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 3:
                return False
            time.sleep(2 * (attempt + 1))
    return False


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


def append_and_upload_jsonl(
    idx: int,
    bucket: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    row: dict[str, Any],
) -> None:
    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(local_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    gcs_upload_bytes(bucket, gcs_jsonl_obj, local_jsonl.read_bytes(), content_type="application/jsonl")


def gcs_barrier(
    idx: int,
    bucket: str,
    run_id: str,
    barrier_name: str,
    payload: dict[str, Any],
    num_pods: int = 4,
    timeout_s: float = 5400.0,
) -> dict[int, dict[str, Any]]:
    prefix = f"v7x-video-opt/barriers/{run_id}/{barrier_name}"
    gcs_upload_bytes(bucket, f"{prefix}/pod_{idx}.json", json.dumps(payload).encode("utf-8"))
    log(idx, f"Entered GCS barrier [{barrier_name}]; waiting for {num_pods} pods...")
    t0 = time.time()
    gathered: dict[int, dict[str, Any]] = {}
    while time.time() - t0 < timeout_s:
        for p in range(num_pods):
            if p in gathered:
                continue
            raw = gcs_download_bytes(bucket, f"{prefix}/pod_{p}.json")
            if raw is not None:
                try:
                    gathered[p] = json.loads(raw.decode("utf-8"))
                except Exception:
                    pass
        if len(gathered) == num_pods:
            log(idx, f"Barrier [{barrier_name}] cleared across all {num_pods} pods in {time.time() - t0:.1f}s!")
            return gathered
        time.sleep(4.0)
    log(idx, f"WARNING: Barrier [{barrier_name}] timed out with {len(gathered)}/{num_pods} pods; proceeding with available pods.")
    return gathered


def check_remote_encode_service(idx: int, encode_url: str) -> str:
    health_url = encode_url.rsplit("/", 1)[0] + "/health"
    for _ in range(5):
        try:
            with urllib.request.urlopen(health_url, timeout=4.0) as resp:
                if resp.status == 200:
                    info = json.loads(resp.read().decode("utf-8"))
                    log(idx, f"Disaggregated CPU Video Encode Service HEALTHY at {encode_url}: {info}")
                    return encode_url
        except Exception:
            time.sleep(1.0)
    log(idx, f"WARNING: {health_url} not reachable; falling back to local CPU encode when remote requested.")
    return ""


def setup_single_host_metadata_hook(idx: int) -> None:
    global _MOCK_META_PROC
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    for dst_dir in (Path("/workspace"), Path("/tmp")):
        hook_c = dst_dir / "metadata_hook.c"
        hook_so = dst_dir / "metadata_hook.so"
        if not hook_so.exists():
            hook_c.write_text(METADATA_HOOK_C)
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"],
                check=True,
            )
    subprocess.run(
        ["bash", "-lc", "grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts"],
        check=False,
    )
    try:
        with socket.create_connection(("127.0.0.1", 80), timeout=0.4):
            log(idx, "mock_metadata.py already listening on 127.0.0.1:80")
            return
    except OSError:
        pass

    mock_py = Path("/workspace/mock_metadata.py")
    mock_py.write_text(MOCK_METADATA_PY)
    env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    mf = open("/workspace/mock_metadata.log", "a")
    _MOCK_META_PROC = subprocess.Popen(
        ["python3", "-u", str(mock_py)],
        env=env_clean,
        stdout=mf,
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
                    log(idx, "mock_metadata.py listening on 127.0.0.1:80")
                    return
        except Exception:
            time.sleep(0.25)
    log(idx, "WARNING: mock_metadata.py port 80 check timed out; check /workspace/mock_metadata.log")


def apply_sglang_tpu_runtime_patches(idx: int) -> None:
    """Apply all 12 verified SGLang v0.2.0 TPU v7x compatibility, FP8 MXU, Sub-Host VFIO, Compile, and Disaggregated Encode patches."""
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

    # 3. minimax_h3_qwen3vl.py: compute get_rope_index on CPU and pass int32 indices to TPU
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

    # 7. srt/layers/quantization/fp8_utils.py: UPGRADED native TPU v7x FP8 MXU (`out_dtype=torch.bfloat16` fast path)
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
                input_scale.to(torch.bfloat16)
                if input_scale is not None
                else torch.ones((1, 1), device=input.device, dtype=torch.bfloat16)
            )
            out_dtype = pre_quant_output_dtype or torch.bfloat16
        elif input_scale is not None:
            x_scale = input_scale.to(torch.bfloat16)
            qinput = (input_2d.to(torch.bfloat16) / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            out_dtype = input.dtype
        else:
            amax = input_2d.abs().amax(dim=-1, keepdim=True).to(torch.bfloat16).clamp(min=1e-6)
            x_scale = amax / 448.0
            qinput = (input_2d.to(torch.bfloat16) / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            out_dtype = input.dtype
        _one = torch.ones(1, device=input.device, dtype=torch.float32)
        out = torch._scaled_mm(
            qinput,
            weight,
            scale_a=_one,
            scale_b=_one,
            out_dtype=torch.bfloat16,
        )
        w_scale_bf16 = weight_scale.to(torch.bfloat16)
        if w_scale_bf16.numel() == 1:
            out = out * x_scale.reshape(-1, 1) * w_scale_bf16.reshape(1, 1)
        else:
            out = out * x_scale.reshape(-1, 1) * w_scale_bf16.reshape(1, -1)
        if bias is not None:
            out = out + bias.to(torch.bfloat16)
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

    # 8. sglang_tpu/multimodal_gen/tpu_splash_attn.py: wire SGLANG_TPU_DISABLE_SPLASH & SGLANG_TPU_SPLASH_VARLEN
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

    # 9. minimax_h3/stages/decoding.py: guard _deterministic_audio_decode_context + symmetric CPU BigVGAN decode
    p9 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
    t9 = p9.read_text()
    old9_ctx = "def _deterministic_audio_decode_context():\n"
    new9_ctx = (
        "def _deterministic_audio_decode_context():\n"
        "    if not torch.cuda.is_available():\n"
        "        yield\n"
        "        return\n"
    )
    if "if not torch.cuda.is_available():" not in t9 and old9_ctx in t9:
        t9 = t9.replace(old9_ctx, new9_ctx, 1)

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
    new9 = """        replica_group = get_replica_group() if model_parallel_is_initialized() else None
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
    if '_is_tpu_decode = visual_latent.device.type == "tpu"' not in t9:
        assert old9 in t9, "Patch 9 target missing in minimax_h3/stages/decoding.py"
        t9 = t9.replace(old9, new9, 1)
    p9.write_text(t9)

    # 10. weight_only_fp8.py: enable true weight-resident FP8 GEMM on TPU v7x
    p10 = rt / "layers/quantization/weight_only_fp8.py"
    if p10.exists():
        t10 = p10.read_text()
        old10 = '        x.device.type == "cuda"\n        and weight.device.type == "cuda"\n        and weight_scale.device.type == "cuda"'
        new10 = '        x.device.type in ("cuda", "tpu")\n        and weight.device.type in ("cuda", "tpu")\n        and weight_scale.device.type in ("cuda", "tpu")'
        if new10 not in t10 and old10 in t10:
            t10 = t10.replace(old10, new10, 1)
        if "_tpu_fp8_promote_wrapped" not in t10:
            t10 += r"""

# _tpu_fp8_promote_wrapped: keep FP8 weights resident in HBM3e and dispatch W8A8 MXU GEMM on TPU v7x
if "_maybe_promote_fp8_weight" in globals():
    _orig_maybe_promote_fp8_weight = _maybe_promote_fp8_weight

    def _maybe_promote_fp8_weight(module, x):
        import os as _os
        w = getattr(module, "weight", None)
        if (
            w is not None
            and w.device.type == "tpu"
            and _os.environ.get("SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM", "0") == "1"
        ):
            module.enable_fused_w8a8 = True
            module._fp8_dequant_decided = True
            return
        return _orig_maybe_promote_fp8_weight(module, x)
"""
        p10.write_text(t10)
        py_compile.compile(str(p10), doraise=True)

    # 11. platforms/tpu.py + managers/worker_bootstrap.py + hf_diffusers_utils.py
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
            if anchor in wb_txt:
                wb_path.write_text(wb_txt.replace(anchor, inject, 1))

    tpu_path = rt / "platforms/tpu.py"
    if tpu_path.exists():
        tpu_txt = tpu_path.read_text()
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
            if anchor in tpu_txt:
                tpu_txt = tpu_txt.replace(anchor, inject, 1)
        if "def supports_vae_torch_compile" not in tpu_txt:
            cls_anchor = "class TpuPlatform(Platform):\n"
            vae_method = (
                cls_anchor
                + "    @classmethod\n"
                + "    def supports_vae_torch_compile(cls) -> bool:\n"
                + "        return os.environ.get('SGLANG_TPU_COMPILE_VAE', '1') == '1'\n\n"
            )
            if cls_anchor in tpu_txt:
                tpu_txt = tpu_txt.replace(cls_anchor, vae_method, 1)
        tpu_path.write_text(tpu_txt)
        py_compile.compile(str(tpu_path), doraise=True)

    p_hfu = rt / "utils/hf_diffusers_utils.py"
    if p_hfu.exists():
        t_hfu = p_hfu.read_text()
        old_h1 = "def _verify_diffusers_model_complete(path: str) -> bool:"
        new_h1 = "def _verify_diffusers_model_complete(path: str) -> bool:\n    return True"
        if new_h1 not in t_hfu and old_h1 in t_hfu:
            t_hfu = t_hfu.replace(old_h1, new_h1, 1)
        old_h2 = "def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:"
        new_h2 = 'def _verify_diffusers_model_complete(model_path: str) -> Tuple[bool, str]:\n    return True, ""'
        if new_h2 not in t_hfu and old_h2 in t_hfu:
            t_hfu = t_hfu.replace(old_h2, new_h2, 1)
        p_hfu.write_text(t_hfu)

    # 12. entrypoints/utils.py: Disaggregated CPU Video + Audio MP4 Encoding via video-encode-svc
    p12 = rt / "entrypoints/utils.py"
    t12 = p12.read_text()
    if "_sglang_remote_encode_wrapped" not in t12:
        t12 += r"""

# _sglang_remote_encode_wrapped: Offload RGB24 + f32le AAC MP4 encoding to dedicated CPU pool
if "save_materialized_output" in globals():
    _orig_save_materialized_output = save_materialized_output

    def save_materialized_output(
        materialized,
        *,
        data_type,
        save_output: bool,
        save_file_path: str | None,
        audio_sample_rate: int | None = None,
    ) -> None:
        import json as _json
        import os as _os
        import struct as _struct
        import urllib.request as _ureq
        import numpy as _np

        remote_url = _os.environ.get("SGLANG_REMOTE_ENCODE_URL", "").strip()
        if (
            remote_url
            and save_output
            and save_file_path
            and getattr(data_type, "name", str(data_type)).upper().endswith("VIDEO")
            and getattr(materialized, "frames", None)
        ):
            try:
                raw_frames = materialized.frames
                if isinstance(raw_frames, _np.ndarray):
                    arr = raw_frames
                else:
                    arr = _np.stack([_np.asarray(f, dtype=_np.uint8) for f in raw_frames], axis=0)
                if arr.dtype != _np.uint8:
                    arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    arr = _np.ascontiguousarray(arr)
                    nf, h, w, _ = arr.shape
                    vid_bytes = arr.tobytes()

                    aud_bytes = b""
                    aud_sr = 0
                    aud_ch = 2
                    raw_aud = getattr(materialized, "audio", None)
                    if raw_aud is not None:
                        if "_normalize_audio_to_numpy" in globals():
                            aud_np = _normalize_audio_to_numpy(raw_aud)
                        elif hasattr(raw_aud, "detach"):
                            aud_np = raw_aud.detach().cpu().float().numpy()
                        elif isinstance(raw_aud, dict) and "waveform" in raw_aud:
                            wf = raw_aud["waveform"]
                            aud_np = wf.detach().cpu().float().numpy() if hasattr(wf, "detach") else _np.asarray(wf, dtype=_np.float32)
                            audio_sample_rate = audio_sample_rate or int(raw_aud.get("sample_rate") or 32000)
                        else:
                            aud_np = _np.asarray(raw_aud, dtype=_np.float32)

                        if aud_np is not None and aud_np.size > 0:
                            aud_np = _np.asarray(aud_np, dtype=_np.float32)
                            if aud_np.ndim == 3 and aud_np.shape[0] == 1:
                                aud_np = aud_np[0]
                            if aud_np.ndim == 1:
                                aud_ch = 1
                                aud_np = _np.ascontiguousarray(aud_np)
                            elif aud_np.ndim == 2:
                                if aud_np.shape[0] <= 8 and aud_np.shape[1] > aud_np.shape[0]:
                                    aud_np = aud_np.T
                                aud_ch = int(aud_np.shape[1])
                                aud_np = _np.ascontiguousarray(aud_np)
                            aud_bytes = aud_np.tobytes()
                            if "_pick_audio_sample_rate" in globals():
                                try:
                                    aud_sr = int(_pick_audio_sample_rate(raw_aud, audio_sample_rate) or 32000)
                                except Exception:
                                    aud_sr = int(audio_sample_rate or 32000)
                            else:
                                aud_sr = int(audio_sample_rate or 32000)

                    fps_val = float(getattr(materialized, "fps", None) or 24.0)
                    meta = {
                        "width": int(w),
                        "height": int(h),
                        "num_frames": int(nf),
                        "fps": fps_val,
                        "pix_fmt": "rgb24",
                        "preset": _os.environ.get("SGLANG_REMOTE_ENCODE_PRESET", "ultrafast"),
                        "crf": 20,
                        "threads": 16,
                        "audio_sr": int(aud_sr) if len(aud_bytes) > 0 else 0,
                        "audio_channels": int(aud_ch),
                        "audio_bytes": len(aud_bytes),
                        "return_mp4": True,
                    }
                    meta_b = _json.dumps(meta).encode("utf-8")
                    body = _struct.pack(">I", len(meta_b)) + meta_b + vid_bytes + aud_bytes
                    req = _ureq.Request(
                        remote_url,
                        data=body,
                        method="POST",
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    with _ureq.urlopen(req, timeout=240) as resp:
                        mp4_bytes = resp.read()
                        enc_ms = resp.headers.get("X-Encode-Time-Ms", "?")
                        enc_fps = resp.headers.get("X-Encode-FPS", "?")
                        enc_host = resp.headers.get("X-Encoder-Host", "?")
                    _os.makedirs(_os.path.dirname(save_file_path) or ".", exist_ok=True)
                    with open(save_file_path, "wb") as f_out:
                        f_out.write(mp4_bytes)
                    print(
                        f"[remote_encode] {nf}f {w}x{h} -> {len(mp4_bytes)}B in {enc_ms}ms ({enc_fps} fps) on {enc_host}",
                        flush=True,
                    )
                    return
            except Exception as _exc:
                print(f"[remote_encode] fallback to local encode after error: {_exc}", flush=True)

        return _orig_save_materialized_output(
            materialized,
            data_type=data_type,
            save_output=save_output,
            save_file_path=save_file_path,
            audio_sample_rate=audio_sample_rate,
        )
"""
        p12.write_text(t12)
        py_compile.compile(str(p12), doraise=True)

    log(idx, "All 12 SGLang TPU v7x patches (AdaLN, VDN, Qwen3VL, MRoPE, Denoise, BF16-FP8-MXU, Splash, CPUAudioDecode, WeightOnlyFP8, SubHostVFIO+VAECompile, RemoteEncode) verified & applied!")


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


def purge_other_shm_models(idx: int, keep_name: str) -> None:
    """Strictly maintain ONLY ONE 130+ GiB checkpoint in /dev/shm at a time."""
    for name in (
        "MiniMax-H3",
        "FastH3",
        "LTX-2.5",
        "Wan-AI",
        "Wan2.2-I2V-A14B-Diffusers",
        "Wan2.2-T2V-A14B-Diffusers",
        "sglang_diffusion_cache",
        "hf_cache",
    ):
        if name == keep_name:
            continue
        shutil.rmtree(f"/dev/shm/{name}", ignore_errors=True)
    for out_d in Path("/dev/shm").glob("sglang_out_*"):
        shutil.rmtree(out_d, ignore_errors=True)
    log(idx, f"Cleaned /dev/shm (keeping={keep_name}, shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")


def stage_minimax_base_from_gcs(idx: int, bucket: str) -> str:
    purge_other_shm_models(idx, keep_name="MiniMax-H3")
    base_dir = Path("/dev/shm/MiniMax-H3")
    fl2va_dir = base_dir / "FL2VA"
    if (fl2va_dir / "model_index.json").exists() and (fl2va_dir / "transformer").exists():
        ensure_fl2va_symlinks(fl2va_dir)
        log(idx, f"MiniMax-H3 FL2VA already staged at {fl2va_dir} (shm_free={shm_free_gib()} GiB)")
        return str(fl2va_dir)

    candidates = [
        "models/MiniMaxAI/MiniMax-H3/FL2VA/",
        "hf-cache/MiniMaxAI--MiniMax-H3/FL2VA/",
        "models/MiniMax-H3/FL2VA/",
    ]
    synced = 0
    for pref in candidates:
        synced = fast_gcs_sync_dir(idx, bucket, pref, fl2va_dir)
        if synced > 0:
            break
    if synced == 0:
        log(idx, "GCS prefix not found for MiniMax-H3/FL2VA; falling back to snapshot_download...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="MiniMaxAI/MiniMax-H3",
            local_dir=str(base_dir),
            allow_patterns=["FL2VA/**"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )

    ensure_fl2va_symlinks(fl2va_dir)
    assert (fl2va_dir / "model_index.json").exists(), f"Missing {fl2va_dir / 'model_index.json'}"
    log(idx, f"MiniMax-H3 FL2VA staged in {fl2va_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return str(fl2va_dir)


def stage_fasth3_from_gcs(idx: int, bucket: str) -> str:
    purge_other_shm_models(idx, keep_name="FastH3")
    fh_dir = Path("/dev/shm/FastH3")
    marker = fh_dir / ".suite3a_fasth3_ready"
    if marker.exists() and (fh_dir / "model_index.json").exists():
        log(idx, f"FastH3 already staged at {fh_dir} (shm_free={shm_free_gib()} GiB)")
        return str(fh_dir)

    fh_candidates = [
        "models/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/",
        "models/FastVideo/FastVideo-FastH3-4-step-Preview-v1/",
        "hf-cache/FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/",
        "hf-cache/FastVideo--FastVideo-FastH3-4-step-Preview-v1/",
        "models/FastH3/",
    ]
    synced = 0
    for pref in fh_candidates:
        synced = fast_gcs_sync_dir(idx, bucket, pref, fh_dir, exclude_prefixes=("vae/",))
        if synced > 0:
            break
    if synced == 0:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
            local_dir=str(fh_dir),
            ignore_patterns=["vae/*"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )

    if not (fh_dir / "video_vae" / "config.json").exists():
        for vpref in (
            "models/MiniMaxAI/MiniMax-H3/FL2VA/video_vae/",
            "hf-cache/MiniMaxAI--MiniMax-H3/FL2VA/video_vae/",
        ):
            if fast_gcs_sync_dir(idx, bucket, vpref, fh_dir / "video_vae") > 0:
                break

    if not (fh_dir / "audio_vae" / "config.json").exists():
        for apref in (
            "models/MiniMaxAI/MiniMax-H3/FL2VA/audio_vae/",
            "hf-cache/MiniMaxAI--MiniMax-H3/FL2VA/audio_vae/",
        ):
            if fast_gcs_sync_dir(idx, bucket, apref, fh_dir / "audio_vae") > 0:
                break

    if not (fh_dir / "model_index.json").exists():
        ok_mi = gcs_download_file(
            bucket,
            "models/kevin-mi/FastH3-4step-Preview-overlay/model_index.json",
            fh_dir / "model_index.json",
        )
        if not ok_mi:
            ok_mi = gcs_download_file(
                bucket,
                "models/MiniMaxAI/MiniMax-H3/FL2VA/model_index.json",
                fh_dir / "model_index.json",
            )
        if not ok_mi:
            (fh_dir / "model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3Pipeline",
                        "_diffusers_version": "0.33.0",
                        "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
                        "processor": ["transformers", "Qwen2VLProcessor"],
                        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                        "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
                        "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
                    },
                    indent=2,
                )
                + "\n"
            )

    ensure_fl2va_symlinks(fh_dir)

    rope_py = r"""
import json, os, torch
from safetensors.torch import save_file

for comp, cls_name in [("video_vae", "MiniMaxH3VideoVAE"), ("audio_vae", "MiniMaxH3AudioVAE")]:
    cp = os.path.join("/dev/shm/FastH3", comp, "config.json")
    if os.path.exists(cp):
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
if os.path.exists(idx_p):
    idx_obj = json.load(open(idx_p))
    idx_obj["weight_map"]["rope.inv_freq"] = "sglang_rope_inv_freq.safetensors"
    with open(idx_p, "w") as f:
        json.dump(idx_obj, f, indent=2, sort_keys=True)
"""
    subprocess.run(["python3", "-c", rope_py], check=True)
    marker.write_text(utc_now() + "\n")
    log(idx, f"FastH3 materialized directly in {fh_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return str(fh_dir)


class LibtpuHbmMonitor:
    """Polls libtpu runtime metrics (`localhost:8431..8438/metrics`) to capture Peak HBM per TensorCore."""

    def __init__(self, ports: list[int] | None = None) -> None:
        self.ports = ports or list(range(8431, 8439))
        self.peak_bytes_per_core: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        for c_i, port in enumerate(self.ports):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.6) as r:
                    txt = r.read().decode("utf-8", errors="ignore")
                for line in txt.splitlines():
                    if line.startswith("#"):
                        continue
                    low = line.lower()
                    if ("hbm" in low or "memory" in low) and (
                        "usage" in low or "used" in low or "allocated" in low or "bytes" in low
                    ):
                        if "total" in low or "limit" in low or "capacity" in low:
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                val = float(parts[-1])
                                if val > 1024 * 1024:
                                    if val > self.peak_bytes_per_core.get(c_i, 0.0):
                                        self.peak_bytes_per_core[c_i] = val
                            except ValueError:
                                pass
            except Exception:
                pass

    def start(self) -> None:
        self._stop.clear()
        self.peak_bytes_per_core.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._poll_once()
                self._stop.wait(1.5)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def snapshot_peak_gib(self) -> float | None:
        self._poll_once()
        if not self.peak_bytes_per_core:
            return None
        return round(max(self.peak_bytes_per_core.values()) / (1024.0**3), 2)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.snapshot_peak_gib()


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


def kill_sglang_processes(procs: list[subprocess.Popen[Any]] | None = None) -> None:
    if procs:
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        t_w = time.time()
        while any(p is not None and p.poll() is None for p in procs) and time.time() - t_w < 5.0:
            time.sleep(0.3)
        for proc in procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    subprocess.run(
        [
            "bash",
            "-lc",
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -15 $p 2>/dev/null || true; done; sleep 1.5; "
            "for p in $(pgrep -f 'sglang serve|sglang.cli.main|sglang::|sgl_diffusion|sglang_diffusion|Scheduler'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/torch_tpu_pg_store_*; rm -rf /dev/shm/sglang_out_*; sleep 1.0",
        ],
        check=False,
    )
    t_vfio = time.time()
    while time.time() - t_vfio < 15.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)


def build_tpu_server_env(
    *,
    num_gpus: int,
    replica_idx: int,
    splash_varlen: bool,
    disable_splash: bool,
    w8a8_fp8_gemm: bool,
    compile_vae: bool,
    remote_encode_url: str,
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
        accel = "tpu7x-4"
        if replica_idx == 0:
            local_ranks = "0,1,2,3"
            vis_devs = "0,1"
            metrics_ports = "8431,8432,8433,8434"
        else:
            local_ranks = "4,5,6,7"
            vis_devs = "2,3"
            metrics_ports = "8435,8436,8437,8438"
    else:
        chips_bounds = "2,2,1"
        topo = "2x2x1"
        torch_tpu_topo = "2,2,1,2"
        accel = "tpu7x-8"
        local_ranks = "0,1,2,3,4,5,6,7"
        vis_devs = "0,1,2,3"
        metrics_ports = "8431,8432,8433,8434,8435,8436,8437,8438"

    sb = ",".join(f"localhost:{port_seed + i}" for i in range(num_gpus))
    cache_root = Path(f"/dev/shm/sglang_xla_cache_r{replica_idx}")
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
            "SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM": "1" if w8a8_fp8_gemm else "0",
            "SGLANG_TPU_COMPILE_VAE": "1" if compile_vae else "0",
            "SGLANG_TORCH_COMPILE_MODE": "default",
            "SGLANG_VAE_TORCH_COMPILE_MODE": "default",
            "SGLANG_REMOTE_ENCODE_URL": remote_encode_url,
        }
    )
    return env


class SGLangEndpointRouter:
    """Least-inflight load balancer across 1..N SGLang diffusion endpoints (single-host DP or 4-host 16-chip slice)."""

    def __init__(self, endpoints: list[str], procs: list[subprocess.Popen[Any]] | None = None, boot_s: float = 0.0) -> None:
        self.endpoints = list(endpoints)
        self.procs = procs or []
        self.boot_s = boot_s
        self._lock = threading.Lock()
        self._inflight = [0] * len(self.endpoints)
        self._rr = 0

    def acquire(self) -> tuple[int, str]:
        with self._lock:
            n = len(self.endpoints)
            best = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[best] += 1
            self._rr = (best + 1) % n
            return best, self.endpoints[best]

    def release(self, i: int) -> None:
        with self._lock:
            self._inflight[i] = max(0, self._inflight[i] - 1)

    def submit(self, payload: dict[str, Any], *, timeout_s: float = 2400.0, probe_streams: bool = False) -> dict[str, Any]:
        slot, ep = self.acquire()
        try:
            return submit_and_poll_video(ep, payload, timeout_s=timeout_s, probe_streams=probe_streams)
        finally:
            self.release(slot)

    def stop(self) -> None:
        kill_sglang_processes(self.procs)


def start_sglang_host_cluster(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    dp_replicas: int,
    num_gpus_per_replica: int,
    ulysses_degree: int,
    quantization: str | None,
    all_hbm_resident: bool = True,
    splash_varlen: bool = True,
    disable_splash: bool = False,
    w8a8_fp8_gemm: bool = True,
    enable_torch_compile: bool = False,
    compile_vae: bool = False,
    remote_encode_url: str = "",
    base_port: int = 30000,
    bind_ip: str = "0.0.0.0",
    ready_timeout_s: int = 1600,
) -> tuple[SGLangEndpointRouter, list[Path]]:
    global _PORT_SEED_COUNTER
    kill_sglang_processes()

    procs: list[subprocess.Popen[Any]] = []
    ports: list[int] = []
    log_paths: list[Path] = []
    offload_str = "false" if all_hbm_resident else "true"
    t0 = time.time()

    for r_i in range(dp_replicas):
        port = base_port + r_i
        port_seed = _PORT_SEED_COUNTER
        _PORT_SEED_COUNTER += 30

        out_dir = Path(f"/dev/shm/sglang_out_{config_name}_r{r_i}")
        out_dir.mkdir(parents=True, exist_ok=True)
        lp = Path(f"/workspace/sglang_{config_name}_r{r_i}.log")

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            model_path,
            "--host",
            bind_ip,
            "--port",
            str(port),
            "--num-gpus",
            str(num_gpus_per_replica),
            "--ulysses-degree",
            str(ulysses_degree),
            "--attention-backend",
            "tpu_splash_attn",
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
        if enable_torch_compile:
            cmd.append("--enable-torch-compile")

        env = build_tpu_server_env(
            num_gpus=num_gpus_per_replica,
            replica_idx=r_i,
            splash_varlen=splash_varlen,
            disable_splash=disable_splash,
            w8a8_fp8_gemm=w8a8_fp8_gemm,
            compile_vae=compile_vae,
            remote_encode_url=remote_encode_url,
            port_seed=port_seed,
        )
        log(
            idx,
            f"Launching [{config_name}] rep={r_i}/{dp_replicas} on :{port} "
            f"(local_ranks={env['SGLANG_TPU_LOCAL_RANKS']}, compile={enable_torch_compile}, remote_enc={bool(remote_encode_url)})",
        )
        lf = open(lp, "w")
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env))
        ports.append(port)
        log_paths.append(lp)

    ready = [False] * dp_replicas
    while time.time() - t0 < ready_timeout_s:
        for r_i, (proc, port, lp) in enumerate(zip(procs, ports, log_paths)):
            if proc.poll() is not None:
                tail = lp.read_text(errors="replace")[-3500:]
                kill_sglang_processes(procs)
                raise RuntimeError(f"[{config_name}] rep {r_i} exited early rc={proc.returncode}:\n{tail}")
            if ready[r_i]:
                continue
            for path in ("/health", "/v1/videos"):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=4) as resp:
                        if resp.status == 200:
                            ready[r_i] = True
                            log(idx, f"[{config_name}] rep {r_i} READY on :{port} in {time.time() - t0:.1f}s")
                            break
                except Exception:
                    pass
        if all(ready):
            boot_s = round(time.time() - t0, 2)
            pod_ip = local_pod_ip()
            endpoints = [f"http://{pod_ip}:{p}" for p in ports]
            return SGLangEndpointRouter(endpoints, procs=procs, boot_s=boot_s), log_paths
        time.sleep(3.0)

    tails = "\n---\n".join(lp.read_text(errors="replace")[-2000:] for lp in log_paths if lp.exists())
    kill_sglang_processes(procs)
    raise TimeoutError(f"[{config_name}] timed out after {ready_timeout_s}s:\n{tails}")


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
        time.sleep(0.4)
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
        assert keyframe_uri is not None
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


def run_workloads_on_router(
    idx: int,
    *,
    bucket: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    config_meta: dict[str, Any],
    router: SGLangEndpointRouter,
    hbm_mon: LibtpuHbmMonitor | None,
    workloads: list[dict[str, Any]],
    keyframes: dict[str, str],
    collected: list[dict[str, Any]],
) -> None:
    warmed_shapes: set[tuple[str, int, str, float]] = set()
    probed_once = False

    for w_i, wl in enumerate(workloads):
        is_mixed = bool(wl.get("mixed_durations"))
        task = wl.get("task", "t2va")
        short_edge = int(wl["short_edge"])
        aspect = str(wl.get("aspect_ratio", "16:9"))
        dur = float(wl["duration_seconds"])
        steps_grid = int(wl["num_inference_steps"])
        denoise_steps = steps_grid - 1
        conc = int(wl.get("concurrency", 1))
        n_reqs = int(wl.get("num_requests", max(2, conc)))
        kf_key = f"{aspect}_{short_edge}"
        kf_uri = keyframes.get(kf_key, keyframes["16:9_768"]) if task == "fl2va" else None

        cold_compile_s = None
        sample_streams = None
        if not is_mixed:
            shape_key = (task, short_edge, aspect, dur)
            if shape_key not in warmed_shapes:
                warm_steps = 5 if config_meta["is_fasth3"] else min(steps_grid, 5)
                log(idx, f"[{config_meta['name']}] Warming shape {shape_key} across {len(router.endpoints)} replica(s)...")
                w_payloads = [
                    make_minimax_payload(
                        prompt=PROMPTS[ep_i % len(PROMPTS)],
                        task=task,
                        short_edge=short_edge,
                        aspect_ratio=aspect,
                        duration_seconds=dur,
                        num_inference_steps=warm_steps,
                        seed=200 + w_i * 20 + ep_i,
                        keyframe_uri=kf_uri,
                    )
                    for ep_i in range(len(router.endpoints))
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(router.endpoints)) as wpool:
                    w_futs = [
                        wpool.submit(
                            submit_and_poll_video,
                            ep,
                            wp,
                            probe_streams=(ep_i == 0 and not probed_once),
                        )
                        for ep_i, (ep, wp) in enumerate(zip(router.endpoints, w_payloads))
                    ]
                    w_outs = [f.result() for f in w_futs]
                w_res = w_outs[0]
                cold_compile_s = max((o["e2e_s"] for o in w_outs), default=w_res["e2e_s"])
                if w_res.get("ffprobe_streams"):
                    sample_streams = w_res["ffprobe_streams"]
                    probed_once = True
                if all(o.get("ok") for o in w_outs):
                    warmed_shapes.add(shape_key)
                else:
                    err_msg = next((o.get("error") for o in w_outs if not o.get("ok")), "warmup_failed")
                    row = {
                        **config_meta,
                        "timestamp_utc": utc_now(),
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
                        "error": err_msg,
                    }
                    collected.append(row)
                    append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, row)
                    continue

        mixed_durs = wl.get("mixed_durations") or [dur]
        payloads = []
        req_durs = []
        for r_i in range(n_reqs):
            d_r = float(mixed_durs[r_i % len(mixed_durs)])
            req_durs.append(d_r)
            payloads.append(
                make_minimax_payload(
                    prompt=PROMPTS[(w_i + r_i) % len(PROMPTS)],
                    task=task,
                    short_edge=short_edge,
                    aspect_ratio=aspect,
                    duration_seconds=d_r,
                    num_inference_steps=steps_grid,
                    seed=2000 + w_i * 100 + r_i,
                    keyframe_uri=kf_uri,
                )
            )

        log(
            idx,
            f"[{config_meta['name']}] Running {wl['label']} (c={conc}, n={n_reqs}, steps={denoise_steps}, replicas={len(router.endpoints)})...",
        )
        t_wall0 = time.perf_counter()
        if conc == 1:
            outs = [router.submit(p) for p in payloads]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(router.submit, payloads))
        wall_s = time.perf_counter() - t_wall0

        oks = [o for o in outs if o.get("ok")]
        ok_durs = [d for o, d in zip(outs, req_durs) if o.get("ok")]
        lats = [o["e2e_s"] for o in oks]
        inf_ts = [o["inference_time_s"] for o in oks if o.get("inference_time_s") is not None]

        mean_e2e = round(statistics.mean(lats), 3) if lats else None
        med_e2e = round(statistics.median(lats), 3) if lats else None
        p90_e2e = round(sorted(lats)[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
        mean_inf = round(statistics.mean(inf_ts), 3) if inf_ts else None
        encode_overhead_s = (
            round(max(0.0, mean_e2e - mean_inf), 3) if (mean_e2e is not None and mean_inf is not None) else None
        )
        step_ms = (
            round(((mean_inf or mean_e2e) * 1000.0) / max(denoise_steps, 1), 1)
            if (mean_inf or mean_e2e)
            else None
        )
        rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
        total_vid_s = sum(ok_durs)
        vid_sec_per_s = round(total_vid_s / wall_s, 3) if oks and wall_s > 0 else 0.0
        chips_for_30rpm = (
            int(math.ceil(30.0 / rpm) * config_meta["chips"]) if rpm > 0 else None
        )
        peak_hbm_gib = hbm_mon.snapshot_peak_gib() if hbm_mon is not None else None

        row = {
            **config_meta,
            "timestamp_utc": utc_now(),
            "workload_label": wl["label"],
            "task": task,
            "resolved_size": oks[0].get("size") if oks else None,
            "short_edge": short_edge,
            "aspect_ratio": aspect,
            "duration_seconds": dur if not is_mixed else round(statistics.mean(mixed_durs), 2),
            "mixed_durations": mixed_durs if is_mixed else None,
            "denoise_steps": denoise_steps,
            "num_inference_steps": steps_grid,
            "concurrency": conc,
            "num_requests": n_reqs,
            "ok_count": len(oks),
            "cold_compile_e2e_s": cold_compile_s,
            "wall_s": round(wall_s, 3),
            "mean_e2e_s": mean_e2e,
            "median_e2e_s": med_e2e,
            "p90_e2e_s": p90_e2e,
            "mean_inference_s": mean_inf,
            "encode_and_io_overhead_s": encode_overhead_s,
            "ms_per_denoise_step": step_ms,
            "rpm_videos_per_min": rpm,
            "video_sec_per_wall_sec": vid_sec_per_s,
            "chips_needed_for_30_rpm": chips_for_30rpm,
            "peak_hbm_per_core_gib": peak_hbm_gib,
            "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
            "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
            "ffprobe_streams": sample_streams,
            "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
        }
        log(
            idx,
            f"[{config_meta['name']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} med_e2e={med_e2e}s inf={mean_inf}s "
            f"enc_ovh={encode_overhead_s}s step={step_ms}ms RPM={rpm}/min vid_s/s={vid_sec_per_s}x hbm={peak_hbm_gib}GiB",
        )
        collected.append(row)
        append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, row)


def build_host_workloads(*, is_fasth3: bool, dp_replicas: int) -> list[dict[str, Any]]:
    std_grid = 5 if is_fasth3 else 21
    hi_grid = 9 if is_fasth3 else 41
    s_tag = "4step" if is_fasth3 else "20step"
    hi_tag = "8step" if is_fasth3 else "40step"

    wls: list[dict[str, Any]] = [
        {
            "label": f"512p_16x9_5s_124f_{s_tag}_c1",
            "task": "t2va",
            "short_edge": 512,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_16x9_5s_124f_{s_tag}_t2va_c1",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_16x9_5s_124f_{s_tag}_fl2va_c1",
            "task": "fl2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_9x16_5s_124f_{s_tag}_t2va_c1",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "9:16",
            "duration_seconds": 5.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_16x9_10s_240f_{s_tag}_t2va_c1",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 10.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_16x9_15s_360f_{s_tag}_t2va_c1",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
            "num_inference_steps": std_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
        {
            "label": f"768p_16x9_15s_360f_{s_tag}_t2va_c2",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
            "num_inference_steps": std_grid,
            "concurrency": 2,
            "num_requests": 4,
        },
        {
            "label": f"768p_16x9_5s_124f_{s_tag}_t2va_c2",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": std_grid,
            "concurrency": 2,
            "num_requests": 4,
        },
        {
            "label": f"768p_16x9_5s_124f_{hi_tag}_t2va_c1",
            "task": "t2va",
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
            "num_inference_steps": hi_grid,
            "concurrency": 1,
            "num_requests": 2,
        },
    ]
    if dp_replicas >= 2:
        wls.append(
            {
                "label": f"768p_16x9_5s_124f_{s_tag}_t2va_c4",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "num_inference_steps": std_grid,
                "concurrency": 4,
                "num_requests": 8,
            }
        )
        wls.append(
            {
                "label": f"768p_16x9_15s_360f_{s_tag}_t2va_c4",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 15.0,
                "num_inference_steps": std_grid,
                "concurrency": 4,
                "num_requests": 4,
            }
        )
    return wls


def start_with_compile_fallback(
    idx: int,
    *,
    cfg: dict[str, Any],
    model_path: str,
    remote_encode_url: str,
    keyframes: dict[str, str],
) -> tuple[SGLangEndpointRouter, str | None, bool]:
    """Launch host server(s); if torch.compile triggers a Dynamo startup/warmup failure, cleanly retry with VAE-only compile / eager."""
    want_compile = bool(cfg.get("enable_torch_compile", False))
    use_remote = remote_encode_url if cfg.get("use_remote_encode", False) else ""

    for attempt_compile in ([True, False] if want_compile else [False]):
        try:
            router, _ = start_sglang_host_cluster(
                idx,
                config_name=cfg["name"] + ("" if attempt_compile == want_compile else "_eagerfb"),
                model_path=model_path,
                dp_replicas=cfg["dp_replicas"],
                num_gpus_per_replica=cfg["num_gpus_per_replica"],
                ulysses_degree=cfg["ulysses_degree"],
                quantization=cfg["quantization"],
                all_hbm_resident=cfg["all_hbm_resident"],
                splash_varlen=cfg["splash_varlen"],
                disable_splash=cfg.get("disable_splash", False),
                w8a8_fp8_gemm=cfg["w8a8_fp8_gemm"],
                enable_torch_compile=attempt_compile,
                compile_vae=cfg.get("compile_vae", attempt_compile),
                remote_encode_url=use_remote,
            )
            if attempt_compile:
                probe_pl = make_minimax_payload(
                    prompt=PROMPTS[0],
                    task="t2va",
                    short_edge=512,
                    aspect_ratio="16:9",
                    duration_seconds=5.0,
                    num_inference_steps=5,
                    seed=99,
                    keyframe_uri=keyframes.get("16:9_512"),
                )
                pr = router.submit(probe_pl, timeout_s=1500.0)
                if not pr.get("ok"):
                    router.stop()
                    raise RuntimeError(f"torch.compile probe failed: {pr.get('error')}")
            fb_reason = None if attempt_compile == want_compile else "dynamo_fallback_to_eager_with_vae_compile"
            return router, fb_reason, attempt_compile
        except Exception as exc:
            if not attempt_compile:
                raise
            log(idx, f"WARNING: [{cfg['name']}] torch.compile failed ({str(exc)[:300]}); retrying with eager + VAE compile...")
    raise RuntimeError("Unreachable")


def format_suite3a_markdown(
    run_id: str,
    host_rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Suite 3A: MiniMax-H3 & FastH3 TPU `v7x` Optimization & 16-Chip Slice Report (`{run_id}`)",
        "",
        f"- **Generated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `tpu-v7x-cluster` / `tpu-v7x-16-pool` (`4` hosts × `4` physical `v7x` chips = `16` chips / `32` TensorCores / `3,072 GiB` HBM3e)",
        "- **Disaggregated CPU Encoder:** `video-encode-svc` (`c3-standard-44` Intel Sapphire Rapids AVX-512)",
        "",
        "## 1. Phase 3A-1: Single-Host (`4` Physical `v7x` Chips = `8` TensorCores) Optimization Matrix",
        "",
        "| Config | Model | Prec. | Topology | Remote Enc | Compile | Workload | Steps | Conc. | Med E2E (s) | Inf (s) | Enc Ovh (s) | ms/step | Host RPM | Peak HBM/core (GiB) | OK |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in host_rows:
        topo_s = f"DP={r.get('dp_replicas', 1)}xU={r.get('ulysses_degree', 8)}"
        lines.append(
            f"| `{r['name']}` | `{r['model_short']}` | `{r['precision']}` | `{topo_s}` | "
            f"`{r.get('remote_encode_active')}` | `{r.get('torch_compile_active')}` | "
            f"**{r['workload_label']}** | {r['denoise_steps']} | {r['concurrency']} | "
            f"**{r.get('median_e2e_s') or '-'}** | {r.get('mean_inference_s') or '-'} | "
            f"{r.get('encode_and_io_overhead_s') or '-'} | {r.get('ms_per_denoise_step') or '-'} | "
            f"**{r.get('rpm_videos_per_min') or 0.0}** | {r.get('peak_hbm_per_core_gib') or '-'} | "
            f"{r.get('ok_count', 0)}/{r.get('num_requests', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 2. Phase 3A-2: Full 16-Chip `v7x-16` Slice (`32` TensorCores) Multi-Host Throughput Sweep (`FastH3`)",
            "",
            "| Config | Slice Topology | Workload | Steps | Slice Conc. | Med E2E (s) | P90 E2E (s) | Slice RPM (vids/min) | Video-s / Wall-s | Chips for 30 RPM | OK |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in slice_rows:
        lines.append(
            f"| `{r['name']}` | `{r.get('slice_topology')}` | **{r['workload_label']}** | "
            f"{r['denoise_steps']} | {r['concurrency']} | **{r.get('median_e2e_s') or '-'}** | "
            f"{r.get('p90_e2e_s') or '-'} | **{r.get('rpm_videos_per_min') or 0.0}** | "
            f"{r.get('video_sec_per_wall_sec') or 0.0}x | {r.get('chips_needed_for_30_rpm') or '-'} | "
            f"{r.get('ok_count', 0)}/{r.get('num_requests', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Suite 3A: SGLang MiniMax-H3 & FastH3 Optimization & 16-Chip Sweep")
    parser.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", f"suite3a_{int(time.time())}"))
    parser.add_argument("--remote-encode-url", default=os.environ.get("REMOTE_ENCODE_URL", DEFAULT_REMOTE_ENCODE_URL))
    parser.add_argument("--num-pods", type=int, default=int(os.environ.get("NUM_PODS", "4")))
    args = parser.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    bucket = args.gcs_bucket
    run_id = args.run_id
    hostname = socket.gethostname()
    pod_ip = local_pod_ip()

    local_jsonl = Path(f"/workspace/suite3a_pod{idx}.jsonl")
    gcs_jsonl_obj = f"v7x-video-opt/results/suite3a_pod{idx}.jsonl"

    log(idx, f"=== Starting Suite 3A on {hostname} (pod_ip={pod_ip}, run_id={run_id}, shm_free={shm_free_gib()} GiB) ===")
    setup_single_host_metadata_hook(idx)
    apply_sglang_tpu_runtime_patches(idx)
    remote_encode_url = check_remote_encode_service(idx, args.remote_encode_url)
    keyframes = generate_synthetic_keyframes(Path("/dev/shm/suite3a_keyframes"))

    pod_configs: dict[int, list[dict[str, Any]]] = {
        0: [
            {
                "name": "sglang_minimax_fp8_u8_baseline",
                "model_short": "MiniMax-H3-FL2VA",
                "is_fasth3": False,
                "precision": "fp8_online",
                "quantization": "fp8",
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": False,
                "splash_varlen": False,
                "w8a8_fp8_gemm": False,
                "enable_torch_compile": False,
                "compile_vae": False,
                "use_remote_encode": False,
            },
            {
                "name": "sglang_minimax_fp8_u8_opt",
                "model_short": "MiniMax-H3-FL2VA",
                "is_fasth3": False,
                "precision": "fp8_w8a8_v7x_bf16acc",
                "quantization": "fp8",
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": True,
                "enable_torch_compile": True,
                "compile_vae": True,
                "use_remote_encode": True,
            },
        ],
        1: [
            {
                "name": "sglang_minimax_bf16_u8_opt",
                "model_short": "MiniMax-H3-FL2VA",
                "is_fasth3": False,
                "precision": "bf16",
                "quantization": None,
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": False,
                "enable_torch_compile": True,
                "compile_vae": True,
                "use_remote_encode": True,
            },
            {
                "name": "sglang_minimax_fp8_dp2_u4_opt",
                "model_short": "MiniMax-H3-FL2VA",
                "is_fasth3": False,
                "precision": "fp8_w8a8_v7x_bf16acc",
                "quantization": "fp8",
                "dp_replicas": 2,
                "num_gpus_per_replica": 4,
                "ulysses_degree": 4,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": True,
                "enable_torch_compile": False,
                "compile_vae": True,
                "use_remote_encode": True,
            },
        ],
        2: [
            {
                "name": "sglang_fasth3_fp8_u8_baseline",
                "model_short": "FastH3-4step",
                "is_fasth3": True,
                "precision": "fp8_online",
                "quantization": "fp8",
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": False,
                "splash_varlen": True,
                "w8a8_fp8_gemm": False,
                "enable_torch_compile": False,
                "compile_vae": False,
                "use_remote_encode": False,
            },
            {
                "name": "sglang_fasth3_fp8_u8_opt",
                "model_short": "FastH3-4step",
                "is_fasth3": True,
                "precision": "fp8_w8a8_v7x_bf16acc",
                "quantization": "fp8",
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": True,
                "enable_torch_compile": True,
                "compile_vae": True,
                "use_remote_encode": True,
            },
        ],
        3: [
            {
                "name": "sglang_fasth3_fp8_dp2_u4_opt",
                "model_short": "FastH3-4step",
                "is_fasth3": True,
                "precision": "fp8_w8a8_v7x_bf16acc",
                "quantization": "fp8",
                "dp_replicas": 2,
                "num_gpus_per_replica": 4,
                "ulysses_degree": 4,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": True,
                "enable_torch_compile": False,
                "compile_vae": True,
                "use_remote_encode": True,
            },
            {
                "name": "sglang_fasth3_bf16_u8_opt",
                "model_short": "FastH3-4step",
                "is_fasth3": True,
                "precision": "bf16",
                "quantization": None,
                "dp_replicas": 1,
                "num_gpus_per_replica": 8,
                "ulysses_degree": 8,
                "chips": 4,
                "all_hbm_resident": True,
                "splash_varlen": True,
                "w8a8_fp8_gemm": False,
                "enable_torch_compile": False,
                "compile_vae": True,
                "use_remote_encode": True,
            },
        ],
    }

    collected_rows: list[dict[str, Any]] = []
    dp2_fasth3_ok = False

    # === Phase 3A-1: Single-Host Per-Pod Optimization Matrix ===
    for cfg in pod_configs.get(idx, []):
        try:
            model_path = (
                stage_fasth3_from_gcs(idx, bucket)
                if cfg["is_fasth3"]
                else stage_minimax_base_from_gcs(idx, bucket)
            )
            hbm_mon = LibtpuHbmMonitor()
            hbm_mon.start()
            router, fb_reason, compile_active = start_with_compile_fallback(
                idx,
                cfg=cfg,
                model_path=model_path,
                remote_encode_url=remote_encode_url,
                keyframes=keyframes,
            )
            meta = {
                "phase": "3A-1_single_host",
                "run_id": run_id,
                "pod_idx": idx,
                "hostname": hostname,
                "engine": "sglang-torchtpu-v0.2.0",
                "name": cfg["name"],
                "model_short": cfg["model_short"],
                "is_fasth3": cfg["is_fasth3"],
                "precision": cfg["precision"],
                "dp_replicas": cfg["dp_replicas"],
                "num_gpus_per_replica": cfg["num_gpus_per_replica"],
                "ulysses_degree": cfg["ulysses_degree"],
                "chips": cfg["chips"],
                "all_hbm_resident": cfg["all_hbm_resident"],
                "w8a8_fp8_gemm": cfg["w8a8_fp8_gemm"],
                "torch_compile_requested": cfg["enable_torch_compile"],
                "torch_compile_active": compile_active,
                "compile_fallback_reason": fb_reason,
                "remote_encode_active": bool(remote_encode_url and cfg["use_remote_encode"]),
                "server_boot_s": router.boot_s,
            }
            wls = build_host_workloads(is_fasth3=cfg["is_fasth3"], dp_replicas=cfg["dp_replicas"])
            run_workloads_on_router(
                idx,
                bucket=bucket,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_meta=meta,
                router=router,
                hbm_mon=hbm_mon,
                workloads=wls,
                keyframes=keyframes,
                collected=collected_rows,
            )
            if cfg["name"] == "sglang_fasth3_fp8_dp2_u4_opt":
                dp2_fasth3_ok = any(
                    r.get("name") == "sglang_fasth3_fp8_dp2_u4_opt" and r.get("ok_count", 0) > 0
                    for r in collected_rows
                )
            hbm_mon.stop()
            router.stop()
        except Exception as exc:
            log(idx, f"ERROR in config [{cfg['name']}]: {exc}")
            err_row = {
                "phase": "3A-1_single_host",
                "run_id": run_id,
                "pod_idx": idx,
                "hostname": hostname,
                "name": cfg["name"],
                "model_short": cfg["model_short"],
                "is_fasth3": cfg["is_fasth3"],
                "precision": cfg["precision"],
                "dp_replicas": cfg["dp_replicas"],
                "ulysses_degree": cfg["ulysses_degree"],
                "chips": cfg["chips"],
                "workload_label": "server_boot_or_config",
                "denoise_steps": 4 if cfg["is_fasth3"] else 20,
                "concurrency": 1,
                "num_requests": 0,
                "ok_count": 0,
                "error": str(exc)[:1000],
            }
            collected_rows.append(err_row)
            append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, err_row)
            kill_sglang_processes()

    # Synchronize all 4 hosts before Phase 3A-2 16-chip slice sweep
    p1_info = gcs_barrier(
        idx,
        bucket,
        run_id,
        "phase3a1_done",
        {"pod_idx": idx, "hostname": hostname, "pod_ip": pod_ip, "dp2_fasth3_ok": dp2_fasth3_ok},
        num_pods=args.num_pods,
    )
    use_dp2_slice = bool(p1_info.get(3, {}).get("dp2_fasth3_ok", False))

    # === Phase 3A-2: Full 16-Chip `v7x-16` Slice Multi-Host Sweep (`FastH3` across all 4 pods) ===
    log(idx, f"=== Phase 3A-2: Staging FastH3 on all pods (use_dp2_slice={use_dp2_slice}) ===")
    fasth3_path = stage_fasth3_from_gcs(idx, bucket)
    slice_dp_per_host = 2 if use_dp2_slice else 1
    slice_gpus = 4 if use_dp2_slice else 8
    slice_u = 4 if use_dp2_slice else 8

    slice_router_local, _ = start_sglang_host_cluster(
        idx,
        config_name="suite3a_slice16_fasth3_fp8_opt",
        model_path=fasth3_path,
        dp_replicas=slice_dp_per_host,
        num_gpus_per_replica=slice_gpus,
        ulysses_degree=slice_u,
        quantization="fp8",
        all_hbm_resident=True,
        splash_varlen=True,
        w8a8_fp8_gemm=True,
        enable_torch_compile=False,
        compile_vae=True,
        remote_encode_url=remote_encode_url,
        base_port=30100,
    )

    ep_map = gcs_barrier(
        idx,
        bucket,
        run_id,
        "phase3a2_endpoints_ready",
        {
            "pod_idx": idx,
            "hostname": hostname,
            "pod_ip": pod_ip,
            "endpoints": slice_router_local.endpoints,
        },
        num_pods=args.num_pods,
    )

    if idx == 0:
        all_slice_eps: list[str] = []
        for p_k in sorted(ep_map.keys()):
            all_slice_eps.extend(ep_map[p_k].get("endpoints", []))
        active_hosts = len(ep_map)
        total_chips = active_hosts * 4
        slice_topo = f"DP={len(all_slice_eps)}xUlysses={slice_u}"
        log(idx, f"Driving 16-chip slice sweep across {len(all_slice_eps)} replicas ({total_chips} chips): {all_slice_eps}")

        slice_router = SGLangEndpointRouter(all_slice_eps)
        slice_meta = {
            "phase": "3A-2_slice16",
            "run_id": run_id,
            "pod_idx": 0,
            "hostname": hostname,
            "engine": "sglang-torchtpu-v0.2.0",
            "name": f"sglang_fasth3_fp8_slice16_{slice_topo.lower().replace('=', '')}",
            "slice_topology": slice_topo,
            "model_short": "FastH3-4step",
            "is_fasth3": True,
            "precision": "fp8_w8a8_v7x_bf16acc",
            "dp_replicas": len(all_slice_eps),
            "ulysses_degree": slice_u,
            "chips": total_chips,
            "remote_encode_active": bool(remote_encode_url),
        }
        slice_workloads = [
            {
                "label": "slice16_768p_16x9_5s_124f_4step_c4",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "num_inference_steps": 5,
                "concurrency": 4,
                "num_requests": 8,
            },
            {
                "label": "slice16_768p_16x9_5s_124f_4step_c8",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
                "num_inference_steps": 5,
                "concurrency": 8,
                "num_requests": 16,
            },
            {
                "label": "slice16_768p_16x9_15s_360f_4step_c4",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 15.0,
                "num_inference_steps": 5,
                "concurrency": 4,
                "num_requests": 8,
            },
            {
                "label": "slice16_768p_16x9_15s_360f_4step_c8",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 15.0,
                "num_inference_steps": 5,
                "concurrency": 8,
                "num_requests": 16,
            },
            {
                "label": "slice16_768p_mixed_5s_10s_15s_4step_c4",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 10.0,
                "mixed_durations": [5.0, 10.0, 15.0],
                "num_inference_steps": 5,
                "concurrency": 4,
                "num_requests": 12,
            },
            {
                "label": "slice16_768p_mixed_5s_10s_15s_4step_c8",
                "task": "t2va",
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 10.0,
                "mixed_durations": [5.0, 10.0, 15.0],
                "num_inference_steps": 5,
                "concurrency": 8,
                "num_requests": 16,
            },
        ]
        run_workloads_on_router(
            idx,
            bucket=bucket,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            config_meta=slice_meta,
            router=slice_router,
            hbm_mon=None,
            workloads=slice_workloads,
            keyframes=keyframes,
            collected=collected_rows,
        )

    gcs_barrier(
        idx,
        bucket,
        run_id,
        "phase3a2_complete",
        {"pod_idx": idx, "done": True},
        num_pods=args.num_pods,
    )
    slice_router_local.stop()

    if idx == 0:
        all_rows: list[dict[str, Any]] = []
        for p_k in range(args.num_pods):
            raw = gcs_download_bytes(bucket, f"v7x-video-opt/results/suite3a_pod{p_k}.jsonl")
            if raw:
                for ln in raw.decode("utf-8").splitlines():
                    if ln.strip():
                        try:
                            obj = json.loads(ln)
                            if obj.get("run_id") == run_id:
                                all_rows.append(obj)
                        except Exception:
                            pass
        host_rows = [r for r in all_rows if r.get("phase") == "3A-1_single_host"]
        slice_rows = [r for r in all_rows if r.get("phase") == "3A-2_slice16"]
        summary_obj = {
            "run_id": run_id,
            "updated_utc": utc_now(),
            "host_rows": host_rows,
            "slice_rows": slice_rows,
        }
        md_report = format_suite3a_markdown(run_id, host_rows, slice_rows)
        gcs_upload_bytes(
            bucket,
            f"v7x-video-opt/results/{run_id}/suite3a_summary.json",
            json.dumps(summary_obj, indent=2).encode("utf-8"),
        )
        gcs_upload_bytes(
            bucket,
            f"v7x-video-opt/results/{run_id}/suite3a_summary.md",
            md_report.encode("utf-8"),
            content_type="text/markdown",
        )
        log(idx, f"Uploaded Suite 3A consolidated summary ({len(host_rows)} host rows, {len(slice_rows)} slice rows).")

    log(idx, "=== Suite 3A COMPLETE ===")


if __name__ == "__main__":
    main()
