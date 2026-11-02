#!/usr/bin/env python3
"""Suite 3B: `vllm-omni` (`vllm-torchtpu-omni:v0.32.0`) MiniMax-H3 Enablement & Full 16-Chip `v7x-16` Slice Multi-Host Sweeps.

Executes a 3-Step Benchmark & Enablement Campaign across the 4-host `tpu-v7x-16-pool`
(`2x2x4` = 16 physical Cloud TPU `v7x` chips / 32 TensorCores / 3,072 GiB HBM3e):

  - **Step 1 (Simultaneous Per-Pod Enablement & Sub-Host Scaling Verification)**:
      * Pods 0–1 (`try/except`-isolated): `MiniMax-H3` (`FL2VA` 44B Video+Audio DiT) on `vllm-omni` with TPU `v7x` runtime patches (`_AudioVAEDeterminismContext`, `cudnn_sdp`, `int32` `chunked_decode`, `int32` `adaln_cache`) + Disaggregated CPU Video Encode (`VLLM_OMNI_REMOTE_ENCODE_URL`).
      * Pod 2: `LTX-2.5` single-host Disaggregated CPU Video Encode ablation (`local` vs `video-encode-svc`) + `DP=2 x Ulysses=4` (`2`x 2-chip servers on 1 host).
      * Pod 3: `LTX-2.5` single-host `DP=4 x Ulysses=2` (`4`x 1-chip servers on 1 host) sub-host VFIO scaling verification.

  - **Step 2 (Full 16-Chip `v7x-16` Slice Multi-Host `LTX-2.5` Throughput Sweeps)**:
      * Synchronized 4-host `LTX-2.5` (`Lightricks/LTX-2.5` 22B Video+Audio DiT) sweeps comparing:
          1. `DP=4 x Ulysses=8` (`4` replicas x `8` TensorCores = `32` cores) at `c=4, 8`
          2. `DP=8 x Ulysses=4` (`8` replicas x `4` TensorCores = `32` cores) at `c=4, 8, 16`
          3. `DP=16 x Ulysses=2` (`16` replicas x `2` TensorCores = `32` cores, if 1-chip `Ulysses=2` verified in Step 1) at `c=8, 16, 32`
        across `544p 5s (121f)`, `720p 5s (121f)`, `1080p 5s (121f)`, and **`1080p 20s (481f)` Kling-V4 Proxy**.

  - **Step 3 (Full 16-Chip `v7x-16` Slice Multi-Host `Wan 2.2 I2V-A14B` Throughput Sweeps)**:
      * Clean `/dev/shm` swap to `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (28.6B Total / 14.3B Active MoE DiT) across all 4 hosts:
          1. `DP=4 x Ulysses=8` at `c=4, 8`
          2. `DP=8 x Ulysses=4` at `c=4, 8, 16`
        across `480p 5s (81f) 4-step`, `720p 5s (81f) 4-step`, `720p 5s (81f) 20-step`, and `720p 10s (161f) 20-step`.

Incremental JSONL persistence after EVERY workload:
  `gs://<bucket>/v7x-video-opt/results/suite3b_pod{idx}.jsonl`
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_cache/hub"

import argparse
import concurrent.futures
import json
import math
import py_compile
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
import uuid
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
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic sea spray and gull calls.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, crisp tire hiss and city ambience.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, soft footsteps crunching in snow.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist, gentle mountain wind and distant stream.",
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

PATCHED_SET_DEVICE_METHOD = """    @classmethod
    def set_device(cls, device: torch.device | int) -> None:
        # Sub-host VFIO rank pinning for multi-replica TPU v7x serving (VLLM_OMNI_LOCAL_RANKS / TPU_VISIBLE_CHIPS)
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        raw_lrs = os.environ.get("VLLM_OMNI_LOCAL_RANKS", "").strip()
        raw_vis = os.environ.get("TPU_VISIBLE_CHIPS", "").strip()
        ranks: list[str] = []
        if raw_lrs:
            ranks = [x.strip() for x in raw_lrs.split(",") if x.strip()]
        elif raw_vis:
            vis_list = [x.strip() for x in raw_vis.split(",") if x.strip()]
            if len(vis_list) == world_size and [int(c) for c in vis_list] != list(range(world_size)):
                ranks = vis_list

        phys_rank: str | None = None
        sub_topo: str | None = None
        custom_sb = os.environ.get("VLLM_OMNI_SLICEBUILDER_ADDRESSES", "").strip()

        if ranks:
            for _k in (
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
                "TPU_WORKER_ID",
                "TPU_WORKER_HOSTNAMES",
                "CLOUD_TPU_TASK_ID",
            ):
                os.environ.pop(_k, None)
            w_rank = int(os.environ.get("RANK", "0") or "0")
            phys_rank = ranks[w_rank % len(ranks)]
            vllm_topo = os.environ.get("VLLM_OMNI_TOPOLOGY", "").strip()
            if vllm_topo:
                sub_topo = vllm_topo
            elif world_size == 1:
                sub_topo = ""
            elif world_size == 2:
                sub_topo = "1,1,1,2"
            elif world_size == 4:
                sub_topo = "1,2,1,2"
            else:
                sub_topo = os.environ.get("TORCH_TPU_TOPOLOGY", "2,2,1,2")

            os.environ["LOCAL_RANK"] = phys_rank
            if sub_topo:
                os.environ["TORCH_TPU_TOPOLOGY"] = sub_topo
            else:
                os.environ.pop("TORCH_TPU_TOPOLOGY", None)

        if world_size > 1:
            cls._worker_world_size = world_size
            if custom_sb:
                os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb
            else:
                cls._set_shared_slicebuilder_addresses(world_size)
            cls._offload_dense_all_to_all()
        elif custom_sb:
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb

        cls._prepare_singlehost_tpu_env(world_size)

        if phys_rank is not None:
            os.environ["LOCAL_RANK"] = phys_rank
            os.environ["TPU_VISIBLE_CHIPS"] = phys_rank
            if sub_topo:
                os.environ["TORCH_TPU_TOPOLOGY"] = sub_topo
            else:
                os.environ.pop("TORCH_TPU_TOPOLOGY", None)
        if custom_sb:
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = custom_sb

        torch.accelerator.set_device_index(0)
        _ = torch.empty(1, device=torch.device("tpu"))
        torch.accelerator.synchronize()
"""

_MOCK_META_PROC: subprocess.Popen[Any] | None = None
_SB_PORT_CURSOR = 31400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][suite3b-pod-{idx}] {msg}", flush=True)


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
    log(idx, f"WARNING: {health_url} not reachable; falling back to local CPU encode.")
    return ""


def _resolve_package_dir(pkg_name: str, fallbacks: list[Path]) -> Path | None:
    try:
        out = subprocess.check_output(
            ["python3", "-c", f"import {pkg_name}, pathlib; print(pathlib.Path({pkg_name}.__file__).resolve().parent)"],
            text=True,
        ).strip()
        if out and Path(out).exists():
            return Path(out)
    except Exception:
        pass
    for fb in fallbacks:
        if fb.exists():
            return fb
    return None


def ensure_subhost_and_omni_patches(idx: int) -> None:
    """Compile metadata_hook.so, start mock_metadata.py, patch OmniTpuPlatform.set_device, MiniMax-H3 TPU guards, and Disaggregated Video Encode."""
    global _MOCK_META_PROC
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    for dst_dir in (Path("/tmp"), Path("/workspace")):
        hook_c = dst_dir / "metadata_hook.c"
        hook_so = dst_dir / "metadata_hook.so"
        if not hook_so.exists():
            hook_c.write_text(METADATA_HOOK_C)
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"], check=True)
    subprocess.run(
        ["bash", "-lc", "grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts"],
        check=False,
    )

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

    # 1. Patch OmniTpuPlatform.set_device for sub-host VFIO rank binding (DP=2xU=4 and DP=4xU=2)
    vt_dir = _resolve_package_dir(
        "vllm_torchtpu",
        [
            Path("/workspace/vllm-torchtpu/src/vllm_torchtpu"),
            Path("/workspace/vllm-torchtpu/vllm_torchtpu"),
            Path("/usr/local/lib/python3.12/site-packages/vllm_torchtpu"),
        ],
    )
    if vt_dir is not None:
        plat_p = vt_dir / "omni/platform.py"
        if plat_p.exists():
            txt = plat_p.read_text()
            if "VLLM_OMNI_LOCAL_RANKS" not in txt:
                start_marker = "    @classmethod\n    def set_device(cls, device: torch.device | int) -> None:\n"
                end_marker = "\n    @staticmethod\n    def _offload_dense_all_to_all() -> None:"
                if start_marker in txt and end_marker in txt:
                    pre, rest = txt.split(start_marker, 1)
                    _, post = rest.split(end_marker, 1)
                    plat_p.write_text(pre + PATCHED_SET_DEVICE_METHOD + end_marker + post)
                    py_compile.compile(str(plat_p), doraise=True)
                    log(idx, f"Patched OmniTpuPlatform.set_device in {plat_p}")

    # 2. Patch vllm_omni MiniMax-H3 TPU guards + Disaggregated CPU Video Encode
    vo_dir = _resolve_package_dir(
        "vllm_omni",
        [
            Path("/workspace/vllm-omni/vllm_omni"),
            Path("/usr/local/lib/python3.12/site-packages/vllm_omni"),
        ],
    )
    if vo_dir is None:
        raise RuntimeError("Could not locate vllm_omni package root in container!")

    mm_dir = vo_dir / "diffusion/models/minimax_h3"
    if mm_dir.exists():
        # 2a. vae.py: guard _AudioVAEDeterminismContext when CUDA is unavailable
        p_vae = mm_dir / "vae.py"
        if p_vae.exists():
            t_vae = p_vae.read_text()
            if "_tpu_audio_det_guarded" not in t_vae:
                old_enter = "    def __enter__(self):\n"
                new_enter = (
                    "    def __enter__(self):\n"
                    "        # _tpu_audio_det_guarded\n"
                    "        if not torch.cuda.is_available():\n"
                    "            return self\n"
                )
                old_exit = "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                new_exit = (
                    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
                    "        if not torch.cuda.is_available():\n"
                    "            return False\n"
                )
                if old_enter in t_vae:
                    t_vae = t_vae.replace(old_enter, new_enter, 1)
                if old_exit in t_vae:
                    t_vae = t_vae.replace(old_exit, new_exit, 1)
                p_vae.write_text(t_vae)
                py_compile.compile(str(p_vae), doraise=True)

        # 2b. encoder.py: guard torch.backends.cuda.cudnn_sdp_enabled / enable_cudnn_sdp
        p_enc = mm_dir / "encoder.py"
        if p_enc.exists():
            t_enc = p_enc.read_text()
            old_sdp1 = (
                "        cudnn_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled()\n"
                "        torch.backends.cuda.enable_cudnn_sdp(True)"
            )
            new_sdp1 = (
                "        cudnn_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled() if torch.cuda.is_available() else False\n"
                "        if torch.cuda.is_available():\n"
                "            torch.backends.cuda.enable_cudnn_sdp(True)"
            )
            if old_sdp1 in t_enc:
                t_enc = t_enc.replace(old_sdp1, new_sdp1)
            old_sdp2 = "            torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_enabled)"
            new_sdp2 = (
                "            if torch.cuda.is_available():\n"
                "                torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_enabled)"
            )
            if old_sdp2 in t_enc and "if torch.cuda.is_available():\n                torch.backends.cuda.enable_cudnn_sdp" not in t_enc:
                t_enc = t_enc.replace(old_sdp2, new_sdp2)
            p_enc.write_text(t_enc)
            py_compile.compile(str(p_enc), doraise=True)

        # 2c. chunked_decode.py: int32 index_select indices on TPU
        p_cd = mm_dir / "chunked_decode.py"
        if p_cd.exists():
            t_cd = p_cd.read_text()
            old_cd = "dtype=torch.int64, device=latent.device"
            new_cd = 'dtype=torch.int32 if latent.device.type == "tpu" else torch.int64, device=latent.device'
            if old_cd in t_cd:
                t_cd = t_cd.replace(old_cd, new_cd)
                p_cd.write_text(t_cd)
                py_compile.compile(str(p_cd), doraise=True)

        # 2d. adaln_cache.py: CPU safe_open + int32 slot indices on TPU
        p_ac = mm_dir / "adaln_cache.py"
        if p_ac.exists():
            t_ac = p_ac.read_text()
            if 'safe_open(f, framework="pt", device=str(device))' in t_ac:
                t_ac = t_ac.replace(
                    'safe_open(f, framework="pt", device=str(device))',
                    'safe_open(f, framework="pt", device="cpu")',
                )
            if "dtype=torch.int64, device=self.block_params.device" in t_ac:
                t_ac = t_ac.replace(
                    "dtype=torch.int64, device=self.block_params.device",
                    'dtype=torch.int32 if self.block_params.device.type == "tpu" else torch.int64, device=self.block_params.device',
                )
            p_ac.write_text(t_ac)
            py_compile.compile(str(p_ac), doraise=True)

    # 3. entrypoints/openai/video_api_utils.py: Disaggregated CPU Video + Audio MP4 Encoding
    p_vapi = vo_dir / "entrypoints/openai/video_api_utils.py"
    if p_vapi.exists():
        t_vapi = p_vapi.read_text()
        if "_omni_remote_encode_wrapped" not in t_vapi:
            t_vapi += r"""

# _omni_remote_encode_wrapped: Offload post-VAE RGB24 + f32le AAC MP4 encoding to video-encode-svc
if "_encode_video_bytes" in globals():
    import inspect as _inspect
    _orig_encode_video_bytes = _encode_video_bytes
    _orig_sig = _inspect.signature(_orig_encode_video_bytes)

    def _encode_video_bytes(*args, **kwargs) -> bytes:
        import json as _json
        import os as _os
        import struct as _struct
        import urllib.request as _ureq
        import numpy as _np

        remote_url = _os.environ.get("VLLM_OMNI_REMOTE_ENCODE_URL", "").strip()
        if remote_url:
            try:
                bound = _orig_sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                bargs = bound.arguments
                raw_vid = bargs.get("video") if "video" in bargs else bargs.get("frames")
                fps_val = float(bargs.get("fps") or 24.0)
                raw_aud = bargs.get("audio")
                aud_sr_in = bargs.get("audio_sample_rate") or bargs.get("sample_rate") or 24000

                arr = None
                if raw_vid is not None:
                    if hasattr(raw_vid, "detach"):
                        t_cpu = raw_vid.detach().cpu()
                        if t_cpu.dtype not in (getattr(__import__("torch"), "uint8"),):
                            t_np = t_cpu.float().numpy()
                        else:
                            t_np = t_cpu.numpy()
                    elif isinstance(raw_vid, _np.ndarray):
                        t_np = raw_vid
                    elif isinstance(raw_vid, (list, tuple)) and len(raw_vid) > 0:
                        elems = []
                        for fr in raw_vid:
                            if hasattr(fr, "detach"):
                                elems.append(fr.detach().cpu().float().numpy())
                            else:
                                elems.append(_np.asarray(fr))
                        t_np = _np.stack(elems, axis=0)
                    else:
                        t_np = None

                    if t_np is not None:
                        if t_np.ndim == 5 and t_np.shape[0] == 1:
                            t_np = t_np[0]
                        if t_np.ndim == 4 and t_np.shape[1] == 3 and t_np.shape[-1] != 3:
                            t_np = _np.transpose(t_np, (0, 2, 3, 1))
                        elif t_np.ndim == 4 and t_np.shape[0] == 3 and t_np.shape[-1] != 3:
                            t_np = _np.transpose(t_np, (1, 2, 3, 0))
                        if t_np.dtype != _np.uint8:
                            mn = float(t_np.min())
                            mx = float(t_np.max())
                            if mn < -0.05:
                                t_np = ((t_np + 1.0) * 0.5 * 255.0).clip(0.0, 255.0).astype(_np.uint8)
                            elif mx <= 1.05:
                                t_np = (t_np * 255.0).clip(0.0, 255.0).astype(_np.uint8)
                            else:
                                t_np = t_np.clip(0.0, 255.0).astype(_np.uint8)
                        if t_np.ndim == 4 and t_np.shape[-1] == 3:
                            arr = _np.ascontiguousarray(t_np)

                if arr is not None:
                    nf, h, w, _ = arr.shape
                    vid_bytes = arr.tobytes()

                    aud_bytes = b""
                    aud_ch = 2
                    aud_sr = 0
                    if raw_aud is not None:
                        if isinstance(raw_aud, dict) and "waveform" in raw_aud:
                            aud_sr_in = raw_aud.get("sample_rate") or aud_sr_in
                            raw_aud = raw_aud["waveform"]
                        if hasattr(raw_aud, "detach"):
                            a_np = raw_aud.detach().cpu().float().numpy()
                        else:
                            a_np = _np.asarray(raw_aud, dtype=_np.float32)
                        if a_np.size > 0:
                            if a_np.ndim == 3 and a_np.shape[0] == 1:
                                a_np = a_np[0]
                            if a_np.ndim == 1:
                                aud_ch = 1
                                a_np = _np.ascontiguousarray(a_np, dtype=_np.float32)
                            elif a_np.ndim == 2:
                                if a_np.shape[0] <= 8 and a_np.shape[1] > a_np.shape[0]:
                                    a_np = a_np.T
                                aud_ch = int(a_np.shape[1])
                                a_np = _np.ascontiguousarray(a_np, dtype=_np.float32)
                            aud_bytes = a_np.tobytes()
                            aud_sr = int(aud_sr_in or 24000)

                    meta = {
                        "width": int(w),
                        "height": int(h),
                        "num_frames": int(nf),
                        "fps": fps_val,
                        "pix_fmt": "rgb24",
                        "preset": "ultrafast",
                        "crf": 20,
                        "threads": 16,
                        "audio_sr": aud_sr if len(aud_bytes) > 0 else 0,
                        "audio_channels": aud_ch,
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
                    print(
                        f"[omni_remote_encode] {nf}f {w}x{h} -> {len(mp4_bytes)}B in {enc_ms}ms ({enc_fps} fps) on {enc_host}",
                        flush=True,
                    )
                    return mp4_bytes
            except Exception as _exc:
                print(f"[omni_remote_encode] fallback to local encode after error: {_exc}", flush=True)

        return _orig_encode_video_bytes(*args, **kwargs)
"""
            p_vapi.write_text(t_vapi)
            py_compile.compile(str(p_vapi), doraise=True)
            log(idx, f"Patched {p_vapi} for Disaggregated CPU Video + Audio MP4 Encoding")

    log(idx, "All vllm-omni TPU v7x sub-host, MiniMax-H3, and Disaggregated Encode patches verified & applied!")


def purge_other_shm_models(idx: int, keep_name: str) -> None:
    """Strictly maintain ONLY ONE large video checkpoint in /dev/shm at a time."""
    for name in (
        "MiniMax-H3",
        "FastH3",
        "LTX-2.5",
        "Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI",
        "hf_cache",
    ):
        if name == keep_name:
            continue
        shutil.rmtree(f"/dev/shm/{name}", ignore_errors=True)
    log(idx, f"Cleaned /dev/shm (keeping={keep_name}, shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")


def stage_minimax_for_omni(idx: int, bucket: str) -> str:
    purge_other_shm_models(idx, keep_name="MiniMax-H3")
    base_dir = Path("/dev/shm/MiniMax-H3")
    fl2va_dir = base_dir / "FL2VA"
    if not (fl2va_dir / "model_index.json").exists():
        synced = 0
        for pref in (
            "models/MiniMaxAI/MiniMax-H3/FL2VA/",
            "hf-cache/MiniMaxAI--MiniMax-H3/FL2VA/",
            "models/MiniMax-H3/FL2VA/",
        ):
            synced = fast_gcs_sync_dir(idx, bucket, pref, fl2va_dir)
            if synced > 0:
                break
        if synced == 0:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="MiniMaxAI/MiniMax-H3",
                local_dir=str(base_dir),
                allow_patterns=["FL2VA/**"],
                token=os.environ.get("HF_TOKEN") or None,
                max_workers=16,
            )

    if not (fl2va_dir / "vae").exists() and (fl2va_dir / "video_vae").exists():
        (fl2va_dir / "vae").symlink_to("video_vae")
    v_sf = fl2va_dir / "video_vae" / "model.safetensors"
    v_src = fl2va_dir / "video_vae" / "source" / "model.safetensors"
    if not v_sf.exists() and v_src.exists():
        v_sf.symlink_to("source/model.safetensors")

    # Expose zero-cost symlinks at /dev/shm/MiniMax-H3 root AND keep /dev/shm/MiniMax-H3/FL2VA intact
    for item in fl2va_dir.iterdir():
        root_link = base_dir / item.name
        if not root_link.exists():
            root_link.symlink_to(f"FL2VA/{item.name}")
    log(idx, f"MiniMax-H3 staged at {fl2va_dir} (shm_free={shm_free_gib()} GiB)")
    return str(fl2va_dir)


def stage_ltx25_for_omni(idx: int, bucket: str) -> str:
    purge_other_shm_models(idx, keep_name="LTX-2.5")
    local_dir = Path("/dev/shm/LTX-2.5")
    if (local_dir / "model_index.json").exists():
        log(idx, f"LTX-2.5 already staged at {local_dir} (shm_free={shm_free_gib()} GiB)")
        return str(local_dir)

    synced = 0
    for pref in (
        "models/Lightricks/LTX-2.5/",
        "models/Lightricks/LTX-2.5-Diffusers/",
        "hf-cache/Lightricks--LTX-2.5/",
        "models/LTX-2.5/",
    ):
        synced = fast_gcs_sync_dir(idx, bucket, pref, local_dir)
        if synced > 0 and (local_dir / "model_index.json").exists():
            break

    if not (local_dir / "model_index.json").exists():
        from huggingface_hub import snapshot_download

        log(idx, "Downloading Lightricks/LTX-2.5 via snapshot_download into /dev/shm/LTX-2.5 ...")
        snapshot_download(
            repo_id="Lightricks/LTX-2.5",
            local_dir=str(local_dir),
            ignore_patterns=["*.mp4", "*.mov", "*.gif"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )
    log(idx, f"LTX-2.5 ready at {local_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return str(local_dir)


def stage_wan22_i2v_for_omni(idx: int, bucket: str) -> str:
    purge_other_shm_models(idx, keep_name="Wan2.2-I2V-A14B-Diffusers")
    local_dir = Path("/dev/shm/Wan2.2-I2V-A14B-Diffusers")
    if (local_dir / "model_index.json").exists():
        log(idx, f"Wan2.2-I2V-A14B-Diffusers already staged at {local_dir} (shm_free={shm_free_gib()} GiB)")
        return str(local_dir)

    synced = 0
    for pref in (
        "models/Wan-AI/Wan2.2-I2V-A14B-Diffusers/",
        "hf-cache/Wan-AI--Wan2.2-I2V-A14B-Diffusers/",
        "models/Wan2.2-I2V-A14B-Diffusers/",
    ):
        synced = fast_gcs_sync_dir(idx, bucket, pref, local_dir)
        if synced > 0 and (local_dir / "model_index.json").exists():
            break

    if not (local_dir / "model_index.json").exists():
        from huggingface_hub import snapshot_download

        log(idx, "Downloading Wan-AI/Wan2.2-I2V-A14B-Diffusers via snapshot_download...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            local_dir=str(local_dir),
            ignore_patterns=["*.mp4", "*.mov", "*.gif"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )
    log(idx, f"Wan2.2-I2V-A14B-Diffusers ready at {local_dir} (shm_free={shm_free_gib()} GiB, RAM={host_ram_summary()})")
    return str(local_dir)


def generate_reference_images(input_dir: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "832x480": (832, 480),
        "1280x720": (1280, 720),
        "896x512": (896, 512),
        "1344x768": (1344, 768),
    }
    out: dict[str, Path] = {}
    for size_s, (w, h) in specs.items():
        p = input_dir / f"ref_{size_s}.png"
        img = Image.new("RGB", (w, h), color=(28, 48, 76))
        draw = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(35 + 150 * (y / max(h, 1))))
            g = min(255, int(60 + 105 * (y / max(h, 1))))
            b = min(255, int(100 + 75 * (1.0 - y / max(h, 1))))
            draw.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        draw.ellipse([w // 3, h // 5, w // 3 + 120, h // 5 + 120], fill=(250, 214, 132))
        img.save(p, format="PNG")
        out[size_s] = p
    return out


class LibtpuHbmMonitor:
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
                                if val > 1024 * 1024 and val > self.peak_bytes_per_core.get(c_i, 0.0):
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


def stop_omni_cluster_cleanly(procs: list[subprocess.Popen[Any]] | None = None) -> None:
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
            "pkill -15 -f '[v]llm serve' 2>/dev/null || true; "
            "pkill -15 -f '[v]llm_omni' 2>/dev/null || true; "
            "pkill -15 -f '[D]iffusionWorker' 2>/dev/null || true; "
            "pkill -15 -f '[S]tageDiffusionProc' 2>/dev/null || true; "
            "sleep 1.0; "
            "pkill -9 -f '[v]llm serve' 2>/dev/null || true; "
            "pkill -9 -f '[v]llm_omni' 2>/dev/null || true; "
            "pkill -9 -f '[D]iffusionWorker' 2>/dev/null || true; "
            "pkill -9 -f '[S]tageDiffusionProc' 2>/dev/null || true; "
            "pkill -9 -f '[m]ultiprocessing.spawn' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/vllm_omni_* /dev/shm/psm_* 2>/dev/null || true",
        ],
        check=False,
    )

    t_vfio = time.time()
    while time.time() - t_vfio < 20.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(1.0)
    time.sleep(1.0)


def extract_clean_error_tail(log_path: Path, max_chars: int = 3000) -> str:
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


def _write_video_deploy_yaml(
    cfg_path: Path,
    *,
    ulysses_degree: int,
    quantization: str | None,
) -> None:
    stage_cfg: dict[str, Any] = {
        "stage_id": 0,
        "ulysses_degree": ulysses_degree,
        "max_num_seqs": 1,
        "enable_diffusion_pipeline_profiler": True,
        "devices": ",".join(str(i) for i in range(ulysses_degree)),
        "engine_extras": {
            "num_gpus": ulysses_degree,
        },
    }
    deploy_cfg: dict[str, Any] = {"stages": [stage_cfg]}
    if quantization:
        deploy_cfg["quantization"] = quantization
    cfg_path.write_text(json.dumps(deploy_cfg, indent=2) + "\n")


def _encode_multipart_formdata(
    fields: dict[str, str],
    file_part: tuple[str, str, bytes, str] | None = None,
) -> tuple[bytes, str]:
    boundary = f"----Suite3BBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{v}\r\n".encode("utf-8"))
    if file_part is not None:
        field_name, filename, content_bytes, mime_type = file_part
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        chunks.append(content_bytes)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def generate_one_video_omni(
    endpoint: str,
    *,
    prompt: str,
    size: str,
    num_frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    ref_image_path: Path | None = None,
    extra_params: dict[str, Any] | None = None,
    probe_streams: bool = False,
    timeout_s: float = 2400.0,
) -> dict[str, Any]:
    fields: dict[str, str] = {
        "prompt": prompt,
        "size": size,
        "num_frames": str(num_frames),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "seed": str(seed),
    }
    if extra_params:
        fields["extra_params"] = json.dumps(extra_params)

    file_part = None
    if ref_image_path is not None and ref_image_path.exists():
        file_part = ("input_reference", ref_image_path.name, ref_image_path.read_bytes(), "image/png")

    body, ctype = _encode_multipart_formdata(fields, file_part=file_part)
    req = urllib.request.Request(
        f"{endpoint}/v1/videos/sync",
        data=body,
        method="POST",
        headers={"Content-Type": ctype},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            mp4_bytes = resp.read()
            e2e_s = time.perf_counter() - t0
            inf_hdr = resp.headers.get("X-Inference-Time-S")
            req_id = resp.headers.get("X-Request-Id", "")
            srv_inf_s = float(inf_hdr) if inf_hdr is not None else None

        valid_mp4 = len(mp4_bytes) >= 1024 and (b"ftyp" in mp4_bytes[:64])
        ffprobe_streams = None
        if valid_mp4 and probe_streams:
            tmp_mp4 = Path(f"/dev/shm/probe_{uuid.uuid4().hex[:8]}.mp4")
            try:
                tmp_mp4.write_bytes(mp4_bytes)
                pr = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "stream=codec_type,codec_name,width,height,nb_frames,sample_rate,channels",
                        "-of",
                        "json",
                        str(tmp_mp4),
                    ],
                    capture_output=True,
                    text=True,
                )
                if pr.returncode == 0:
                    ffprobe_streams = json.loads(pr.stdout).get("streams")
            except Exception:
                pass
            finally:
                try:
                    tmp_mp4.unlink(missing_ok=True)
                except Exception:
                    pass

        return {
            "ok": valid_mp4,
            "request_id": req_id,
            "e2e_s": round(e2e_s, 3),
            "inference_time_s": round(srv_inf_s, 3) if srv_inf_s is not None else None,
            "mp4_bytes": len(mp4_bytes),
            "ffprobe_streams": ffprobe_streams,
            "error": None if valid_mp4 else f"Invalid MP4 payload len={len(mp4_bytes)}",
        }
    except urllib.error.HTTPError as he:
        err_txt = ""
        try:
            err_txt = he.read().decode("utf-8", errors="ignore")[:800]
        except Exception:
            pass
        return {
            "ok": False,
            "e2e_s": round(time.perf_counter() - t0, 3),
            "error": f"HTTP {he.code}: {err_txt}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "e2e_s": round(time.perf_counter() - t0, 3),
            "error": repr(exc)[:800],
        }


class VllmOmniVideoRouter:
    """Least-inflight load balancer across 1..16 vllm-omni video endpoints (single-host or 4-host 16-chip slice)."""

    def __init__(
        self,
        endpoints: list[str],
        procs: list[subprocess.Popen[Any]] | None = None,
        log_paths: list[Path] | None = None,
        boot_s: float = 0.0,
    ) -> None:
        self.endpoints = list(endpoints)
        self.procs = procs or []
        self.log_paths = log_paths or []
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

    def submit(self, req_kwargs: dict[str, Any]) -> dict[str, Any]:
        slot, ep = self.acquire()
        try:
            return generate_one_video_omni(ep, **req_kwargs)
        finally:
            self.release(slot)

    def stop(self) -> None:
        stop_omni_cluster_cleanly(self.procs)


def start_vllm_omni_host_cluster(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    dp_replicas_per_host: int,
    ulysses_degree: int,
    quantization: str | None,
    remote_encode_url: str,
    base_port: int = 8000,
    boot_timeout_s: float = 1500.0,
) -> VllmOmniVideoRouter:
    """Launch 1 (U=8), 2 (DP=2 x U=4), or 4 (DP=4 x U=2) isolated vllm-omni video servers on this host."""
    global _SB_PORT_CURSOR
    stop_omni_cluster_cleanly()

    if dp_replicas_per_host == 1 and ulysses_degree == 8:
        rank_groups = ["0,1,2,3,4,5,6,7"]
        sub_topo = "2,2,1,2"
        metric_groups = ["8431,8432,8433,8434,8435,8436,8437,8438"]
    elif dp_replicas_per_host == 2 and ulysses_degree == 4:
        rank_groups = ["0,1,2,3", "4,5,6,7"]
        sub_topo = "1,2,1,2"
        metric_groups = ["8431,8432,8433,8434", "8435,8436,8437,8438"]
    elif dp_replicas_per_host == 4 and ulysses_degree == 2:
        rank_groups = ["0,1", "2,3", "4,5", "6,7"]
        sub_topo = "1,1,1,2"
        metric_groups = ["8431,8432", "8433,8434", "8435,8436", "8437,8438"]
    else:
        raise ValueError(f"Unsupported per-host layout: dp={dp_replicas_per_host}, ulysses={ulysses_degree}")

    procs: list[subprocess.Popen[Any]] = []
    ports: list[int] = []
    log_paths: list[Path] = []
    t0 = time.time()

    for r_i in range(dp_replicas_per_host):
        port = base_port + r_i
        sb_base = _SB_PORT_CURSOR
        _SB_PORT_CURSOR += 20
        sb_addrs = ",".join(f"127.0.0.1:{sb_base + k}" for k in range(ulysses_degree))

        cfg_p = Path(f"/tmp/deploy_{config_name}_r{r_i}.yaml")
        _write_video_deploy_yaml(cfg_p, ulysses_degree=ulysses_degree, quantization=quantization)
        lp = Path(f"/workspace/omni_{config_name}_r{r_i}.log")

        env = os.environ.copy()
        env.update(
            {
                "PJRT_DEVICE": "TPU",
                "TPU_SKIP_MDS_QUERY": "1",
                "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
                "JAX_COMPILATION_CACHE_DIR": f"/dev/shm/xla_cache_omni_r{r_i}",
                "VLLM_DISABLE_LOG_LOGO": "1",
                "TPU_VISIBLE_CHIPS": "0,1,2,3,4,5,6,7",
                "VLLM_OMNI_LOCAL_RANKS": rank_groups[r_i],
                "VLLM_OMNI_TOPOLOGY": sub_topo,
                "VLLM_OMNI_SLICEBUILDER_ADDRESSES": sb_addrs,
                "TPU_RUNTIME_METRICS_PORTS": metric_groups[r_i],
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(sb_base - 1),
                "GCE_METADATA_HOST": "127.0.0.1:80",
                "GCE_METADATA_IP": "127.0.0.1",
                "LD_PRELOAD": "/tmp/metadata_hook.so",
                "HF_HOME": "/dev/shm/hf_cache",
                "HF_HUB_CACHE": "/dev/shm/hf_cache/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_OMNI_REMOTE_ENCODE_URL": remote_encode_url,
                "PYTHONUNBUFFERED": "1",
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
            "--deploy-config",
            str(cfg_p),
        ]
        log(
            idx,
            f"Launching [{config_name}] rep={r_i}/{dp_replicas_per_host} on :{port} "
            f"(ranks={rank_groups[r_i]}, topo={sub_topo}, quant={quantization}, remote_enc={bool(remote_encode_url)})",
        )
        lf = open(lp, "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True))
        ports.append(port)
        log_paths.append(lp)

    ready = [False] * dp_replicas_per_host
    while time.time() - t0 < boot_timeout_s:
        for r_i, (proc, port, lp) in enumerate(zip(procs, ports, log_paths)):
            if proc.poll() is not None:
                tail = extract_clean_error_tail(lp)
                stop_omni_cluster_cleanly(procs)
                raise RuntimeError(f"[{config_name}] rep {r_i} exited early rc={proc.returncode}:\n{tail}")
            if ready[r_i]:
                continue
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as resp:
                    if resp.status == 200:
                        ready[r_i] = True
                        log(idx, f"[{config_name}] rep {r_i} HEALTHY on :{port} in {time.time() - t0:.1f}s")
            except Exception:
                pass
        if all(ready):
            boot_s = round(time.time() - t0, 2)
            pod_ip = local_pod_ip()
            endpoints = [f"http://{pod_ip}:{p}" for p in ports]
            return VllmOmniVideoRouter(endpoints, procs=procs, log_paths=log_paths, boot_s=boot_s)
        time.sleep(3.0)

    tails = "\n---\n".join(extract_clean_error_tail(lp, 1800) for lp in log_paths)
    stop_omni_cluster_cleanly(procs)
    raise TimeoutError(f"[{config_name}] timed out after {boot_timeout_s}s:\n{tails}")


def run_omni_workloads(
    idx: int,
    *,
    bucket: str,
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    config_meta: dict[str, Any],
    router: VllmOmniVideoRouter,
    hbm_mon: LibtpuHbmMonitor | None,
    workloads: list[dict[str, Any]],
    ref_images: dict[str, Path],
    collected: list[dict[str, Any]],
) -> None:
    warmed_shapes: set[tuple[str, int, int, bool]] = set()
    probed_once = False

    for w_i, wl in enumerate(workloads):
        size_s = str(wl["size"])
        num_frames = int(wl["num_frames"])
        fps = int(wl["fps"])
        steps = int(wl["steps"])
        gs = float(wl["guidance_scale"])
        conc = int(wl.get("concurrency", 1))
        n_reqs = int(wl.get("num_requests", max(2, conc)))
        use_i2v = bool(wl.get("use_i2v", False))
        extra_params = wl.get("extra_params")
        dur_s = round((num_frames - 1) / max(fps, 1), 2) if num_frames % fps == 1 else round(num_frames / max(fps, 1), 2)
        ref_img = ref_images.get(size_s) if use_i2v else None

        shape_key = (size_s, num_frames, fps, use_i2v)
        cold_compile_s = None
        sample_streams = None

        if shape_key not in warmed_shapes:
            warm_steps = min(steps, 4)
            log(idx, f"[{config_meta['name']}] Warming shape {shape_key} across {len(router.endpoints)} replica(s)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(router.endpoints)) as wpool:
                w_futs = [
                    wpool.submit(
                        generate_one_video_omni,
                        ep,
                        prompt=PROMPTS[ep_i % len(PROMPTS)],
                        size=size_s,
                        num_frames=num_frames,
                        fps=fps,
                        steps=warm_steps,
                        guidance_scale=gs,
                        seed=300 + w_i * 20 + ep_i,
                        ref_image_path=ref_img,
                        extra_params=extra_params,
                        probe_streams=(ep_i == 0 and not probed_once),
                    )
                    for ep_i, ep in enumerate(router.endpoints)
                ]
                w_outs = [f.result() for f in w_futs]
            cold_compile_s = max((o["e2e_s"] for o in w_outs), default=None)
            if w_outs and w_outs[0].get("ffprobe_streams"):
                sample_streams = w_outs[0]["ffprobe_streams"]
                probed_once = True
            if all(o.get("ok") for o in w_outs):
                warmed_shapes.add(shape_key)
            else:
                err_msg = next((o.get("error") for o in w_outs if not o.get("ok")), "warmup_failed")
                row = {
                    **config_meta,
                    "timestamp_utc": utc_now(),
                    "workload_label": wl["label"],
                    "size": size_s,
                    "num_frames": num_frames,
                    "fps": fps,
                    "duration_seconds": dur_s,
                    "denoise_steps": steps,
                    "guidance_scale": gs,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": 0,
                    "cold_compile_e2e_s": cold_compile_s,
                    "error": err_msg,
                }
                collected.append(row)
                append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, row)
                continue

        req_list = [
            {
                "prompt": PROMPTS[(w_i + r_i) % len(PROMPTS)],
                "size": size_s,
                "num_frames": num_frames,
                "fps": fps,
                "steps": steps,
                "guidance_scale": gs,
                "seed": 3000 + w_i * 100 + r_i,
                "ref_image_path": ref_img,
                "extra_params": extra_params,
                "probe_streams": False,
            }
            for r_i in range(n_reqs)
        ]

        log(
            idx,
            f"[{config_meta['name']}] Running {wl['label']} ({size_s} {num_frames}f @{fps}fps, {steps} steps, c={conc}, n={n_reqs})...",
        )
        t_wall0 = time.perf_counter()
        if conc == 1:
            outs = [router.submit(rq) for rq in req_list]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(pool.map(router.submit, req_list))
        wall_s = time.perf_counter() - t_wall0

        oks = [o for o in outs if o.get("ok")]
        lats = [o["e2e_s"] for o in oks]
        inf_ts = [o["inference_time_s"] for o in oks if o.get("inference_time_s") is not None]

        mean_e2e = round(statistics.mean(lats), 3) if lats else None
        med_e2e = round(statistics.median(lats), 3) if lats else None
        p90_e2e = round(sorted(lats)[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
        mean_inf = round(statistics.mean(inf_ts), 3) if inf_ts else None
        encode_overhead_s = (
            round(max(0.0, mean_e2e - mean_inf), 3) if (mean_e2e is not None and mean_inf is not None) else None
        )
        step_ms = round(((mean_inf or mean_e2e) * 1000.0) / max(steps, 1), 1) if (mean_inf or mean_e2e) else None
        rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
        vid_sec_per_s = round((len(oks) * dur_s) / wall_s, 3) if oks and wall_s > 0 else 0.0
        chips_for_30rpm = int(math.ceil(30.0 / rpm) * config_meta["chips"]) if rpm > 0 else None
        peak_hbm_gib = hbm_mon.snapshot_peak_gib() if hbm_mon is not None else None

        row = {
            **config_meta,
            "timestamp_utc": utc_now(),
            "workload_label": wl["label"],
            "size": size_s,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": dur_s,
            "denoise_steps": steps,
            "guidance_scale": gs,
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
            "sample_mp4_bytes": oks[0].get("mp4_bytes") if oks else None,
            "ffprobe_streams": sample_streams,
            "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
        }
        log(
            idx,
            f"[{config_meta['name']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} med_e2e={med_e2e}s inf={mean_inf}s "
            f"enc_ovh={encode_overhead_s}s step={step_ms}ms RPM={rpm}/min vid_s/s={vid_sec_per_s}x",
        )
        collected.append(row)
        append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, row)


def run_synchronized_slice16_topology(
    idx: int,
    *,
    bucket: str,
    run_id: str,
    step_tag: str,
    model_short: str,
    model_path: str,
    dp_per_host: int,
    ulysses_degree: int,
    quantization: str | None,
    remote_encode_url: str,
    workloads: list[dict[str, Any]],
    ref_images: dict[str, Path],
    local_jsonl: Path,
    gcs_jsonl_obj: str,
    collected: list[dict[str, Any]],
    num_pods: int,
) -> None:
    topo_key = f"{step_tag}_dp{dp_per_host * num_pods}_u{ulysses_degree}"
    local_router: VllmOmniVideoRouter | None = None
    boot_err: str | None = None
    try:
        local_router = start_vllm_omni_host_cluster(
            idx,
            config_name=topo_key,
            model_path=model_path,
            dp_replicas_per_host=dp_per_host,
            ulysses_degree=ulysses_degree,
            quantization=quantization,
            remote_encode_url=remote_encode_url,
            base_port=8100,
        )
    except Exception as exc:
        boot_err = str(exc)[:1000]
        log(idx, f"WARNING: [{topo_key}] local cluster boot failed: {boot_err}")

    ep_map = gcs_barrier(
        idx,
        bucket,
        run_id,
        f"{topo_key}_ready",
        {
            "pod_idx": idx,
            "endpoints": local_router.endpoints if local_router is not None else [],
            "boot_error": boot_err,
        },
        num_pods=num_pods,
    )

    if idx == 0:
        all_eps: list[str] = []
        active_hosts = 0
        for p_k in sorted(ep_map.keys()):
            eps = ep_map[p_k].get("endpoints", [])
            if eps:
                active_hosts += 1
                all_eps.extend(eps)
        if not all_eps:
            err_row = {
                "step": step_tag,
                "run_id": run_id,
                "pod_idx": 0,
                "engine": "vllm-torchtpu-omni-v0.32.0",
                "name": f"omni_{topo_key}",
                "slice_topology": f"DP={dp_per_host * num_pods}xUlysses={ulysses_degree}",
                "model_short": model_short,
                "chips": num_pods * 4,
                "workload_label": "slice_boot",
                "denoise_steps": 0,
                "concurrency": 0,
                "num_requests": 0,
                "ok_count": 0,
                "error": boot_err or str(ep_map),
            }
            collected.append(err_row)
            append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, err_row)
        else:
            slice_topo = f"DP={len(all_eps)}xUlysses={ulysses_degree}"
            slice_router = VllmOmniVideoRouter(all_eps)
            meta = {
                "step": step_tag,
                "run_id": run_id,
                "pod_idx": 0,
                "engine": "vllm-torchtpu-omni-v0.32.0",
                "name": f"omni_{topo_key}",
                "slice_topology": slice_topo,
                "model_short": model_short,
                "precision": quantization or "bf16",
                "dp_replicas": len(all_eps),
                "ulysses_degree": ulysses_degree,
                "chips": active_hosts * 4,
                "remote_encode_active": bool(remote_encode_url),
            }
            run_omni_workloads(
                idx,
                bucket=bucket,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_meta=meta,
                router=slice_router,
                hbm_mon=None,
                workloads=workloads,
                ref_images=ref_images,
                collected=collected,
            )

    gcs_barrier(
        idx,
        bucket,
        run_id,
        f"{topo_key}_done",
        {"pod_idx": idx, "done": True},
        num_pods=num_pods,
    )
    if local_router is not None:
        local_router.stop()


def format_suite3b_markdown(run_id: str, all_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Suite 3B: `vllm-omni` MiniMax-H3 Enablement & Full 16-Chip `v7x-16` Slice Multi-Host Report (`{run_id}`)",
        "",
        f"- **Generated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `tpu-v7x-cluster` / `tpu-v7x-16-pool` (`16` physical `v7x` chips / `32` TensorCores / `3,072 GiB` HBM3e)",
        "- **Serving Engine:** `vllm-torchtpu-omni:v0.32.0` + Disaggregated CPU Encoder (`video-encode-svc`)",
        "",
        "| Step | Config | Model | Topology | Chips | Workload | Frames | Steps | Conc. | Med E2E (s) | Inf (s) | Enc Ovh (s) | RPM (vids/min) | Video-s/s | Chips for 30 RPM | OK |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        topo = r.get("slice_topology") or f"DP={r.get('dp_replicas', 1)}xU={r.get('ulysses_degree', 8)}"
        lines.append(
            f"| `{r.get('step')}` | `{r.get('name')}` | `{r.get('model_short')}` | `{topo}` | "
            f"{r.get('chips', 4)} | **{r.get('workload_label')}** | {r.get('num_frames') or '-'} | "
            f"{r.get('denoise_steps') or '-'} | {r.get('concurrency') or '-'} | "
            f"**{r.get('median_e2e_s') or '-'}** | {r.get('mean_inference_s') or '-'} | "
            f"{r.get('encode_and_io_overhead_s') or '-'} | **{r.get('rpm_videos_per_min') or 0.0}** | "
            f"{r.get('video_sec_per_wall_sec') or 0.0}x | {r.get('chips_needed_for_30_rpm') or '-'} | "
            f"{r.get('ok_count', 0)}/{r.get('num_requests', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Suite 3B: vllm-omni MiniMax-H3 & Full 16-Chip Slice Sweeps")
    parser.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", f"suite3b_{int(time.time())}"))
    parser.add_argument("--remote-encode-url", default=os.environ.get("REMOTE_ENCODE_URL", DEFAULT_REMOTE_ENCODE_URL))
    parser.add_argument("--num-pods", type=int, default=int(os.environ.get("NUM_PODS", "4")))
    args = parser.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    bucket = args.gcs_bucket
    run_id = args.run_id
    hostname = socket.gethostname()
    pod_ip = local_pod_ip()

    local_jsonl = Path(f"/workspace/suite3b_pod{idx}.jsonl")
    gcs_jsonl_obj = f"v7x-video-opt/results/suite3b_pod{idx}.jsonl"

    log(idx, f"=== Starting Suite 3B on {hostname} (pod_ip={pod_ip}, run_id={run_id}, shm_free={shm_free_gib()} GiB) ===")
    ensure_subhost_and_omni_patches(idx)
    remote_encode_url = check_remote_encode_service(idx, args.remote_encode_url)
    ref_images = generate_reference_images(Path("/dev/shm/suite3b_refs"))

    collected_rows: list[dict[str, Any]] = []
    ltx25_dp4_u2_verified = False

    # =========================================================================
    # STEP 1: MiniMax-H3 on vllm-omni (Pods 0–1, try/except isolated) +
    #         LTX-2.5 Disaggregated Encode & Sub-Host Scaling (Pods 2–3)
    # =========================================================================
    if idx in (0, 1):
        mm_configs = (
            [
                {"name": "omni_minimax_bf16_u8_opt", "quantization": None, "dp": 1, "ulysses": 8},
            ]
            if idx == 0
            else [
                {"name": "omni_minimax_fp8_u8_opt", "quantization": "fp8", "dp": 1, "ulysses": 8},
                {"name": "omni_minimax_fp8_dp2_u4_opt", "quantization": "fp8", "dp": 2, "ulysses": 4},
            ]
        )
        try:
            mm_path = stage_minimax_for_omni(idx, bucket)
            mm_wls = [
                {
                    "label": "minimax_512p_5s_124f_20step_c1",
                    "size": "896x512",
                    "num_frames": 124,
                    "fps": 24,
                    "steps": 20,
                    "guidance_scale": 1.0,
                    "concurrency": 1,
                    "num_requests": 2,
                },
                {
                    "label": "minimax_768p_5s_124f_20step_c1",
                    "size": "1344x768",
                    "num_frames": 124,
                    "fps": 24,
                    "steps": 20,
                    "guidance_scale": 1.0,
                    "concurrency": 1,
                    "num_requests": 2,
                },
                {
                    "label": "minimax_768p_15s_360f_20step_c1",
                    "size": "1344x768",
                    "num_frames": 360,
                    "fps": 24,
                    "steps": 20,
                    "guidance_scale": 1.0,
                    "concurrency": 1,
                    "num_requests": 2,
                },
            ]
            for mcfg in mm_configs:
                try:
                    hbm_mon = LibtpuHbmMonitor()
                    hbm_mon.start()
                    router = start_vllm_omni_host_cluster(
                        idx,
                        config_name=mcfg["name"],
                        model_path=mm_path,
                        dp_replicas_per_host=mcfg["dp"],
                        ulysses_degree=mcfg["ulysses"],
                        quantization=mcfg["quantization"],
                        remote_encode_url=remote_encode_url,
                    )
                    meta = {
                        "step": "step1_minimax_omni",
                        "run_id": run_id,
                        "pod_idx": idx,
                        "hostname": hostname,
                        "engine": "vllm-torchtpu-omni-v0.32.0",
                        "name": mcfg["name"],
                        "model_short": "MiniMax-H3-FL2VA",
                        "precision": mcfg["quantization"] or "bf16",
                        "dp_replicas": mcfg["dp"],
                        "ulysses_degree": mcfg["ulysses"],
                        "chips": 4,
                        "remote_encode_active": bool(remote_encode_url),
                        "server_boot_s": router.boot_s,
                    }
                    run_omni_workloads(
                        idx,
                        bucket=bucket,
                        local_jsonl=local_jsonl,
                        gcs_jsonl_obj=gcs_jsonl_obj,
                        config_meta=meta,
                        router=router,
                        hbm_mon=hbm_mon,
                        workloads=mm_wls,
                        ref_images=ref_images,
                        collected=collected_rows,
                    )
                    hbm_mon.stop()
                    router.stop()
                except Exception as inner_exc:
                    log(idx, f"Step 1 MiniMax-H3 [{mcfg['name']}] diagnostic captured: {inner_exc}")
                    diag_row = {
                        "step": "step1_minimax_omni",
                        "run_id": run_id,
                        "pod_idx": idx,
                        "hostname": hostname,
                        "engine": "vllm-torchtpu-omni-v0.32.0",
                        "name": mcfg["name"],
                        "model_short": "MiniMax-H3-FL2VA",
                        "precision": mcfg["quantization"] or "bf16",
                        "dp_replicas": mcfg["dp"],
                        "ulysses_degree": mcfg["ulysses"],
                        "chips": 4,
                        "workload_label": "minimax_enablement_probe",
                        "ok_count": 0,
                        "num_requests": 0,
                        "error": str(inner_exc)[:1200],
                    }
                    collected_rows.append(diag_row)
                    append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, diag_row)
                    stop_omni_cluster_cleanly()
        except Exception as stage_exc:
            log(idx, f"Step 1 MiniMax-H3 staging diagnostic: {stage_exc}")
            diag_row = {
                "step": "step1_minimax_omni",
                "run_id": run_id,
                "pod_idx": idx,
                "hostname": hostname,
                "name": f"omni_minimax_stage_pod{idx}",
                "model_short": "MiniMax-H3-FL2VA",
                "chips": 4,
                "workload_label": "minimax_stage",
                "ok_count": 0,
                "num_requests": 0,
                "error": str(stage_exc)[:1200],
            }
            collected_rows.append(diag_row)
            append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, diag_row)

    elif idx == 2:
        # Pod 2: LTX-2.5 Disaggregated CPU Encode Ablation (local vs remote) + DP=2 x Ulysses=4
        try:
            ltx_path = stage_ltx25_for_omni(idx, bucket)
            ablation_wls = [
                {
                    "label": "ltx25_720p_5s_121f_8step_c1",
                    "size": "1280x704",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 1,
                    "num_requests": 2,
                },
                {
                    "label": "ltx25_1080p_5s_121f_8step_c1",
                    "size": "1920x1088",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 1,
                    "num_requests": 2,
                },
                {
                    "label": "ltx25_720p_5s_121f_8step_c2",
                    "size": "1280x704",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 2,
                    "num_requests": 4,
                },
            ]
            for sub_cfg in (
                {"name": "omni_ltx25_fp8_u8_local_enc", "dp": 1, "u": 8, "rem": ""},
                {"name": "omni_ltx25_fp8_u8_remote_enc", "dp": 1, "u": 8, "rem": remote_encode_url},
                {"name": "omni_ltx25_fp8_dp2_u4_remote_enc", "dp": 2, "u": 4, "rem": remote_encode_url},
            ):
                try:
                    hbm_mon = LibtpuHbmMonitor()
                    hbm_mon.start()
                    router = start_vllm_omni_host_cluster(
                        idx,
                        config_name=sub_cfg["name"],
                        model_path=ltx_path,
                        dp_replicas_per_host=sub_cfg["dp"],
                        ulysses_degree=sub_cfg["u"],
                        quantization="fp8",
                        remote_encode_url=sub_cfg["rem"],
                    )
                    meta = {
                        "step": "step1_ltx25_host_opt",
                        "run_id": run_id,
                        "pod_idx": idx,
                        "hostname": hostname,
                        "engine": "vllm-torchtpu-omni-v0.32.0",
                        "name": sub_cfg["name"],
                        "model_short": "LTX-2.5",
                        "precision": "fp8",
                        "dp_replicas": sub_cfg["dp"],
                        "ulysses_degree": sub_cfg["u"],
                        "chips": 4,
                        "remote_encode_active": bool(sub_cfg["rem"]),
                        "server_boot_s": router.boot_s,
                    }
                    run_omni_workloads(
                        idx,
                        bucket=bucket,
                        local_jsonl=local_jsonl,
                        gcs_jsonl_obj=gcs_jsonl_obj,
                        config_meta=meta,
                        router=router,
                        hbm_mon=hbm_mon,
                        workloads=ablation_wls,
                        ref_images=ref_images,
                        collected=collected_rows,
                    )
                    hbm_mon.stop()
                    router.stop()
                except Exception as exc:
                    log(idx, f"Pod 2 [{sub_cfg['name']}] error: {exc}")
                    stop_omni_cluster_cleanly()
        except Exception as exc:
            log(idx, f"Pod 2 LTX-2.5 staging error: {exc}")

    elif idx == 3:
        # Pod 3: LTX-2.5 DP=4 x Ulysses=2 (4x 1-chip servers on 1 host) verification
        try:
            ltx_path = stage_ltx25_for_omni(idx, bucket)
            hbm_mon = LibtpuHbmMonitor()
            hbm_mon.start()
            router = start_vllm_omni_host_cluster(
                idx,
                config_name="omni_ltx25_fp8_dp4_u2_remote_enc",
                model_path=ltx_path,
                dp_replicas_per_host=4,
                ulysses_degree=2,
                quantization="fp8",
                remote_encode_url=remote_encode_url,
            )
            meta = {
                "step": "step1_ltx25_host_opt",
                "run_id": run_id,
                "pod_idx": idx,
                "hostname": hostname,
                "engine": "vllm-torchtpu-omni-v0.32.0",
                "name": "omni_ltx25_fp8_dp4_u2_remote_enc",
                "model_short": "LTX-2.5",
                "precision": "fp8",
                "dp_replicas": 4,
                "ulysses_degree": 2,
                "chips": 4,
                "remote_encode_active": bool(remote_encode_url),
                "server_boot_s": router.boot_s,
            }
            u2_wls = [
                {
                    "label": "ltx25_544p_5s_121f_8step_c4",
                    "size": "960x544",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 4,
                    "num_requests": 8,
                },
                {
                    "label": "ltx25_720p_5s_121f_8step_c4",
                    "size": "1280x704",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 4,
                    "num_requests": 8,
                },
                {
                    "label": "ltx25_1080p_5s_121f_8step_c4",
                    "size": "1920x1088",
                    "num_frames": 121,
                    "fps": 24,
                    "steps": 8,
                    "guidance_scale": 1.0,
                    "concurrency": 4,
                    "num_requests": 8,
                },
            ]
            run_omni_workloads(
                idx,
                bucket=bucket,
                local_jsonl=local_jsonl,
                gcs_jsonl_obj=gcs_jsonl_obj,
                config_meta=meta,
                router=router,
                hbm_mon=hbm_mon,
                workloads=u2_wls,
                ref_images=ref_images,
                collected=collected_rows,
            )
            ltx25_dp4_u2_verified = any(
                r.get("name") == "omni_ltx25_fp8_dp4_u2_remote_enc" and r.get("ok_count", 0) > 0
                for r in collected_rows
            )
            hbm_mon.stop()
            router.stop()
        except Exception as exc:
            log(idx, f"Pod 3 DP=4xU=2 verification diagnostic: {exc}")
            diag_row = {
                "step": "step1_ltx25_host_opt",
                "run_id": run_id,
                "pod_idx": idx,
                "hostname": hostname,
                "name": "omni_ltx25_fp8_dp4_u2_remote_enc",
                "model_short": "LTX-2.5",
                "dp_replicas": 4,
                "ulysses_degree": 2,
                "chips": 4,
                "workload_label": "dp4_u2_probe",
                "ok_count": 0,
                "num_requests": 0,
                "error": str(exc)[:1000],
            }
            collected_rows.append(diag_row)
            append_and_upload_jsonl(idx, bucket, local_jsonl, gcs_jsonl_obj, diag_row)
            stop_omni_cluster_cleanly()

    s1_info = gcs_barrier(
        idx,
        bucket,
        run_id,
        "step1_complete",
        {"pod_idx": idx, "ltx25_dp4_u2_verified": ltx25_dp4_u2_verified},
        num_pods=args.num_pods,
    )
    enable_slice_dp16_u2 = bool(s1_info.get(3, {}).get("ltx25_dp4_u2_verified", False))

    # =========================================================================
    # STEP 2: Full 16-Chip `v7x-16` Slice Multi-Host `LTX-2.5` Sweeps
    # =========================================================================
    log(idx, f"=== Step 2: Staging LTX-2.5 across all 4 hosts (enable_dp16_u2={enable_slice_dp16_u2}) ===")
    ltx_slice_path = stage_ltx25_for_omni(idx, bucket)

    ltx_dp4_u8_wls = [
        {"label": "slice16_ltx25_544p_5s_121f_8step_c4", "size": "960x544", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_ltx25_544p_5s_121f_8step_c8", "size": "960x544", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_ltx25_720p_5s_121f_8step_c4", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_ltx25_720p_5s_121f_8step_c8", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_ltx25_1080p_5s_121f_8step_c4", "size": "1920x1088", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_ltx25_1080p_20s_481f_klingv4_c4", "size": "1920x1088", "num_frames": 481, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_ltx25_1080p_20s_481f_klingv4_c8", "size": "1920x1088", "num_frames": 481, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 8},
    ]
    run_synchronized_slice16_topology(
        idx,
        bucket=bucket,
        run_id=run_id,
        step_tag="step2_slice16_ltx25",
        model_short="LTX-2.5",
        model_path=ltx_slice_path,
        dp_per_host=1,
        ulysses_degree=8,
        quantization="fp8",
        remote_encode_url=remote_encode_url,
        workloads=ltx_dp4_u8_wls,
        ref_images=ref_images,
        local_jsonl=local_jsonl,
        gcs_jsonl_obj=gcs_jsonl_obj,
        collected=collected_rows,
        num_pods=args.num_pods,
    )

    ltx_dp8_u4_wls = [
        {"label": "slice16_ltx25_544p_5s_121f_8step_c8", "size": "960x544", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_ltx25_544p_5s_121f_8step_c16", "size": "960x544", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 16, "num_requests": 32},
        {"label": "slice16_ltx25_720p_5s_121f_8step_c8", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_ltx25_720p_5s_121f_8step_c16", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 16, "num_requests": 32},
        {"label": "slice16_ltx25_1080p_5s_121f_8step_c8", "size": "1920x1088", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_ltx25_1080p_20s_481f_klingv4_c8", "size": "1920x1088", "num_frames": 481, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 8, "num_requests": 8},
    ]
    run_synchronized_slice16_topology(
        idx,
        bucket=bucket,
        run_id=run_id,
        step_tag="step2_slice16_ltx25",
        model_short="LTX-2.5",
        model_path=ltx_slice_path,
        dp_per_host=2,
        ulysses_degree=4,
        quantization="fp8",
        remote_encode_url=remote_encode_url,
        workloads=ltx_dp8_u4_wls,
        ref_images=ref_images,
        local_jsonl=local_jsonl,
        gcs_jsonl_obj=gcs_jsonl_obj,
        collected=collected_rows,
        num_pods=args.num_pods,
    )

    if enable_slice_dp16_u2:
        ltx_dp16_u2_wls = [
            {"label": "slice16_ltx25_544p_5s_121f_8step_c16", "size": "960x544", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 16, "num_requests": 32},
            {"label": "slice16_ltx25_720p_5s_121f_8step_c16", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 16, "num_requests": 32},
            {"label": "slice16_ltx25_720p_5s_121f_8step_c32", "size": "1280x704", "num_frames": 121, "fps": 24, "steps": 8, "guidance_scale": 1.0, "concurrency": 32, "num_requests": 32},
        ]
        run_synchronized_slice16_topology(
            idx,
            bucket=bucket,
            run_id=run_id,
            step_tag="step2_slice16_ltx25",
            model_short="LTX-2.5",
            model_path=ltx_slice_path,
            dp_per_host=4,
            ulysses_degree=2,
            quantization="fp8",
            remote_encode_url=remote_encode_url,
            workloads=ltx_dp16_u2_wls,
            ref_images=ref_images,
            local_jsonl=local_jsonl,
            gcs_jsonl_obj=gcs_jsonl_obj,
            collected=collected_rows,
            num_pods=args.num_pods,
        )

    # =========================================================================
    # STEP 3: Full 16-Chip `v7x-16` Slice Multi-Host `Wan 2.2 I2V-A14B` Sweeps
    # =========================================================================
    log(idx, "=== Step 3: Swapping /dev/shm to Wan2.2-I2V-A14B-Diffusers across all 4 hosts ===")
    wan_slice_path = stage_wan22_i2v_for_omni(idx, bucket)

    wan_dp4_u8_wls = [
        {"label": "slice16_wan22_i2v_480p_5s_81f_4step_c4", "size": "832x480", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_wan22_i2v_480p_5s_81f_4step_c8", "size": "832x480", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_wan22_i2v_720p_5s_81f_4step_c4", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_wan22_i2v_720p_5s_81f_4step_c8", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_wan22_i2v_720p_5s_81f_20step_c4", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 20, "guidance_scale": 3.5, "use_i2v": True, "concurrency": 4, "num_requests": 8},
        {"label": "slice16_wan22_i2v_720p_10s_161f_20step_c4", "size": "1280x720", "num_frames": 161, "fps": 16, "steps": 20, "guidance_scale": 3.5, "use_i2v": True, "concurrency": 4, "num_requests": 4},
    ]
    run_synchronized_slice16_topology(
        idx,
        bucket=bucket,
        run_id=run_id,
        step_tag="step3_slice16_wan22_i2v",
        model_short="Wan2.2-I2V-A14B",
        model_path=wan_slice_path,
        dp_per_host=1,
        ulysses_degree=8,
        quantization="fp8",
        remote_encode_url=remote_encode_url,
        workloads=wan_dp4_u8_wls,
        ref_images=ref_images,
        local_jsonl=local_jsonl,
        gcs_jsonl_obj=gcs_jsonl_obj,
        collected=collected_rows,
        num_pods=args.num_pods,
    )

    wan_dp8_u4_wls = [
        {"label": "slice16_wan22_i2v_480p_5s_81f_4step_c8", "size": "832x480", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_wan22_i2v_480p_5s_81f_4step_c16", "size": "832x480", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 16, "num_requests": 32},
        {"label": "slice16_wan22_i2v_720p_5s_81f_4step_c8", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 8, "num_requests": 16},
        {"label": "slice16_wan22_i2v_720p_5s_81f_4step_c16", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 4, "guidance_scale": 1.0, "use_i2v": True, "concurrency": 16, "num_requests": 32},
        {"label": "slice16_wan22_i2v_720p_5s_81f_20step_c8", "size": "1280x720", "num_frames": 81, "fps": 16, "steps": 20, "guidance_scale": 3.5, "use_i2v": True, "concurrency": 8, "num_requests": 16},
    ]
    run_synchronized_slice16_topology(
        idx,
        bucket=bucket,
        run_id=run_id,
        step_tag="step3_slice16_wan22_i2v",
        model_short="Wan2.2-I2V-A14B",
        model_path=wan_slice_path,
        dp_per_host=2,
        ulysses_degree=4,
        quantization="fp8",
        remote_encode_url=remote_encode_url,
        workloads=wan_dp8_u4_wls,
        ref_images=ref_images,
        local_jsonl=local_jsonl,
        gcs_jsonl_obj=gcs_jsonl_obj,
        collected=collected_rows,
        num_pods=args.num_pods,
    )

    if idx == 0:
        all_rows: list[dict[str, Any]] = []
        for p_k in range(args.num_pods):
            raw = gcs_download_bytes(bucket, f"v7x-video-opt/results/suite3b_pod{p_k}.jsonl")
            if raw:
                for ln in raw.decode("utf-8").splitlines():
                    if ln.strip():
                        try:
                            obj = json.loads(ln)
                            if obj.get("run_id") == run_id:
                                all_rows.append(obj)
                        except Exception:
                            pass
        summary_obj = {
            "run_id": run_id,
            "updated_utc": utc_now(),
            "rows": all_rows,
        }
        md_report = format_suite3b_markdown(run_id, all_rows)
        gcs_upload_bytes(
            bucket,
            f"v7x-video-opt/results/{run_id}/suite3b_summary.json",
            json.dumps(summary_obj, indent=2).encode("utf-8"),
        )
        gcs_upload_bytes(
            bucket,
            f"v7x-video-opt/results/{run_id}/suite3b_summary.md",
            md_report.encode("utf-8"),
            content_type="text/markdown",
        )
        log(idx, f"Uploaded Suite 3B consolidated summary ({len(all_rows)} rows).")

    log(idx, "=== Suite 3B COMPLETE ===")


if __name__ == "__main__":
    main()
