#!/usr/bin/env python3
"""Suite 1: TPU v7x Video + Stereo-Audio Diffusion Serving Optimization on vLLM-Omni.

Runs as a 2-pod Indexed Job on GKE nodepool `tpu-v7x-cluster-8-pool-a` (`2x2x2` slice = 2 hosts x 4 v7x chips):
  - Pod 0 (`JOB_COMPLETION_INDEX=0`): `Lightricks/LTX-2.5` (22B Audio-Video DiT, 24 fps + 24 kHz stereo audio)
      * Attention ablation: `TORCH_SDPA` vs `TPU_FLASH_ATTN` (Pallas FlashAttention)
      * Disaggregated CPU encoding ablation: local PyAV vs `video-encode-svc` (`c3-standard-44` AVX-512)
      * Precision & full resolution/duration matrix: BF16 vs online FP8 up to `1080p (1920x1088) 20s 481f` (Kling-V4 Proxy)
      * Sub-host topology scaling: `DP=1 x U=8` (4 chips), `DP=2 x U=4` (2x 2-chip servers), `DP=4 x U=2` (4x 1-chip servers)
  - Pod 1 (`JOB_COMPLETION_INDEX=1`): `Wan-AI/Wan2.2-T2V-A14B-Diffusers` & `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (28.6B MoE, 16 fps)
      * Phase A (T2V): local PyAV vs Disagg encode, BF16 vs FP8, 4-step & 20-step, Portrait/10s, `DP=2 x U=4` & `DP=4 x U=2`
      * Phase B (I2V): `/dev/shm` cleanup + I2V staging, BF16 vs FP8 with `input_reference` PNG upload, `DP=2 x U=4` & `DP=4 x U=2`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import math
import os
import py_compile
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BUCKET = "YOUR_GCS_BUCKET"
DEFAULT_GCS_PREFIX = "v7x-video-opt"
DEFAULT_ENCODE_SVC_URL = "http://video-encode-svc.default.svc.cluster.local:8090/encode"

LTX_PROMPTS = [
    (
        "Cinematic tracking shot of an emerald mountain express train crossing a mossy stone viaduct above a rushing "
        "glacial canyon at golden hour; crisp rhythmic clatter of steel wheels on rails, rushing river rapids below, "
        "and a distant echoing train horn."
    ),
    (
        "Close-up 35mm film shot of a master violin luthier carving spruce wood in a sunlit workshop, fine wood "
        "curls falling in slow motion; crisp chisel scrapes on resonant timber and gentle warm acoustic violin notes."
    ),
    (
        "Aerial drone shot gliding low over bioluminescent ocean waves breaking against black basalt sea stacks at "
        "twilight; deep booming surf, fizzing seafoam, and coastal wind."
    ),
    (
        "Cyberpunk night street market in heavy rain with neon reflections rippling in puddles as a street chef "
        "sears noodles in a flaming wok; sizzling oil, rain pattering on umbrellas, and lively night-market ambience."
    ),
]

WAN_PROMPTS = [
    (
        "Cinematic wide tracking shot of a sleek silver high-speed train gliding across a misty alpine viaduct at "
        "sunrise, golden rays piercing low valley clouds, crisp reflections, ultra-detailed photorealistic 4K."
    ),
    (
        "Close-up natural-light portrait of a sea turtle swimming gracefully above a vibrant coral reef, caustic "
        "sunbeams dancing across turquoise water, smooth slow-motion camera movement."
    ),
    (
        "Sweeping drone shot along dramatic coastal cliffs at sunset as amber waves crash against dark volcanic rock, "
        "fine sea spray glowing in warm backlight, cinematic anamorphic lens."
    ),
    (
        "Cozy rain-streaked Tokyo alleyway at dusk, paper lanterns swaying in a gentle breeze, puddles mirroring warm "
        "shopfront lights, shallow depth of field, rich film grain."
    ),
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

REMOTE_VIDEO_ENCODE_PATCH = r"""

# V7X_DISAGG_REMOTE_ENCODE_PATCH_V1
import json as _v7x_json
import struct as _v7x_struct
import time as _v7x_time
import urllib.request as _v7x_urlreq

_V7X_ORIG_ENCODE_VIDEO_BYTES = _encode_video_bytes
_V7X_TELEM_LOCK = threading.Lock()


def _v7x_append_encode_telemetry(entry: dict[str, Any]) -> None:
    p = os.environ.get("VLLM_OMNI_ENCODE_LOG_PATH", "").strip()
    if not p:
        return
    try:
        line = _v7x_json.dumps(entry) + "\n"
        with _V7X_TELEM_LOCK:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def _v7x_coerce_rgb24_frames(video: Any) -> np.ndarray:
    if isinstance(video, list):
        if not video:
            raise ValueError("Empty video frame list")
        first = video[0]
        if isinstance(first, Image.Image):
            arr = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in video], axis=0)
            return np.ascontiguousarray(arr, dtype=np.uint8)
        converted = []
        for item in video:
            if isinstance(item, torch.Tensor):
                t = item.detach().cpu()
                converted.append(t.numpy() if t.dtype == torch.uint8 else t.float().numpy())
            else:
                converted.append(np.asarray(item))
        arr = np.stack(converted, axis=0) if converted[0].ndim == 3 else np.concatenate(converted, axis=0)
    elif isinstance(video, torch.Tensor):
        t = video.detach().cpu()
        arr = t.numpy() if t.dtype == torch.uint8 else t.float().numpy()
    elif isinstance(video, np.ndarray):
        arr = video
    else:
        arr = np.asarray(video)

    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video frames, got shape={arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        pass
    elif arr.shape[1] in (1, 3, 4) and arr.shape[-1] > 4:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[0] in (1, 3, 4) and arr.shape[-1] > 4:
        arr = np.transpose(arr, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unrecognized 4D video layout shape={arr.shape}")

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        amin = float(np.nanmin(arr_f))
        amax = float(np.nanmax(arr_f))
        if amin < -0.05:
            arr_f = arr_f * 0.5 + 0.5
        if amax <= 1.5:
            arr_f = arr_f * 255.0
        arr = np.clip(np.rint(arr_f), 0.0, 255.0).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _v7x_coerce_f32le_audio(audio: Any) -> tuple[bytes, int]:
    if audio is None:
        return b"", 0
    if isinstance(audio, torch.Tensor):
        aud = audio.detach().cpu().float().numpy()
    elif isinstance(audio, np.ndarray):
        aud = audio.astype(np.float32, copy=False)
    else:
        aud = np.asarray(audio, dtype=np.float32)

    while aud.ndim > 2 and aud.shape[0] == 1:
        aud = aud[0]
    if aud.size == 0:
        return b"", 0
    if aud.ndim == 1:
        aud = aud[:, None]
    elif aud.ndim == 2:
        if aud.shape[0] in (1, 2, 4) and aud.shape[1] > aud.shape[0] * 16:
            aud = aud.T
    else:
        raise ValueError(f"Unsupported audio shape={aud.shape}")

    aud_clean = np.ascontiguousarray(
        np.clip(np.nan_to_num(aud, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0),
        dtype="<f4",
    )
    return aud_clean.tobytes(), int(aud_clean.shape[1])


def _encode_video_bytes(video: Any, *args: Any, **kwargs: Any) -> bytes:
    remote_url = os.environ.get("VLLM_OMNI_REMOTE_ENCODE_URL", "").strip()
    if remote_url:
        t0 = _v7x_time.perf_counter()
        try:
            fps_arg = kwargs.get("fps")
            if fps_arg is None and len(args) >= 1 and isinstance(args[0], (int, float)):
                fps_arg = args[0]
            fps_val = float(fps_arg or 24.0)

            frames_u8 = _v7x_coerce_rgb24_frames(video)
            nf, h, w, _ = frames_u8.shape

            audio_obj = kwargs.get("audio")
            aud_bytes, aud_ch = _v7x_coerce_f32le_audio(audio_obj)
            audio_sr = int(kwargs.get("audio_sample_rate") or DEFAULT_AUDIO_SAMPLE_RATE) if aud_bytes else 0

            meta = {
                "width": int(w),
                "height": int(h),
                "num_frames": int(nf),
                "fps": float(fps_val),
                "pix_fmt": "rgb24",
                "preset": "ultrafast",
                "crf": 20,
                "threads": 16,
                "audio_sr": int(audio_sr),
                "audio_channels": int(aud_ch or 2),
                "audio_bytes": len(aud_bytes),
                "return_mp4": True,
            }
            meta_b = _v7x_json.dumps(meta).encode("utf-8")
            hdr_b = _v7x_struct.pack(">I", len(meta_b)) + meta_b
            vid_len = int(frames_u8.nbytes)
            total_len = len(hdr_b) + vid_len + len(aud_bytes)

            payload = bytearray(total_len)
            mv = memoryview(payload)
            mv[: len(hdr_b)] = hdr_b
            mv[len(hdr_b) : len(hdr_b) + vid_len] = frames_u8.reshape(-1).data
            if aud_bytes:
                mv[len(hdr_b) + vid_len :] = aud_bytes

            req = _v7x_urlreq.Request(
                remote_url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(total_len),
                },
            )
            with _v7x_urlreq.urlopen(req, timeout=300.0) as resp:
                mp4_bytes = resp.read()
                enc_host = resp.headers.get("X-Encoder-Host", "")
                rem_ms = float(resp.headers.get("X-Encode-Time-Ms", "0") or 0.0)
            if len(mp4_bytes) < 512:
                raise RuntimeError(f"Remote encoder returned suspiciously small MP4 ({len(mp4_bytes)}B)")
            wall_s = _v7x_time.perf_counter() - t0
            logger.info(
                "[V7X_REMOTE_ENCODE] ok host=%s shape=%dx%dx%d audio_b=%d mp4_b=%d wall_s=%.4f remote_ms=%.2f",
                enc_host,
                nf,
                w,
                h,
                len(aud_bytes),
                len(mp4_bytes),
                wall_s,
                rem_ms,
            )
            _v7x_append_encode_telemetry(
                {
                    "ts": _v7x_time.time(),
                    "mode": "disagg",
                    "encode_s": round(wall_s, 4),
                    "remote_ffmpeg_s": round(rem_ms / 1000.0, 4),
                    "encoder_host": enc_host,
                    "mp4_bytes": len(mp4_bytes),
                    "has_audio": bool(len(aud_bytes) > 0),
                }
            )
            return mp4_bytes
        except Exception as exc:
            logger.warning(
                "[V7X_REMOTE_ENCODE] failed (%s); falling back to local PyAV _encode_video_bytes",
                exc,
            )

    t_loc0 = _v7x_time.perf_counter()
    mp4_bytes = _V7X_ORIG_ENCODE_VIDEO_BYTES(video, *args, **kwargs)
    wall_s = _v7x_time.perf_counter() - t_loc0
    has_aud = bool(kwargs.get("audio") is not None)
    logger.info("[V7X_LOCAL_ENCODE] ok mp4_b=%d wall_s=%.4f", len(mp4_bytes), wall_s)
    _v7x_append_encode_telemetry(
        {
            "ts": _v7x_time.time(),
            "mode": "localenc",
            "encode_s": round(wall_s, 4),
            "remote_ffmpeg_s": None,
            "encoder_host": "local_pyav",
            "mp4_bytes": len(mp4_bytes),
            "has_audio": has_aud,
        }
    )
    return mp4_bytes
"""

_MOCK_META_PROC: subprocess.Popen[Any] | None = None
_POD_IDX = 0


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{utc_now()}][suite1-pod{_POD_IDX}] {msg}", flush=True)


def host_ram_summary() -> str:
    try:
        mem: dict[str, int] = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 2:
                mem[parts[0].rstrip(":")] = int(parts[1])
        avail_g = mem.get("MemAvailable", 0) / (1024**2)
        tot_g = mem.get("MemTotal", 0) / (1024**2)
        shm = shutil.disk_usage("/dev/shm")
        return f"MemAvail={avail_g:.1f}/{tot_g:.1f}GiB, /dev/shm={shm.used / (1024**3):.1f}/{shm.total / (1024**3):.1f}GiB"
    except Exception:
        return "n/a"


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
            with urllib.request.urlopen(req, timeout=120):
                return
        except Exception as exc:
            if attempt == 4:
                log(f"WARNING: GCS upload failed for {obj}: {exc}")
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(
    bucket: str,
    gcs_prefix: str,
    local_dir: Path,
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
        if exclude_prefixes and any(rel.startswith(ep) for ep in exclude_prefixes):
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
        log(f"All {len(blobs)} files already staged in {local_dir}")
        return len(blobs)

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

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        total_b = sum(pool.map(dl_chunk, tasks))
    dt = max(time.time() - t0, 1e-3)
    log(
        f"Hydrated {total_b / 1e9:.2f} GB from gs://{bucket}/{gcs_prefix} -> {local_dir} "
        f"in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s; {host_ram_summary()})"
    )
    return len(blobs)


def _locate_module_file(mod_name: str, fallback: Path) -> Path:
    try:
        spec = importlib.util.find_spec(mod_name)
        if spec and spec.origin:
            return Path(spec.origin)
    except Exception:
        pass
    return fallback


def ensure_subhost_isolation_and_patches_ready() -> None:
    """Compile metadata_hook.so, start mock_metadata.py, patch OmniTpuPlatform.set_device & video_api_utils.py."""
    global _MOCK_META_PROC

    hook_c = Path("/tmp/metadata_hook.c")
    hook_so = Path("/tmp/metadata_hook.so")
    if not hook_so.exists():
        hook_c.write_text(METADATA_HOOK_C)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(hook_c), "-o", str(hook_so), "-ldl"], check=True)
        log("Compiled /tmp/metadata_hook.so for fast GCE metadata interception")

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
                        log("Started mock_metadata.py proxy on 127.0.0.1:80")
                        break
            except Exception:
                time.sleep(0.2)

    plat_candidates = [
        Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/omni/platform.py"),
        Path("/workspace/vllm-torchtpu/vllm_torchtpu/omni/platform.py"),
    ]
    for p_cand in plat_candidates:
        if not p_cand.exists():
            continue
        txt = p_cand.read_text()
        if "VLLM_OMNI_LOCAL_RANKS" in txt:
            log(f"OmniTpuPlatform.set_device already patched in {p_cand}")
            continue
        start_marker = "    @classmethod\n    def set_device(cls, device: torch.device | int) -> None:\n"
        end_marker = "\n    @staticmethod\n    def _offload_dense_all_to_all() -> None:"
        if start_marker in txt and end_marker in txt:
            pre, rest = txt.split(start_marker, 1)
            _, post = rest.split(end_marker, 1)
            new_txt = pre + PATCHED_SET_DEVICE_METHOD + end_marker + post
            p_cand.write_text(new_txt)
            py_compile.compile(str(p_cand), doraise=True)
            log(f"Patched OmniTpuPlatform.set_device in {p_cand}")
        else:
            raise RuntimeError(f"Could not locate set_device markers in {p_cand}")

    # Patch vllm_omni/entrypoints/openai/video_api_utils.py with disaggregated CPU encoder hook
    vutils_path = _locate_module_file(
        "vllm_omni.entrypoints.openai.video_api_utils",
        Path("/usr/local/lib/python3.12/site-packages/vllm_omni/entrypoints/openai/video_api_utils.py"),
    )
    if not vutils_path.exists():
        raise FileNotFoundError(f"Missing video_api_utils.py at {vutils_path}")
    vtxt = vutils_path.read_text()
    if "V7X_DISAGG_REMOTE_ENCODE_PATCH_V1" not in vtxt:
        vutils_path.write_text(vtxt + REMOTE_VIDEO_ENCODE_PATCH)
        py_compile.compile(str(vutils_path), doraise=True)
        log(f"Patched _encode_video_bytes in {vutils_path} for disaggregated video-encode-svc offload")
    else:
        log(f"video_api_utils.py already patched at {vutils_path}")

    # Extend VIDEO_SYNC_TIMEOUT_S to 3600s so cold XLA compile on 481f/14B never hits HTTP 504
    for mod_name, fb in (
        (
            "vllm_omni.entrypoints.openai.api_server",
            Path("/usr/local/lib/python3.12/site-packages/vllm_omni/entrypoints/openai/api_server.py"),
        ),
        (
            "vllm_omni.entrypoints.openai.video.generation.helpers",
            Path("/usr/local/lib/python3.12/site-packages/vllm_omni/entrypoints/openai/video/generation/helpers.py"),
        ),
    ):
        mp = _locate_module_file(mod_name, fb)
        if mp.exists():
            mtxt = mp.read_text()
            if "VIDEO_SYNC_TIMEOUT_S" in mtxt and "V7X_SYNC_TIMEOUT_OVERRIDE" not in mtxt:
                mtxt_new = (
                    mtxt
                    + "\n# V7X_SYNC_TIMEOUT_OVERRIDE\n"
                    + 'VIDEO_SYNC_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_VIDEO_SYNC_TIMEOUT_S", "3600.0"))\n'
                )
                mp.write_text(mtxt_new)
                py_compile.compile(str(mp), doraise=True)
                log(f"Extended VIDEO_SYNC_TIMEOUT_S to 3600s in {mp}")


def probe_remote_encoder_health(encode_url: str) -> dict[str, Any] | None:
    health_url = encode_url.rsplit("/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5.0) as r:
            data = json.loads(r.read().decode("utf-8"))
            log(f"Disaggregated CPU encoder healthy at {health_url}: {data}")
            return data
    except Exception as exc:
        log(f"WARNING: Could not reach {health_url}: {exc}")
        return None


class LibtpuHbmMonitor:
    """Polls libtpu runtime metrics endpoints (`localhost:8431..8438/metrics`) for Peak HBM per TensorCore."""

    def __init__(self, ports: list[int] | None = None) -> None:
        self.ports = ports or [8431, 8432, 8433, 8434, 8435, 8436, 8437, 8438]
        self.peak_bytes_per_core: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        for idx, port in enumerate(self.ports):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.7) as r:
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
                                    if val > self.peak_bytes_per_core.get(idx, 0.0):
                                        self.peak_bytes_per_core[idx] = val
                            except ValueError:
                                pass
            except Exception:
                pass

    def reset_window(self) -> None:
        self.peak_bytes_per_core.clear()
        self._poll_once()

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
            r"(?:memory|hbm|allocated|used)[^\n]{0,55}?([0-9]+\.[0-9]+)\s*(GiB|GB|MiB|MB)",
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
                    if "/dev/vfio/" in os.readlink(fd):
                        holders.add(pid)
                        break
                except Exception:
                    pass
        except Exception:
            pass
    return sorted(holders)


def stop_cluster_cleanly(procs: list[subprocess.Popen[Any]] | None = None) -> None:
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
            "pkill -9 -f '[m]ultiprocessing.resource_tracker' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/vllm_omni_* "
            "/dev/shm/psm_* /dev/shm/torch_tpu_pg_store_* 2>/dev/null || true",
        ],
        check=False,
    )

    t_vfio = time.time()
    while time.time() - t_vfio < 20.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        log(f"Killing lingering VFIO holder PIDs: {holders}")
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


def generate_synthetic_refs(input_dir: Path, sizes: tuple[str, ...], prefix: str) -> dict[str, bytes]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for sz in sizes:
        w, h = (int(x) for x in sz.split("x"))
        p = input_dir / f"{prefix}_{sz}.png"
        img = Image.new("RGB", (w, h), (24, 44, 70))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(28 + 150 * (y / max(h, 1))))
            g = min(255, int(54 + 102 * (y / max(h, 1))))
            b = min(255, int(94 + 78 * (1.0 - y / max(h, 1))))
            d.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        rad = min(w, h) // 6
        d.ellipse([w // 3, h // 4, w // 3 + rad, h // 4 + rad], fill=(250, 210, 122))
        img.save(p, format="PNG")
        out[sz] = p.read_bytes()
    return out


def _verify_shards_present(model_dir: Path, clean_stray_if_unsharded: bool = False) -> bool:
    if not (model_dir / "model_index.json").exists():
        return False
    all_ok = True
    for idx_file in list(model_dir.rglob("*.safetensors.index.json")):
        try:
            data = json.loads(idx_file.read_text())
            shards = sorted(set((data.get("weight_map") or {}).values()))
            missing = [s for s in shards if not (idx_file.parent / s).exists()]
            if missing:
                unsharded = idx_file.parent / idx_file.name[: -len(".index.json")]
                if clean_stray_if_unsharded and unsharded.exists() and unsharded.stat().st_size > 1024 * 1024:
                    log(f"Removing stray index {idx_file} (unsharded {unsharded.name} exists)")
                    idx_file.unlink(missing_ok=True)
                else:
                    all_ok = False
        except Exception:
            all_ok = False
    return all_ok


def stage_ltx25_model(bucket: str) -> str:
    """Stage Lightricks/LTX-2.5-Diffusers from regional GCS into /dev/shm/LTX-2.5."""
    for stale in (
        "/dev/shm/MiniMax-H3",
        "/dev/shm/FastH3",
        "/dev/shm/Wan2.2-T2V-A14B-Diffusers",
        "/dev/shm/Wan2.2-I2V-A14B-Diffusers",
        "/dev/shm/Wan-AI",
        "/dev/shm/hf_cache",
    ):
        shutil.rmtree(stale, ignore_errors=True)

    model_dir = Path("/dev/shm/LTX-2.5")
    fast_gcs_sync_dir(
        bucket,
        "models/Lightricks/LTX-2.5-Diffusers/",
        model_dir,
        exclude_prefixes=("assets/", "examples/", "prompt_enhancer/", "diffusion_decoder/"),
    )

    if not (model_dir / "model_index.json").exists():
        from huggingface_hub import snapshot_download

        log("GCS LTX-2.5-Diffusers missing model_index.json; falling back to HF snapshot_download...")
        snapshot_download(
            repo_id="Lightricks/LTX-2.5",
            local_dir=str(model_dir),
            ignore_patterns=["*.mp4", "*.gif", "*.png", "*.jpg", "assets/*", "examples/*"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )

    idx_path = model_dir / "model_index.json"
    midx = json.loads(idx_path.read_text())
    changed = False
    for opt_comp in ("prompt_enhancer", "diffusion_decoder"):
        if opt_comp in midx and not (model_dir / opt_comp / "config.json").exists():
            midx.pop(opt_comp, None)
            changed = True
    if changed:
        idx_path.write_text(json.dumps(midx, indent=2) + "\n")
        log("Cleaned optional prompt_enhancer / diffusion_decoder entries from LTX-2.5 model_index.json")

    _verify_shards_present(model_dir, clean_stray_if_unsharded=True)
    sz_gb = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) / 1e9
    log(f"Staged LTX-2.5 at {model_dir} ({sz_gb:.2f} GB; {host_ram_summary()})")
    return str(model_dir)


def stage_wan22_model(bucket: str, repo_id: str, local_dir: Path) -> str:
    """Stage Wan2.2-T2V-A14B-Diffusers or Wan2.2-I2V-A14B-Diffusers into /dev/shm with HF fallback."""
    for stale in ("/dev/shm/LTX-2.5", "/dev/shm/MiniMax-H3", "/dev/shm/FastH3", "/dev/shm/hf_cache"):
        shutil.rmtree(stale, ignore_errors=True)

    fast_gcs_sync_dir(
        bucket,
        f"models/{repo_id}/",
        local_dir,
        exclude_prefixes=("assets/", "examples/"),
    )

    if not _verify_shards_present(local_dir, clean_stray_if_unsharded=True):
        from huggingface_hub import snapshot_download

        log(f"Incomplete GCS mirror for {repo_id}; invoking HF snapshot_download fallback into {local_dir}...")
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["assets/*", "examples/*", "*.mp4", "*.gif", "*.webp", "*.png", "*.jpg"],
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )

    assert (local_dir / "model_index.json").exists(), f"Missing {local_dir / 'model_index.json'}"
    sz_gb = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()) / 1e9
    log(f"Staged {repo_id} at {local_dir} ({sz_gb:.2f} GB; {host_ram_summary()})")
    return str(local_dir)


def build_multipart_form_data(
    fields: dict[str, str],
    file_field: tuple[str, str, str, bytes] | None = None,
) -> tuple[bytes, str]:
    boundary = f"----v7xOmniBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(v).encode("utf-8"))
        chunks.append(b"\r\n")
    if file_field is not None:
        field_name, filename, content_type, raw_bytes = file_field
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(raw_bytes)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def ffprobe_mp4_bytes(mp4_bytes: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(dir="/dev/shm", prefix="probe_", suffix=".mp4", delete=True) as tf:
        tf.write(mp4_bytes)
        tf.flush()
        pr = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,nb_frames,sample_rate,channels",
                "-of",
                "json",
                tf.name,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if pr.returncode != 0:
            return {"valid_video": False, "has_audio": False, "error": pr.stderr[:300]}
        info = json.loads(pr.stdout or "{}")
        streams = info.get("streams") or []
        v_streams = [s for s in streams if s.get("codec_type") == "video"]
        a_streams = [s for s in streams if s.get("codec_type") == "audio"]
        return {
            "valid_video": len(v_streams) > 0,
            "has_audio": len(a_streams) > 0,
            "video_width": int(v_streams[0].get("width", 0)) if v_streams else 0,
            "video_height": int(v_streams[0].get("height", 0)) if v_streams else 0,
            "audio_channels": int(a_streams[0].get("channels", 0)) if a_streams else 0,
            "audio_sr": int(a_streams[0].get("sample_rate", 0)) if a_streams else 0,
        }


def _read_new_encode_records(log_path: Path, start_offset: int) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            chunk = f.read().decode("utf-8", errors="ignore")
        for ln in chunk.splitlines():
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def call_video_sync_endpoint(
    endpoint: str,
    *,
    prompt: str,
    size: str,
    num_frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    ref_png_bytes: bytes | None = None,
    encode_log_path: Path | None = None,
    run_ffprobe: bool = False,
    timeout_s: float = 3600.0,
) -> dict[str, Any]:
    fields = {
        "prompt": prompt,
        "size": size,
        "num_frames": str(num_frames),
        "fps": str(fps),
        "num_inference_steps": str(steps),
        "guidance_scale": str(guidance_scale),
        "seed": str(seed),
    }
    file_part = ("input_reference", "ref.png", "image/png", ref_png_bytes) if ref_png_bytes is not None else None
    body, ctype = build_multipart_form_data(fields, file_part)

    enc_offset = encode_log_path.stat().st_size if (encode_log_path and encode_log_path.exists()) else 0
    t0 = time.perf_counter()

    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/videos/sync",
            data=body,
            method="POST",
            headers={"Content-Type": ctype, "Content-Length": str(len(body))},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            mp4_bytes = resp.read()
            hdr_inf = resp.headers.get("X-Inference-Time-S")
            req_id = resp.headers.get("X-Request-Id", "")
        e2e_s = time.perf_counter() - t0
        srv_inf_s = float(hdr_inf) if hdr_inf is not None else e2e_s
    except urllib.error.HTTPError as he:
        if he.code in (404, 405):
            return _call_video_async_fallback(
                endpoint,
                body=body,
                ctype=ctype,
                enc_offset=enc_offset,
                encode_log_path=encode_log_path,
                run_ffprobe=run_ffprobe,
                timeout_s=timeout_s,
            )
        err_txt = ""
        try:
            err_txt = he.read().decode("utf-8", errors="ignore")[:700]
        except Exception:
            pass
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {he.code}: {err_txt}"}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": repr(exc)[:700]}

    if len(mp4_bytes) <= 1000:
        return {
            "ok": False,
            "e2e_s": round(e2e_s, 4),
            "error": f"MP4 payload too small ({len(mp4_bytes)} bytes)",
        }

    enc_recs = _read_new_encode_records(encode_log_path, enc_offset) if encode_log_path else []
    enc_s = enc_recs[-1].get("encode_s") if enc_recs else None
    has_audio = enc_recs[-1].get("has_audio") if enc_recs else None

    probe_info: dict[str, Any] = {}
    if run_ffprobe:
        probe_info = ffprobe_mp4_bytes(mp4_bytes)
        if not probe_info.get("valid_video", False):
            return {
                "ok": False,
                "e2e_s": round(e2e_s, 4),
                "error": f"ffprobe validation failed: {probe_info}",
            }
        has_audio = bool(probe_info.get("has_audio", False))

    return {
        "ok": True,
        "request_id": req_id,
        "e2e_s": round(e2e_s, 4),
        "server_inference_s": round(srv_inf_s, 4),
        "encode_s": round(float(enc_s), 4) if enc_s is not None else None,
        "mp4_bytes": len(mp4_bytes),
        "has_audio": has_audio,
        "probe": probe_info if run_ffprobe else None,
        "error": None,
    }


def _call_video_async_fallback(
    endpoint: str,
    *,
    body: bytes,
    ctype: str,
    enc_offset: int,
    encode_log_path: Path | None,
    run_ffprobe: bool,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/videos",
            data=body,
            method="POST",
            headers={"Content-Type": ctype, "Content-Length": str(len(body))},
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            job = json.loads(resp.read().decode("utf-8"))
        vid_id = job.get("id")
        if not vid_id and job.get("data"):
            b64 = job["data"][0].get("b64_json", "")
            import base64

            mp4_bytes = base64.b64decode(b64)
            e2e_s = time.perf_counter() - t0
            enc_recs = _read_new_encode_records(encode_log_path, enc_offset) if encode_log_path else []
            probe_info = ffprobe_mp4_bytes(mp4_bytes) if run_ffprobe else {}
            return {
                "ok": len(mp4_bytes) > 1000,
                "e2e_s": round(e2e_s, 4),
                "server_inference_s": round(e2e_s, 4),
                "encode_s": enc_recs[-1].get("encode_s") if enc_recs else None,
                "mp4_bytes": len(mp4_bytes),
                "has_audio": probe_info.get("has_audio") if run_ffprobe else (enc_recs[-1].get("has_audio") if enc_recs else None),
                "error": None,
            }

        while time.perf_counter() - t0 < timeout_s:
            time.sleep(0.5)
            with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30.0) as r:
                st = json.loads(r.read().decode("utf-8"))
            status = st.get("status")
            if status in ("completed", "succeeded"):
                with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}/content", timeout=120.0) as cr:
                    mp4_bytes = cr.read()
                e2e_s = time.perf_counter() - t0
                enc_recs = _read_new_encode_records(encode_log_path, enc_offset) if encode_log_path else []
                probe_info = ffprobe_mp4_bytes(mp4_bytes) if run_ffprobe else {}
                return {
                    "ok": len(mp4_bytes) > 1000,
                    "e2e_s": round(e2e_s, 4),
                    "server_inference_s": round(float(st.get("inference_time_s") or e2e_s), 4),
                    "encode_s": enc_recs[-1].get("encode_s") if enc_recs else None,
                    "mp4_bytes": len(mp4_bytes),
                    "has_audio": probe_info.get("has_audio") if run_ffprobe else (enc_recs[-1].get("has_audio") if enc_recs else None),
                    "error": None,
                }
            if status in ("failed", "cancelled", "error"):
                return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": str(st.get("error"))[:600]}
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": "async polling timeout"}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": repr(exc)[:700]}


def _write_deploy_yaml(
    cfg_path: Path,
    *,
    ulysses_degree: int,
    max_num_seqs: int,
    attention_backend: str,
    quantization: str | None,
    num_replicas: int = 1,
    devices: str | None = None,
) -> None:
    stage_cfg: dict[str, Any] = {
        "stage_id": 0,
        "ulysses_degree": ulysses_degree,
        "max_num_seqs": max_num_seqs,
        "max_generated_image_size": 16777216,
        "enable_diffusion_pipeline_profiler": True,
        "engine_extras": {
            "num_gpus": ulysses_degree,
            "request_batch_max_wait_ms": 0.0,
        },
    }
    if num_replicas > 1:
        stage_cfg["num_replicas"] = num_replicas
    if devices is not None:
        stage_cfg["devices"] = devices
    elif ulysses_degree == 8:
        stage_cfg["devices"] = "0,1,2,3,4,5,6,7"
    if attention_backend.upper() in ("TORCH_SDPA", "TPU_SDPA", "SDPA"):
        stage_cfg["diffusion_attention_backend"] = "TORCH_SDPA"

    deploy_cfg: dict[str, Any] = {"stages": [stage_cfg]}
    if quantization:
        deploy_cfg["quantization"] = quantization
    cfg_path.write_text(json.dumps(deploy_cfg, indent=2) + "\n")


def _spawn_vllm_omni_proc(
    model_path: str,
    port: int,
    deploy_cfg_path: Path,
    log_path: Path,
    encode_log_path: Path,
    remote_encode_url: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    encode_log_path.unlink(missing_ok=True)
    encode_log_path.touch()

    env = os.environ.copy()
    env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_PREMAPPED_BUFFER_SIZE": str(512 * 1024**2),
            "JAX_COMPILATION_CACHE_DIR": "/dev/shm/xla_cache_vllm_omni",
            "VLLM_DISABLE_LOG_LOGO": "1",
            "TPU_VISIBLE_CHIPS": "0,1,2,3,4,5,6,7",
            "VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT": "3600.0",
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT_S": "3600.0",
            "VLLM_OMNI_REMOTE_ENCODE_URL": remote_encode_url,
            "VLLM_OMNI_ENCODE_LOG_PATH": str(encode_log_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if Path("/tmp/metadata_hook.so").exists():
        env["LD_PRELOAD"] = "/tmp/metadata_hook.so"
    if extra_env:
        env.update(extra_env)

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
        str(deploy_cfg_path),
        "--max-generated-image-size",
        "16777216",
    ]
    lrs_tag = env.get("VLLM_OMNI_LOCAL_RANKS", "all-8")
    topo_tag = env.get("VLLM_OMNI_TOPOLOGY", env.get("TORCH_TPU_TOPOLOGY", "auto"))
    enc_tag = "disagg" if remote_encode_url else "local_pyav"
    log(f"Spawning vllm-omni :{port} (ranks={lrs_tag}, topo={topo_tag}, enc={enc_tag}): {' '.join(cmd)}")
    lf = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)


def _wait_all_healthy(
    procs: list[subprocess.Popen[Any]],
    ports: list[int],
    log_paths: list[Path],
    boot_timeout_s: float = 960.0,
) -> float:
    t0 = time.time()
    ready = [False] * len(ports)
    while time.time() - t0 < boot_timeout_s:
        for idx, (proc, port, lp) in enumerate(zip(procs, ports, log_paths)):
            if proc.poll() is not None:
                tail = extract_clean_error_tail(lp)
                raise RuntimeError(f"vllm-omni :{port} (rep {idx}) exited early (rc={proc.returncode}):\n{tail}")
            if not ready[idx]:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.5) as r:
                        if r.status == 200:
                            ready[idx] = True
                            log(f"vllm-omni replica {idx} healthy on :{port} in {time.time() - t0:.1f}s")
                except Exception:
                    pass
        if all(ready):
            return round(time.time() - t0, 2)
        time.sleep(2.5)

    not_ready = [ports[i] for i, ok in enumerate(ready) if not ok]
    tails = "\n---\n".join(
        f":{ports[i]}:\n{extract_clean_error_tail(log_paths[i], 1500)}" for i, ok in enumerate(ready) if not ok
    )
    raise TimeoutError(f"Timed out waiting for vllm-omni ports {not_ready}:\n{tails}")


class VllmOmniVideoCluster:
    """Manages 1..4 isolated vllm-omni video serving replicas across the 8 v7x TensorCores."""

    def __init__(
        self,
        procs: list[subprocess.Popen[Any]],
        endpoints: list[str],
        log_paths: list[Path],
        encode_log_paths: list[Path],
        boot_s: float,
        mode: str,
    ) -> None:
        self.procs = procs
        self.endpoints = endpoints
        self.log_paths = log_paths
        self.encode_log_paths = encode_log_paths
        self.boot_s = boot_s
        self.mode = mode
        self._lock = threading.Lock()
        self._inflight = [0] * len(endpoints)
        self._rr = 0

    def any_proc_dead(self) -> bool:
        return any(p.poll() is not None for p in self.procs)

    def _acquire_endpoint(self) -> tuple[int, str, Path]:
        with self._lock:
            n = len(self.endpoints)
            best_idx = min(range(n), key=lambda i: (self._inflight[i], (i - self._rr) % n))
            self._inflight[best_idx] += 1
            self._rr = (best_idx + 1) % n
            return best_idx, self.endpoints[best_idx], self.encode_log_paths[best_idx]

    def _release_endpoint(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)

    def generate_single(
        self,
        *,
        prompt: str,
        size: str,
        num_frames: int,
        fps: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        ref_png_bytes: bytes | None = None,
        replica_override: int | None = None,
        run_ffprobe: bool = False,
        timeout_s: float = 3600.0,
    ) -> dict[str, Any]:
        if replica_override is not None:
            return call_video_sync_endpoint(
                self.endpoints[replica_override],
                prompt=prompt,
                size=size,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                ref_png_bytes=ref_png_bytes,
                encode_log_path=self.encode_log_paths[replica_override],
                run_ffprobe=run_ffprobe,
                timeout_s=timeout_s,
            )
        idx, ep, enc_lp = self._acquire_endpoint()
        try:
            return call_video_sync_endpoint(
                ep,
                prompt=prompt,
                size=size,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                ref_png_bytes=ref_png_bytes,
                encode_log_path=enc_lp,
                run_ffprobe=run_ffprobe,
                timeout_s=timeout_s,
            )
        finally:
            self._release_endpoint(idx)

    def stop(self) -> None:
        stop_cluster_cleanly(self.procs)


def start_vllm_omni_video_cluster(
    *,
    model_path: str,
    base_port: int,
    dp_size: int,
    ulysses_degree: int,
    attention_backend: str,
    quantization: str | None,
    remote_encode_url: str,
    group_id: str,
) -> VllmOmniVideoCluster:
    stop_cluster_cleanly(None)
    Path("/dev/shm/xla_cache_vllm_omni").mkdir(parents=True, exist_ok=True)

    # DP=4 x Ulysses=2 -> 4 isolated 1-chip (2-TensorCore) servers on [0,1], [2,3], [4,5], [6,7]
    if dp_size == 4 and ulysses_degree == 2:
        procs: list[subprocess.Popen[Any]] = []
        ports: list[int] = []
        log_paths: list[Path] = []
        enc_logs: list[Path] = []
        try:
            for r in range(4):
                p_http = base_port + r * 20
                p_master = 32300 + r * 30
                sb0 = 29300 + r * 10
                tmp_r = Path(f"/tmp/vo_u2_rep{r}")
                tmp_r.mkdir(parents=True, exist_ok=True)
                cfg_p = Path(f"/tmp/deploy_{group_id}_rep{r}.yaml")
                _write_deploy_yaml(
                    cfg_p,
                    ulysses_degree=2,
                    max_num_seqs=1,
                    attention_backend=attention_backend,
                    quantization=quantization,
                    devices=f"{2 * r},{2 * r + 1}",
                )
                lp = Path(f"/tmp/{group_id}_rep{r}.log")
                elp = Path(f"/tmp/{group_id}_enc_rep{r}.jsonl")
                extra_env = {
                    "VLLM_OMNI_LOCAL_RANKS": f"{2 * r},{2 * r + 1}",
                    "VLLM_OMNI_TOPOLOGY": "1,1,1,2",
                    "VLLM_OMNI_SLICEBUILDER_ADDRESSES": f"localhost:{sb0},localhost:{sb0 + 1}",
                    "MASTER_PORT": str(p_master),
                    "OMNI_MASTER_PORT": str(42300 + r * 20),
                    "TPU_RUNTIME_METRICS_PORTS": f"{8431 + 2 * r},{8432 + 2 * r}",
                    "TMPDIR": str(tmp_r),
                    "TPU_VISIBLE_CHIPS": f"{2 * r},{2 * r + 1}",
                }
                proc = _spawn_vllm_omni_proc(model_path, p_http, cfg_p, lp, elp, remote_encode_url, extra_env)
                procs.append(proc)
                ports.append(p_http)
                log_paths.append(lp)
                enc_logs.append(elp)
                time.sleep(2.5)
            boot_s = _wait_all_healthy(procs, ports, log_paths, boot_timeout_s=900.0)
            endpoints = [f"http://127.0.0.1:{p}" for p in ports]
            log(f"DP=4 x Ulysses=2 isolated 4-server cluster ready in {boot_s}s")
            return VllmOmniVideoCluster(procs, endpoints, log_paths, enc_logs, boot_s, mode="subhost_4x_u2")
        except Exception as sub_exc:
            log(f"subhost_4x_u2 failed ({str(sub_exc)[:300]}); falling back to single-server num_replicas=4...")
            stop_cluster_cleanly(procs)

        cfg_p = Path(f"/tmp/deploy_{group_id}_nr4.yaml")
        _write_deploy_yaml(
            cfg_p,
            ulysses_degree=2,
            max_num_seqs=1,
            attention_backend=attention_backend,
            quantization=quantization,
            num_replicas=4,
            devices="0,1,2,3,4,5,6,7",
        )
        lp = Path(f"/tmp/{group_id}_nr4.log")
        elp = Path(f"/tmp/{group_id}_enc_nr4.jsonl")
        proc = _spawn_vllm_omni_proc(
            model_path,
            base_port,
            cfg_p,
            lp,
            elp,
            remote_encode_url,
            {"TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438"},
        )
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=960.0)
        return VllmOmniVideoCluster(
            [proc], [f"http://127.0.0.1:{base_port}"], [lp], [elp], boot_s, mode="single_server_num_replicas_4_u2"
        )

    # DP=2 x Ulysses=4 -> 2 isolated 2-chip (4-TensorCore) servers on [0,1,2,3] & [4,5,6,7]
    if dp_size == 2 and ulysses_degree == 4:
        for topo_try in ("1,2,1,2", "2,1,1,2"):
            procs = []
            ports = []
            log_paths = []
            enc_logs = []
            try:
                for r, lrs in enumerate(("0,1,2,3", "4,5,6,7")):
                    p_http = base_port + r * 20
                    p_master = 32500 + r * 30
                    sb0 = 29500 + r * 10
                    tmp_r = Path(f"/tmp/vo_u4_rep{r}")
                    tmp_r.mkdir(parents=True, exist_ok=True)
                    t_slug = topo_try.replace(",", "")
                    cfg_p = Path(f"/tmp/deploy_{group_id}_{t_slug}_rep{r}.yaml")
                    _write_deploy_yaml(
                        cfg_p,
                        ulysses_degree=4,
                        max_num_seqs=1,
                        attention_backend=attention_backend,
                        quantization=quantization,
                        devices=lrs,
                    )
                    lp = Path(f"/tmp/{group_id}_{t_slug}_rep{r}.log")
                    elp = Path(f"/tmp/{group_id}_enc_{t_slug}_rep{r}.jsonl")
                    extra_env = {
                        "VLLM_OMNI_LOCAL_RANKS": lrs,
                        "VLLM_OMNI_TOPOLOGY": topo_try,
                        "VLLM_OMNI_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{sb0 + k}" for k in range(4)),
                        "MASTER_PORT": str(p_master),
                        "OMNI_MASTER_PORT": str(42500 + r * 20),
                        "TPU_RUNTIME_METRICS_PORTS": ",".join(str(8431 + 4 * r + k) for k in range(4)),
                        "TMPDIR": str(tmp_r),
                        "TPU_VISIBLE_CHIPS": lrs,
                    }
                    proc = _spawn_vllm_omni_proc(model_path, p_http, cfg_p, lp, elp, remote_encode_url, extra_env)
                    procs.append(proc)
                    ports.append(p_http)
                    log_paths.append(lp)
                    enc_logs.append(elp)
                    time.sleep(2.5)
                boot_s = _wait_all_healthy(procs, ports, log_paths, boot_timeout_s=720.0)
                endpoints = [f"http://127.0.0.1:{p}" for p in ports]
                log(f"DP=2 x Ulysses=4 isolated 2-server cluster ({topo_try}) ready in {boot_s}s")
                return VllmOmniVideoCluster(
                    procs, endpoints, log_paths, enc_logs, boot_s, mode=f"subhost_2x_u4_{topo_try}"
                )
            except Exception as sub_exc:
                log(f"subhost_2x_u4 ({topo_try}) failed ({str(sub_exc)[:280]}); cleaning up...")
                stop_cluster_cleanly(procs)

        cfg_p = Path(f"/tmp/deploy_{group_id}_nr2.yaml")
        _write_deploy_yaml(
            cfg_p,
            ulysses_degree=4,
            max_num_seqs=1,
            attention_backend=attention_backend,
            quantization=quantization,
            num_replicas=2,
            devices="0,1,2,3,4,5,6,7",
        )
        lp = Path(f"/tmp/{group_id}_nr2.log")
        elp = Path(f"/tmp/{group_id}_enc_nr2.jsonl")
        proc = _spawn_vllm_omni_proc(
            model_path,
            base_port,
            cfg_p,
            lp,
            elp,
            remote_encode_url,
            {
                "VLLM_OMNI_TOPOLOGY": "1,2,1,2",
                "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438",
            },
        )
        boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=900.0)
        return VllmOmniVideoCluster(
            [proc], [f"http://127.0.0.1:{base_port}"], [lp], [elp], boot_s, mode="single_server_num_replicas_2_u4"
        )

    # DP=1 x Ulysses=8 -> Single 8-TensorCore server across all 4 physical v7x chips
    cfg_p = Path(f"/tmp/deploy_{group_id}_u8.yaml")
    _write_deploy_yaml(
        cfg_p,
        ulysses_degree=ulysses_degree,
        max_num_seqs=1,
        attention_backend=attention_backend,
        quantization=quantization,
    )
    lp = Path(f"/tmp/{group_id}_u8.log")
    elp = Path(f"/tmp/{group_id}_enc_u8.jsonl")
    proc = _spawn_vllm_omni_proc(
        model_path,
        base_port,
        cfg_p,
        lp,
        elp,
        remote_encode_url,
        {
            "MASTER_PORT": "32700",
            "OMNI_MASTER_PORT": "42700",
            "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434,8435,8436,8437,8438",
        },
    )
    boot_s = _wait_all_healthy([proc], [base_port], [lp], boot_timeout_s=960.0)
    return VllmOmniVideoCluster([proc], [f"http://127.0.0.1:{base_port}"], [lp], [elp], boot_s, mode="single_u8")


@dataclass(frozen=True)
class VideoWorkload:
    workload_id: str
    label: str
    task: str  # "t2va", "i2va", "t2v", "i2v"
    size: str
    num_frames: int
    fps: int
    steps: int
    guidance_scale: float
    concurrencies: tuple[int, ...] = (1,)
    warm_iters_c1: int = 2


@dataclass(frozen=True)
class SuiteConfigSpec:
    config_name: str
    model_id: str
    precision: str
    dp_size: int
    ulysses_degree: int
    attention_backend: str
    quantization: str | None
    disagg_encode: bool
    workloads: tuple[VideoWorkload, ...]


def percentile_p90(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 4)
    pos = 0.90 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(s[lo], 4)
    return round(s[lo] + (pos - lo) * (s[hi] - s[lo]), 4)


def persist_rows(
    bucket: str,
    gcs_prefix: str,
    pod_idx: int,
    rows: list[dict[str, Any]],
    extra_meta: dict[str, Any],
) -> None:
    local_jsonl = Path(f"/tmp/suite1_pod{pod_idx}.jsonl")
    jsonl_bytes = ("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")).encode("utf-8")
    local_jsonl.write_bytes(jsonl_bytes)
    gcs_upload_bytes(bucket, f"{gcs_prefix}/results/suite1_pod{pod_idx}.jsonl", jsonl_bytes, "application/x-ndjson")

    summary = {
        "updated_utc": utc_now(),
        "pod_index": pod_idx,
        "hostname": socket.getfqdn(),
        "total_rows": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "failed_rows": sum(1 for r in rows if r.get("status") != "ok"),
        **extra_meta,
        "rows": rows,
    }
    sum_bytes = (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    Path(f"/tmp/suite1_pod{pod_idx}_summary.json").write_bytes(sum_bytes)
    gcs_upload_bytes(bucket, f"{gcs_prefix}/results/suite1_pod{pod_idx}_summary.json", sum_bytes, "application/json")


def load_prior_rows(bucket: str, gcs_prefix: str, pod_idx: int) -> list[dict[str, Any]]:
    raw = gcs_download_bytes(bucket, f"{gcs_prefix}/results/suite1_pod{pod_idx}.jsonl")
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for ln in raw.decode("utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            if r.get("status") == "ok":
                rows.append(r)
        except Exception:
            pass
    if rows:
        log(f"Resumed {len(rows)} previously completed 'ok' benchmark rows from GCS")
    return rows


def run_suite_config(
    *,
    spec: SuiteConfigSpec,
    model_path: str,
    prompts: list[str],
    refs_png: dict[str, bytes],
    encode_svc_url: str,
    base_port: int,
    bucket: str,
    gcs_prefix: str,
    pod_idx: int,
    rows: list[dict[str, Any]],
    completed_keys: set[tuple[str, str, int]],
    extra_meta: dict[str, Any],
) -> None:
    pending = [
        (wl, c)
        for wl in spec.workloads
        for c in wl.concurrencies
        if (spec.config_name, wl.workload_id, c) not in completed_keys
    ]
    if not pending:
        log(f"[{spec.config_name}] All workloads already completed in prior checkpoint; skipping.")
        return

    remote_url = encode_svc_url if spec.disagg_encode else ""
    hbm_mon = LibtpuHbmMonitor()
    cluster: VllmOmniVideoCluster | None = None

    def _launch() -> VllmOmniVideoCluster:
        return start_vllm_omni_video_cluster(
            model_path=model_path,
            base_port=base_port,
            dp_size=spec.dp_size,
            ulysses_degree=spec.ulysses_degree,
            attention_backend=spec.attention_backend,
            quantization=spec.quantization,
            remote_encode_url=remote_url,
            group_id=spec.config_name,
        )

    try:
        hbm_mon.start()
        cluster = _launch()
        warmed_shapes: dict[tuple[str, str, int, bool], float] = {}
        ffprobed_once = False

        for w_idx, wl in enumerate(spec.workloads):
            if all((spec.config_name, wl.workload_id, c) in completed_keys for c in wl.concurrencies):
                continue

            if cluster.any_proc_dead():
                log(f"[{spec.config_name}] Detected dead worker before {wl.workload_id}; restarting cluster...")
                cluster.stop()
                cluster = _launch()
                warmed_shapes.clear()

            is_img_cond = wl.task in ("i2va", "i2v")
            ref_bytes = refs_png.get(wl.size) if is_img_cond else None
            shape_key = (wl.task, wl.size, wl.num_frames, wl.guidance_scale > 1.0)

            # Warm all cluster replicas on this shape in parallel so XLA compilation happens once concurrently
            cold_e2e_s = warmed_shapes.get(shape_key)
            if cold_e2e_s is None:
                n_rep = len(cluster.endpoints)
                log(f"[{spec.config_name}] Cold warmup ({n_rep} replica(s)) for {wl.label} [{wl.size}, {wl.num_frames}f]...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_rep) as wpool:
                    wfuts = [
                        wpool.submit(
                            cluster.generate_single,
                            prompt=prompts[0],
                            size=wl.size,
                            num_frames=wl.num_frames,
                            fps=wl.fps,
                            steps=wl.steps,
                            guidance_scale=wl.guidance_scale,
                            seed=1000 + w_idx * 10 + rep_i,
                            ref_png_bytes=ref_bytes,
                            replica_override=rep_i,
                            run_ffprobe=(rep_i == 0 and not ffprobed_once),
                        )
                        for rep_i in range(n_rep)
                    ]
                    w_outs = [f.result() for f in wfuts]

                if not all(o.get("ok") for o in w_outs):
                    err_msg = next((o.get("error") for o in w_outs if not o.get("ok")), "warmup failed")
                    log(f"[{spec.config_name}] Warmup FAILED on {wl.label}: {err_msg}")
                    peak_hbm = hbm_mon.snapshot_peak_gib() or parse_hbm_from_logs(cluster.log_paths)
                    for conc in wl.concurrencies:
                        if (spec.config_name, wl.workload_id, conc) in completed_keys:
                            continue
                        fail_row = {
                            "ts_utc": utc_now(),
                            "pod_index": pod_idx,
                            "config": spec.config_name,
                            "model": spec.model_id,
                            "precision": spec.precision,
                            "dp_size": spec.dp_size,
                            "ulysses_degree": spec.ulysses_degree,
                            "cluster_mode": cluster.mode,
                            "attention_backend": spec.attention_backend,
                            "encode_mode": "disagg_cpu_svc" if spec.disagg_encode else "local_pyav",
                            "workload_id": wl.workload_id,
                            "workload": wl.label,
                            "task": wl.task,
                            "size": wl.size,
                            "num_frames": wl.num_frames,
                            "fps": wl.fps,
                            "duration_s": round((wl.num_frames - 1) / max(wl.fps, 1), 2),
                            "steps": wl.steps,
                            "guidance_scale": wl.guidance_scale,
                            "concurrency": conc,
                            "status": "error",
                            "cold_e2e_s": w_outs[0].get("e2e_s") if w_outs else None,
                            "warm_median_e2e_s": None,
                            "warm_mean_e2e_s": None,
                            "p90_e2e_s": None,
                            "server_inference_s": None,
                            "encode_s": None,
                            "ms_per_step": None,
                            "host_rpm": 0.0,
                            "slice16_projected_rpm": 0.0,
                            "peak_hbm_gib": peak_hbm,
                            "mp4_bytes": None,
                            "has_audio": None,
                            "error": err_msg,
                        }
                        rows.append(fail_row)
                        persist_rows(bucket, gcs_prefix, pod_idx, rows, extra_meta)
                    continue

                if not ffprobed_once:
                    ffprobed_once = True
                cold_e2e_s = round(max(float(o["e2e_s"]) for o in w_outs), 4)
                warmed_shapes[shape_key] = cold_e2e_s
                log(f"[{spec.config_name}] Warmup OK for {wl.label}: cold_e2e_s={cold_e2e_s}s")

            for conc in wl.concurrencies:
                if (spec.config_name, wl.workload_id, conc) in completed_keys:
                    continue

                n_reqs = wl.warm_iters_c1 if conc == 1 else max(conc * 2, conc)
                # Keep ultra-heavy 481-frame multi-concurrency waves bounded in wall time while filling all workers
                if wl.num_frames >= 481 and conc > 1:
                    n_reqs = conc

                hbm_mon.reset_window()
                t_wave0 = time.perf_counter()
                if conc == 1:
                    outs = [
                        cluster.generate_single(
                            prompt=prompts[(w_idx + i + 1) % len(prompts)],
                            size=wl.size,
                            num_frames=wl.num_frames,
                            fps=wl.fps,
                            steps=wl.steps,
                            guidance_scale=wl.guidance_scale,
                            seed=5000 + w_idx * 100 + i,
                            ref_png_bytes=ref_bytes,
                            run_ffprobe=(i == 0),
                        )
                        for i in range(n_reqs)
                    ]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                        futs = [
                            pool.submit(
                                cluster.generate_single,
                                prompt=prompts[(w_idx + i + 1) % len(prompts)],
                                size=wl.size,
                                num_frames=wl.num_frames,
                                fps=wl.fps,
                                steps=wl.steps,
                                guidance_scale=wl.guidance_scale,
                                seed=7000 + conc * 100 + w_idx * 50 + i,
                                ref_png_bytes=ref_bytes,
                                run_ffprobe=(i == 0),
                            )
                            for i in range(n_reqs)
                        ]
                        outs = [f.result() for f in futs]
                wave_wall_s = time.perf_counter() - t_wave0

                oks = [o for o in outs if o.get("ok")]
                peak_hbm = hbm_mon.snapshot_peak_gib() or parse_hbm_from_logs(cluster.log_paths)

                if oks:
                    e2es = [float(o["e2e_s"]) for o in oks]
                    srv_infs = [float(o["server_inference_s"]) for o in oks if o.get("server_inference_s") is not None]
                    enc_vals = [float(o["encode_s"]) for o in oks if o.get("encode_s") is not None]

                    med_e2e = round(statistics.median(e2es), 4)
                    mean_e2e = round(statistics.mean(e2es), 4)
                    p90_e2e = percentile_p90(e2es)
                    med_srv_inf = round(statistics.median(srv_infs), 4) if srv_infs else med_e2e
                    med_enc_s = round(statistics.median(enc_vals), 4) if enc_vals else None

                    pure_model_s = (
                        max(0.01, med_srv_inf - med_enc_s) if (med_enc_s is not None and med_srv_inf > med_enc_s) else med_srv_inf
                    )
                    ms_step = round((pure_model_s * 1000.0) / max(wl.steps, 1), 2)
                    host_rpm = round((len(oks) / max(wave_wall_s, 1e-6)) * 60.0, 3)
                    slice16_rpm = round(4.0 * host_rpm, 3)
                    mp4_b = int(statistics.median([int(o["mp4_bytes"]) for o in oks]))
                    has_aud = next((o.get("has_audio") for o in oks if o.get("has_audio") is not None), None)
                    status = "ok" if len(oks) == n_reqs else "partial_ok"
                    err_str = None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None)
                else:
                    med_e2e = mean_e2e = p90_e2e = med_srv_inf = med_enc_s = ms_step = None
                    host_rpm = slice16_rpm = 0.0
                    mp4_b = None
                    has_aud = None
                    status = "error"
                    err_str = next((o.get("error") for o in outs if not o.get("ok")), "all requests failed")

                row = {
                    "ts_utc": utc_now(),
                    "pod_index": pod_idx,
                    "config": spec.config_name,
                    "model": spec.model_id,
                    "precision": spec.precision,
                    "dp_size": spec.dp_size,
                    "ulysses_degree": spec.ulysses_degree,
                    "cluster_mode": cluster.mode,
                    "attention_backend": spec.attention_backend,
                    "encode_mode": "disagg_cpu_svc" if spec.disagg_encode else "local_pyav",
                    "workload_id": wl.workload_id,
                    "workload": wl.label,
                    "task": wl.task,
                    "size": wl.size,
                    "num_frames": wl.num_frames,
                    "fps": wl.fps,
                    "duration_s": round((wl.num_frames - 1) / max(wl.fps, 1), 2),
                    "steps": wl.steps,
                    "guidance_scale": wl.guidance_scale,
                    "concurrency": conc,
                    "n_requests": n_reqs,
                    "ok_count": len(oks),
                    "status": status,
                    "cold_e2e_s": cold_e2e_s,
                    "warm_median_e2e_s": med_e2e,
                    "warm_mean_e2e_s": mean_e2e,
                    "p90_e2e_s": p90_e2e,
                    "server_inference_s": med_srv_inf,
                    "encode_s": med_enc_s,
                    "ms_per_step": ms_step,
                    "host_rpm": host_rpm,
                    "slice16_projected_rpm": slice16_rpm,
                    "peak_hbm_gib": peak_hbm,
                    "mp4_bytes": mp4_b,
                    "has_audio": has_aud,
                    "wave_wall_s": round(wave_wall_s, 4),
                    "server_boot_s": cluster.boot_s,
                    "error": err_str,
                }
                log(
                    f"[{spec.config_name}] {wl.label} (c={conc}): status={status} med={med_e2e}s "
                    f"enc={med_enc_s}s step={ms_step}ms host_rpm={host_rpm} slice16_rpm={slice16_rpm} "
                    f"hbm={peak_hbm}GiB audio={has_aud}"
                )
                rows.append(row)
                if status == "ok":
                    completed_keys.add((spec.config_name, wl.workload_id, conc))
                persist_rows(bucket, gcs_prefix, pod_idx, rows, extra_meta)

    except Exception as cfg_exc:
        tb = traceback.format_exc()[-1800:]
        log(f"[{spec.config_name}] CONFIG EXCEPTION: {cfg_exc}\n{tb}")
        rows.append(
            {
                "ts_utc": utc_now(),
                "pod_index": pod_idx,
                "config": spec.config_name,
                "model": spec.model_id,
                "precision": spec.precision,
                "dp_size": spec.dp_size,
                "ulysses_degree": spec.ulysses_degree,
                "attention_backend": spec.attention_backend,
                "encode_mode": "disagg_cpu_svc" if spec.disagg_encode else "local_pyav",
                "workload_id": "__cluster_boot__",
                "workload": "Cluster Boot / Setup",
                "concurrency": 0,
                "status": "cluster_error",
                "cold_e2e_s": None,
                "warm_median_e2e_s": None,
                "warm_mean_e2e_s": None,
                "p90_e2e_s": None,
                "server_inference_s": None,
                "encode_s": None,
                "ms_per_step": None,
                "host_rpm": 0.0,
                "slice16_projected_rpm": 0.0,
                "peak_hbm_gib": None,
                "mp4_bytes": None,
                "has_audio": None,
                "error": f"{cfg_exc}\n{tb}",
            }
        )
        persist_rows(bucket, gcs_prefix, pod_idx, rows, extra_meta)
    finally:
        hbm_mon.stop()
        stop_cluster_cleanly(cluster.procs if cluster is not None else None)


def build_pod0_ltx25_specs() -> list[SuiteConfigSpec]:
    m = "Lightricks/LTX-2.5"
    w_544_5s_t2va = VideoWorkload("ltx_544p_5s_121f_t2va_8s", "544p (960x544) 5s 121f T2VA (8-step)", "t2va", "960x544", 121, 24, 8, 1.0, (1,), 3)
    w_544_5s_i2va = VideoWorkload("ltx_544p_5s_121f_i2va_8s", "544p (960x544) 5s 121f I2VA (8-step)", "i2va", "960x544", 121, 24, 8, 1.0, (1,), 2)
    w_544_10s_t2va = VideoWorkload("ltx_544p_10s_241f_t2va_8s", "544p (960x544) 10s 241f T2VA (8-step)", "t2va", "960x544", 241, 24, 8, 1.0, (1,), 2)
    w_544_20s_t2va = VideoWorkload("ltx_544p_20s_481f_t2va_8s", "544p (960x544) 20s 481f T2VA (8-step)", "t2va", "960x544", 481, 24, 8, 1.0, (1,), 2)

    w_720_5s_t2va = VideoWorkload("ltx_720p_5s_121f_t2va_8s", "720p (1280x704) 5s 121f T2VA (8-step)", "t2va", "1280x704", 121, 24, 8, 1.0, (1,), 3)
    w_720_5s_i2va = VideoWorkload("ltx_720p_5s_121f_i2va_8s", "720p (1280x704) 5s 121f I2VA (8-step)", "i2va", "1280x704", 121, 24, 8, 1.0, (1,), 2)
    w_720_10s_t2va = VideoWorkload("ltx_720p_10s_241f_t2va_8s", "720p (1280x704) 10s 241f T2VA (8-step)", "t2va", "1280x704", 241, 24, 8, 1.0, (1,), 2)
    w_720_5s_20step = VideoWorkload("ltx_720p_5s_121f_t2va_20s", "720p (1280x704) 5s 121f T2VA (20-step CFG=3.0)", "t2va", "1280x704", 121, 24, 20, 3.0, (1,), 2)

    w_1080_5s_t2va = VideoWorkload("ltx_1080p_5s_121f_t2va_8s", "1080p (1920x1088) 5s 121f T2VA (8-step)", "t2va", "1920x1088", 121, 24, 8, 1.0, (1,), 2)
    w_1080_20s_c1 = VideoWorkload("ltx_1080p_20s_481f_t2va_8s", "1080p Kling-V4 Proxy (1920x1088) 20s 481f T2VA (8-step)", "t2va", "1920x1088", 481, 24, 8, 1.0, (1,), 2)
    w_1080_20s_c12 = VideoWorkload("ltx_1080p_20s_481f_t2va_8s", "1080p Kling-V4 Proxy (1920x1088) 20s 481f T2VA (8-step)", "t2va", "1920x1088", 481, 24, 8, 1.0, (1, 2), 2)

    def _with_concs(wls: tuple[VideoWorkload, ...], concs: tuple[int, ...]) -> tuple[VideoWorkload, ...]:
        return tuple(
            VideoWorkload(w.workload_id, w.label, w.task, w.size, w.num_frames, w.fps, w.steps, w.guidance_scale, concs, 2)
            for w in wls
        )

    dp_base = (w_544_5s_t2va, w_720_5s_t2va, w_1080_5s_t2va, w_1080_20s_c1)

    return [
        SuiteConfigSpec(
            config_name="omni_ltx25_bf16_u8_sdpa_localenc",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TORCH_SDPA",
            quantization=None,
            disagg_encode=False,
            workloads=(w_544_5s_t2va, w_720_5s_t2va, w_1080_20s_c1),
        ),
        SuiteConfigSpec(
            config_name="omni_ltx25_bf16_u8_flash_localenc",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization=None,
            disagg_encode=False,
            workloads=(w_544_5s_t2va, w_720_5s_t2va, w_1080_20s_c1),
        ),
        SuiteConfigSpec(
            config_name="omni_ltx25_bf16_u8_flash_disagg",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization=None,
            disagg_encode=True,
            workloads=(w_544_5s_t2va, w_720_5s_t2va, w_1080_5s_t2va, w_1080_20s_c1),
        ),
        SuiteConfigSpec(
            config_name="omni_ltx25_fp8_u8_flash_disagg",
            model_id=m,
            precision="FP8",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=(
                w_544_5s_t2va,
                w_544_5s_i2va,
                w_544_10s_t2va,
                w_544_20s_t2va,
                w_720_5s_t2va,
                w_720_5s_i2va,
                w_720_10s_t2va,
                w_720_5s_20step,
                w_1080_5s_t2va,
                w_1080_20s_c12,
            ),
        ),
        SuiteConfigSpec(
            config_name="omni_ltx25_fp8_dp2_u4_disagg",
            model_id=m,
            precision="FP8",
            dp_size=2,
            ulysses_degree=4,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs(dp_base, (1, 2, 4)),
        ),
        SuiteConfigSpec(
            config_name="omni_ltx25_fp8_dp4_u2_disagg",
            model_id=m,
            precision="FP8",
            dp_size=4,
            ulysses_degree=2,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs(dp_base, (1, 4, 8)),
        ),
    ]


def build_pod1_wan22_t2v_specs() -> list[SuiteConfigSpec]:
    m = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    w_480_4s = VideoWorkload("wan_t2v_480p_5s_81f_4s", "480p (832x480) 5s 81f T2V (4-step)", "t2v", "832x480", 81, 16, 4, 1.0, (1,), 3)
    w_480_20s = VideoWorkload("wan_t2v_480p_5s_81f_20s", "480p (832x480) 5s 81f T2V (20-step)", "t2v", "832x480", 81, 16, 20, 3.5, (1,), 2)
    w_720_4s = VideoWorkload("wan_t2v_720p_5s_81f_4s", "720p (1280x720) 5s 81f T2V (4-step)", "t2v", "1280x720", 81, 16, 4, 1.0, (1,), 3)
    w_720_20s = VideoWorkload("wan_t2v_720p_5s_81f_20s", "720p (1280x720) 5s 81f T2V (20-step)", "t2v", "1280x720", 81, 16, 20, 3.5, (1,), 2)
    w_720_10s_20s = VideoWorkload("wan_t2v_720p_10s_161f_20s", "720p (1280x720) 10s 161f T2V (20-step)", "t2v", "1280x720", 161, 16, 20, 3.5, (1,), 2)
    w_720p_port_20s = VideoWorkload("wan_t2v_720p_portrait_5s_81f_20s", "720p Portrait (720x1280) 5s 81f T2V (20-step)", "t2v", "720x1280", 81, 16, 20, 3.5, (1,), 2)

    def _with_concs(wls: tuple[VideoWorkload, ...], concs: tuple[int, ...]) -> tuple[VideoWorkload, ...]:
        return tuple(
            VideoWorkload(w.workload_id, w.label, w.task, w.size, w.num_frames, w.fps, w.steps, w.guidance_scale, concs, 2)
            for w in wls
        )

    return [
        SuiteConfigSpec(
            config_name="omni_wan22_t2v_bf16_u8_localenc",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization=None,
            disagg_encode=False,
            workloads=(w_480_4s, w_480_20s, w_720_4s, w_720_20s),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_t2v_bf16_u8_disagg",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization=None,
            disagg_encode=True,
            workloads=(w_480_4s, w_480_20s, w_720_4s, w_720_20s),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_t2v_fp8_u8_disagg",
            model_id=m,
            precision="FP8",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=(w_480_4s, w_480_20s, w_720_4s, w_720_20s, w_720_10s_20s, w_720p_port_20s),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_t2v_fp8_dp2_u4_disagg",
            model_id=m,
            precision="FP8",
            dp_size=2,
            ulysses_degree=4,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs((w_480_4s, w_480_20s, w_720_4s, w_720_20s), (1, 2, 4)),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_t2v_fp8_dp4_u2_disagg",
            model_id=m,
            precision="FP8",
            dp_size=4,
            ulysses_degree=2,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs((w_480_4s, w_720_4s, w_720_20s), (1, 4, 8)),
        ),
    ]


def build_pod1_wan22_i2v_specs() -> list[SuiteConfigSpec]:
    m = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    w_480_4s = VideoWorkload("wan_i2v_480p_5s_81f_4s", "480p (832x480) 5s 81f I2V (4-step)", "i2v", "832x480", 81, 16, 4, 1.0, (1,), 3)
    w_480_20s = VideoWorkload("wan_i2v_480p_5s_81f_20s", "480p (832x480) 5s 81f I2V (20-step)", "i2v", "832x480", 81, 16, 20, 3.5, (1,), 2)
    w_720_4s = VideoWorkload("wan_i2v_720p_5s_81f_4s", "720p (1280x720) 5s 81f I2V (4-step)", "i2v", "1280x720", 81, 16, 4, 1.0, (1,), 3)
    w_720_20s = VideoWorkload("wan_i2v_720p_5s_81f_20s", "720p (1280x720) 5s 81f I2V (20-step)", "i2v", "1280x720", 81, 16, 20, 3.5, (1,), 2)
    w_720_10s_20s = VideoWorkload("wan_i2v_720p_10s_161f_20s", "720p (1280x720) 10s 161f I2V (20-step)", "i2v", "1280x720", 161, 16, 20, 3.5, (1,), 2)
    w_720p_port_20s = VideoWorkload("wan_i2v_720p_portrait_5s_81f_20s", "720p Portrait (720x1280) 5s 81f I2V (20-step)", "i2v", "720x1280", 81, 16, 20, 3.5, (1,), 2)

    def _with_concs(wls: tuple[VideoWorkload, ...], concs: tuple[int, ...]) -> tuple[VideoWorkload, ...]:
        return tuple(
            VideoWorkload(w.workload_id, w.label, w.task, w.size, w.num_frames, w.fps, w.steps, w.guidance_scale, concs, 2)
            for w in wls
        )

    return [
        SuiteConfigSpec(
            config_name="omni_wan22_i2v_bf16_u8_disagg",
            model_id=m,
            precision="BF16",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization=None,
            disagg_encode=True,
            workloads=(w_480_4s, w_480_20s, w_720_4s, w_720_20s, w_720_10s_20s, w_720p_port_20s),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_i2v_fp8_u8_disagg",
            model_id=m,
            precision="FP8",
            dp_size=1,
            ulysses_degree=8,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=(w_480_4s, w_480_20s, w_720_4s, w_720_20s, w_720_10s_20s, w_720p_port_20s),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_i2v_fp8_dp2_u4_disagg",
            model_id=m,
            precision="FP8",
            dp_size=2,
            ulysses_degree=4,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs((w_480_4s, w_480_20s, w_720_4s, w_720_20s), (1, 2, 4)),
        ),
        SuiteConfigSpec(
            config_name="omni_wan22_i2v_fp8_dp4_u2_disagg",
            model_id=m,
            precision="FP8",
            dp_size=4,
            ulysses_degree=2,
            attention_backend="TPU_FLASH_ATTN",
            quantization="fp8",
            disagg_encode=True,
            workloads=_with_concs((w_480_4s, w_720_4s, w_720_20s), (1, 4, 8)),
        ),
    ]


def main() -> None:
    global _POD_IDX

    ap = argparse.ArgumentParser(description="Suite 1 TPU v7x Video+Audio Optimization on vLLM-Omni")
    ap.add_argument("--gcs-bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    ap.add_argument("--encode-svc-url", default=DEFAULT_ENCODE_SVC_URL)
    ap.add_argument("--pod-index", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    args = ap.parse_args()

    _POD_IDX = args.pod_index
    log(f"Starting Suite 1 on Pod {_POD_IDX} (host={socket.getfqdn()}, {host_ram_summary()})")

    subprocess.run(["bash", "-lc", "mount -o remount,size=900G /dev/shm || true"], check=False)
    stop_cluster_cleanly(None)
    ensure_subhost_isolation_and_patches_ready()
    enc_health = probe_remote_encoder_health(args.encode_svc_url)

    rows = load_prior_rows(args.gcs_bucket, args.gcs_prefix, _POD_IDX)
    completed_keys: set[tuple[str, str, int]] = {
        (str(r["config"]), str(r["workload_id"]), int(r["concurrency"]))
        for r in rows
        if r.get("status") == "ok" and "workload_id" in r and "concurrency" in r
    }
    extra_meta = {
        "cluster": "tpu-v7x-cluster",
        "nodepool": "tpu-v7x-cluster-8-pool-a",
        "topology": "2x2x2 (isolated single-host 4x v7x chips = 8 TensorCores)",
        "encode_svc_url": args.encode_svc_url,
        "encode_svc_health": enc_health,
    }

    base_port = 18100 + _POD_IDX * 200

    if _POD_IDX == 0:
        refs_png = generate_synthetic_refs(
            Path("/dev/shm/suite1_refs_pod0"), ("960x544", "1280x704", "1920x1088"), "ltx_ref"
        )
        ltx_specs = build_pod0_ltx25_specs()
        model_path = stage_ltx25_model(args.gcs_bucket)
        for spec in ltx_specs:
            run_suite_config(
                spec=spec,
                model_path=model_path,
                prompts=LTX_PROMPTS,
                refs_png=refs_png,
                encode_svc_url=args.encode_svc_url,
                base_port=base_port,
                bucket=args.gcs_bucket,
                gcs_prefix=args.gcs_prefix,
                pod_idx=_POD_IDX,
                rows=rows,
                completed_keys=completed_keys,
                extra_meta=extra_meta,
            )
    elif _POD_IDX == 1:
        refs_png = generate_synthetic_refs(
            Path("/dev/shm/suite1_refs_pod1"), ("832x480", "1280x720", "720x1280"), "wan_ref"
        )

        # Phase A: Wan 2.2 T2V-A14B
        t2v_specs = build_pod1_wan22_t2v_specs()
        t2v_pending = any(
            (s.config_name, w.workload_id, c) not in completed_keys
            for s in t2v_specs
            for w in s.workloads
            for c in w.concurrencies
        )
        t2v_dir = Path("/dev/shm/Wan2.2-T2V-A14B-Diffusers")
        if t2v_pending:
            t2v_path = stage_wan22_model(args.gcs_bucket, "Wan-AI/Wan2.2-T2V-A14B-Diffusers", t2v_dir)
            for spec in t2v_specs:
                run_suite_config(
                    spec=spec,
                    model_path=t2v_path,
                    prompts=WAN_PROMPTS,
                    refs_png=refs_png,
                    encode_svc_url=args.encode_svc_url,
                    base_port=base_port,
                    bucket=args.gcs_bucket,
                    gcs_prefix=args.gcs_prefix,
                    pod_idx=_POD_IDX,
                    rows=rows,
                    completed_keys=completed_keys,
                    extra_meta=extra_meta,
                )

        # Phase B: Clean T2V from /dev/shm and stage Wan 2.2 I2V-A14B
        log(f"Cleaning {t2v_dir} before staging Wan2.2-I2V-A14B-Diffusers ({host_ram_summary()})...")
        stop_cluster_cleanly(None)
        shutil.rmtree(t2v_dir, ignore_errors=True)
        shutil.rmtree("/dev/shm/xla_cache_vllm_omni", ignore_errors=True)

        i2v_specs = build_pod1_wan22_i2v_specs()
        i2v_pending = any(
            (s.config_name, w.workload_id, c) not in completed_keys
            for s in i2v_specs
            for w in s.workloads
            for c in w.concurrencies
        )
        i2v_dir = Path("/dev/shm/Wan2.2-I2V-A14B-Diffusers")
        if i2v_pending:
            i2v_path = stage_wan22_model(args.gcs_bucket, "Wan-AI/Wan2.2-I2V-A14B-Diffusers", i2v_dir)
            for spec in i2v_specs:
                run_suite_config(
                    spec=spec,
                    model_path=i2v_path,
                    prompts=WAN_PROMPTS,
                    refs_png=refs_png,
                    encode_svc_url=args.encode_svc_url,
                    base_port=base_port,
                    bucket=args.gcs_bucket,
                    gcs_prefix=args.gcs_prefix,
                    pod_idx=_POD_IDX,
                    rows=rows,
                    completed_keys=completed_keys,
                    extra_meta=extra_meta,
                )
    else:
        log(f"Pod index {_POD_IDX} outside [0, 1]; exiting cleanly.")
        return

    persist_rows(args.gcs_bucket, args.gcs_prefix, _POD_IDX, rows, extra_meta)
    log(f"Suite 1 Pod {_POD_IDX} FINISHED! Total rows={len(rows)}, ok={sum(1 for r in rows if r.get('status') == 'ok')}")


if __name__ == "__main__":
    main()
