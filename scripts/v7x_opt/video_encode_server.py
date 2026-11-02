#!/usr/bin/env python3
"""Disaggregated High-Throughput CPU Video + Audio MP4 Encoding Service on GKE (`video-encode-cpu-pool`).

Runs on dedicated Intel Sapphire Rapids (`c3-standard-44`, 44 vCPUs / 176 GB DDR5 RAM) CPU nodes in `us-central1-c`
to offload post-VAE RGB frame formatting, H.264 (`libx264` AVX-512) encoding, and stereo AAC audio muxing completely
off the Cloud TPU `v7x` serving pods.

Wire Protocol (`POST /encode`):
  - Content-Type: `application/octet-stream`
  - Bytes `[0:4]`: big-endian uint32 `meta_len`
  - Bytes `[4:4+meta_len]`: UTF-8 JSON metadata:
      {
        "width": 1280,
        "height": 720,
        "num_frames": 81,
        "fps": 16,
        "pix_fmt": "rgb24",           # raw uint8 THWC bytes = num_frames * height * width * 3
        "preset": "ultrafast",        # libx264 preset (ultrafast / superfast / veryfast / fast)
        "crf": 20,
        "threads": 16,
        "audio_sr": 24000,            # optional int (0 or omitted if silent)
        "audio_channels": 2,          # optional int
        "audio_bytes": 0,             # byte length of trailing float32 PCM audio (`f32le` interleaved)
        "return_mp4": true            # if false, discards MP4 bytes on server and returns JSON stats only
      }
  - Bytes `[4+meta_len : 4+meta_len+video_bytes]`: raw `uint8` RGB24 frames (`T * H * W * 3`)
  - Bytes `[4+meta_len+video_bytes :]`: optional raw `float32` (`f32le`) interleaved audio samples
"""

from __future__ import annotations

import json
import os
import platform
import socket
import struct
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOSTNAME = socket.gethostname()
CPU_COUNT = os.cpu_count() or 1
SHM_DIR = Path("/dev/shm/video_encode_spool")
SHM_DIR.mkdir(parents=True, exist_ok=True)

_STATS_LOCK = threading.Lock()
_TOTAL_JOBS = 0
_TOTAL_FRAMES = 0
_TOTAL_ENCODE_S = 0.0


def _cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in ln:
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


CPU_MODEL = _cpu_model()


def encode_mp4_ffmpeg(
    video_rgb_bytes: memoryview | bytes,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    preset: str = "ultrafast",
    crf: int = 20,
    threads: int = 16,
    audio_pcm_bytes: memoryview | bytes | None = None,
    audio_sr: int = 0,
    audio_channels: int = 2,
) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=str(SHM_DIR), prefix="enc_") as tmpd:
        out_mp4 = os.path.join(tmpd, "out.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if audio_pcm_bytes and len(audio_pcm_bytes) > 0 and audio_sr > 0:
            wav_raw = os.path.join(tmpd, "audio.f32le")
            with open(wav_raw, "wb") as af:
                af.write(audio_pcm_bytes)
            cmd.extend(
                [
                    "-f",
                    "f32le",
                    "-ar",
                    str(audio_sr),
                    "-ac",
                    str(audio_channels),
                    "-i",
                    wav_raw,
                ]
            )
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(threads),
            ]
        )
        if audio_pcm_bytes and len(audio_pcm_bytes) > 0 and audio_sr > 0:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        cmd.extend(["-movflags", "+faststart", out_mp4])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        _, stderr_b = proc.communicate(input=video_rgb_bytes, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr_b.decode('utf-8', 'replace')[:600]}")
        mp4_data = Path(out_mp4).read_bytes()
    dt = time.perf_counter() - t0
    return mp4_data, dt


class EncodeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/health") or self.path.startswith("/stats"):
            with _STATS_LOCK:
                jobs, frames, enc_s = _TOTAL_JOBS, _TOTAL_FRAMES, _TOTAL_ENCODE_S
            self._send_json(
                200,
                {
                    "status": "ok",
                    "hostname": HOSTNAME,
                    "cpu_model": CPU_MODEL,
                    "vcpus": CPU_COUNT,
                    "total_jobs": jobs,
                    "total_frames": frames,
                    "total_encode_s": round(enc_s, 3),
                    "avg_fps": round(frames / enc_s, 1) if enc_s > 0 else 0.0,
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        global _TOTAL_JOBS, _TOTAL_FRAMES, _TOTAL_ENCODE_S
        if not self.path.startswith("/encode"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            clen = int(self.headers.get("Content-Length", "0"))
            if clen < 8:
                self._send_json(400, {"error": "payload too short"})
                return
            t_rx0 = time.perf_counter()
            buf = bytearray(clen)
            view = memoryview(buf)
            pos = 0
            while pos < clen:
                n = self.rfile.readinto(view[pos:])
                if not n:
                    break
                pos += n
            rx_s = time.perf_counter() - t_rx0
            if pos < clen:
                self._send_json(400, {"error": f"incomplete read {pos}/{clen}"})
                return

            meta_len = struct.unpack(">I", view[:4])[0]
            meta = json.loads(bytes(view[4 : 4 + meta_len]).decode("utf-8"))
            w = int(meta["width"])
            h = int(meta["height"])
            nf = int(meta["num_frames"])
            fps = float(meta.get("fps", 16))
            preset = str(meta.get("preset", "ultrafast"))
            crf = int(meta.get("crf", 20))
            threads = int(meta.get("threads", min(16, CPU_COUNT)))
            audio_sr = int(meta.get("audio_sr", 0) or 0)
            audio_ch = int(meta.get("audio_channels", 2) or 2)
            audio_len = int(meta.get("audio_bytes", 0) or 0)
            return_mp4 = bool(meta.get("return_mp4", True))

            vid_start = 4 + meta_len
            vid_len = nf * h * w * 3
            vid_end = vid_start + vid_len
            if vid_end > clen:
                self._send_json(400, {"error": f"video slice {vid_end} > content_length {clen}"})
                return
            vid_slice = view[vid_start:vid_end]
            aud_slice = view[vid_end : vid_end + audio_len] if audio_len > 0 else None

            mp4_bytes, enc_s = encode_mp4_ffmpeg(
                vid_slice,
                width=w,
                height=h,
                num_frames=nf,
                fps=fps,
                preset=preset,
                crf=crf,
                threads=threads,
                audio_pcm_bytes=aud_slice,
                audio_sr=audio_sr,
                audio_channels=audio_ch,
            )
            with _STATS_LOCK:
                _TOTAL_JOBS += 1
                _TOTAL_FRAMES += nf
                _TOTAL_ENCODE_S += enc_s

            if not return_mp4:
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "hostname": HOSTNAME,
                        "mp4_bytes": len(mp4_bytes),
                        "rx_ms": round(rx_s * 1000.0, 2),
                        "encode_ms": round(enc_s * 1000.0, 2),
                        "encode_fps": round(nf / max(enc_s, 1e-6), 1),
                    },
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(mp4_bytes)))
            self.send_header("X-Encoder-Host", HOSTNAME)
            self.send_header("X-Rx-Time-Ms", f"{rx_s * 1000.0:.2f}")
            self.send_header("X-Encode-Time-Ms", f"{enc_s * 1000.0:.2f}")
            self.send_header("X-Encode-FPS", f"{nf / max(enc_s, 1e-6):.1f}")
            self.end_headers()
            self.wfile.write(mp4_bytes)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)[:600]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8090"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), EncodeHandler)
    print(f"[video-encode-svc] Listening on 0.0.0.0:{port} host={HOSTNAME} cpu={CPU_MODEL} vcpus={CPU_COUNT}", flush=True)
    srv.serve_forever()
