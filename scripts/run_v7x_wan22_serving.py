#!/usr/bin/env python3
"""Priority #3: Wan 2.2 A14B (T2V-A14B & I2V-A14B MoE DiT) Video Serving Benchmark Suite on TPU v7x.

Covers single-host (4x TPU v7x chips = 8 TensorCores / 768 GiB HBM) and 16-chip Flex slice (`DP=4 x Ulysses=8`)
across `sglang-torchtpu` and `vllm-torchtpu` (`vLLM-Omni`):
  - Pod 0: `Wan-AI/Wan2.2-I2V-A14B-Diffusers` on `sglang-torchtpu` (BF16, Ulysses=8, Pallas FlashAttention)
  - Pod 1: `Wan-AI/Wan2.2-I2V-A14B-Diffusers` on `sglang-torchtpu` (Online FP8 `--quantization fp8`, `CFG=2 x Ulysses=4`)
  - Pod 2: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` on `sglang-torchtpu` (BF16 & Online FP8, Ulysses=8)
  - Pod 3: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` on `vllm-torchtpu` (`vLLM-Omni` `--omni --usp 8`, BF16)

Input Distributions Swept:
  - `832x480` (480p) `81` frames (`5s` @ 16fps) — 4-step fast, 20-step, 40-step standard
  - `1280x720` (720p) `81` frames (`5s` @ 16fps) — 4-step fast, 20-step, 40-step standard
  - `1280x720` (720p) `161` frames (`10s` @ 16fps) — 20-step long clip
  - `720x1280` (720p 9:16 Portrait) `81` frames (`5s` @ 16fps) — 20-step
  - Full 16-Chip Slice (`DP=4 x Ulysses=8`): `c=4, 8` throughput & 30-RPM slice sizing (no cost columns).
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic spray and sunlight.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, reflections on wet asphalt.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, crisp detail.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist at sunrise.",
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
    return len(blobs)


def setup_single_host_metadata_hook(bucket: str) -> None:
    gcs_download_file(bucket, "dsv41-stack/stack_27be32f509754e88.tar", Path("/tmp/stack.tar"))
    subprocess.run(
        [
            "bash",
            "-lc",
            r"""
set -euo pipefail
mkdir -p /workspace
tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar /workspace/mock_metadata.py /workspace/metadata_hook.c 2>/dev/null \
  || tar --overwrite --keep-directory-symlink -C / -xf /tmp/stack.tar workspace/mock_metadata.py workspace/metadata_hook.c
rm -f /tmp/stack.tar
gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl
grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts
""",
        ],
        check=True,
    )
    env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    mf = open("/workspace/mock_metadata.log", "w")
    subprocess.Popen(["python3", "/workspace/mock_metadata.py"], env=env_clean, stdout=mf, stderr=subprocess.STDOUT)
    time.sleep(1)


def ensure_engines(idx: int, need_vllm_omni: bool) -> None:
    chk = subprocess.run(["python3", "-c", "import shutil, sglang, sglang_tpu; assert shutil.which('ffmpeg')"], capture_output=True)
    if chk.returncode != 0:
        subprocess.run(
            [
                "bash",
                "-lc",
                r"""
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 gcc g++ make git curl netcat-openbsd >/dev/null
rm -rf /workspace/sglang
git clone --depth 1 https://github.com/sgl-project/sglang-torchtpu.git /workspace/sglang
python3 - <<'PY'
import importlib.metadata as md, tomllib
from pathlib import Path
keep = ["torch", "torchvision", "torch-tpu", "libtpu", "jax", "jaxlib", "tpu-sync-torch", "transformers"]
Path("/tmp/tpu_abi_constraints.txt").write_text("\n".join(f"{p}=={md.version(p)}" for p in keep) + "\n")
txt = Path("/workspace/sglang/python/pyproject_tpu.toml").read_text()
for m in ("[[tool.uv.index]]", "[tool.uv.index]", "[tool.uv.sources]"):
    txt = txt.split(m)[0]
drop = ('"torch==', '"torchvision==', '"torch_tpu==', '"tpu-sync-torch==', '"libtpu==', '"jax==', '"jaxlib==', '"torchao==')
lines = [l.replace('"transformers==5.17.0"', '"transformers>=5.17.0"') for l in txt.splitlines() if not any(l.strip().startswith(d) for d in drop)]
rendered = "\n".join(lines) + "\n"
tomllib.loads(rendered)
Path("/workspace/sglang/python/pyproject.toml").write_text(rendered)
PY
uv pip install --system -c /tmp/tpu_abi_constraints.txt "/workspace/sglang/python[diffusion_tpu,all_tpu,ray,runai]"
uv pip install --system --no-deps /workspace/sglang
""",
            ],
            check=True,
        )
    if need_vllm_omni:
        chk2 = subprocess.run(["python3", "-c", "import vllm_omni"], capture_output=True)
        if chk2.returncode != 0:
            subprocess.run(
                [
                    "bash",
                    "-lc",
                    r"""
set -euo pipefail
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
""",
                ],
                check=True,
            )


def stage_wan_repo(idx: int, bucket: str, repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    name = repo_id.split("/")[-1]
    local_dir = Path(f"/dev/shm/{name}")
    marker = local_dir / ".ready"
    if not marker.exists():
        fast_gcs_sync_dir(idx, bucket, f"models/{repo_id}/", local_dir)
        snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["*.mp4", "*.mov", "*.gif", "*.md"],
            local_dir=str(local_dir),
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=16,
        )
        marker.write_text(utc_now())
    return str(local_dir)


def generate_i2v_images(out_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, str] = {}
    for size in ("832x480", "1280x720", "720x1280"):
        w, h = (int(x) for x in size.split("x"))
        p = out_dir / f"wan_ref_{size}.png"
        img = Image.new("RGB", (w, h), (30, 52, 82))
        d = ImageDraw.Draw(img)
        d.ellipse([w // 3, h // 4, w // 3 + 120, h // 4 + 120], fill=(248, 206, 118))
        img.save(p, format="PNG")
        res[size] = str(p)
    return res


def kill_servers() -> None:
    subprocess.run(
        ["bash", "-lc", "pkill -9 -f 'sglang serve' || true; pkill -9 -f 'vllm serve' || true; rm -f /tmp/libtpu_lockfile || true"],
        check=False,
    )
    time.sleep(2)


def start_wan_server(
    engine: str,
    model_path: str,
    port: int,
    ulysses_degree: int = 8,
    cfg_parallel: bool = False,
    quantization: str | None = None,
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
    if engine == "sglang":
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
            "8",
            "--ulysses-degree",
            str(ulysses_degree),
            "--attention-backend",
            "tpu_splash_attn",
            "--save-output",
            "--output-path",
            "/dev/shm/wan_outputs",
        ]
        if cfg_parallel:
            cmd.append("--enable-cfg-parallel")
        if quantization:
            cmd.extend(["--quantization", quantization])
    else:
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
    lf = open(f"/tmp/{engine}_wan_{port}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"{engine} exited ({proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, round(time.time() - t0, 2)
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    raise TimeoutError(f"Timed out waiting for {engine} Wan 2.2 server")


def call_wan_video(engine: str, endpoint: str, payload: dict[str, Any], timeout_s: float = 1800.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    if engine == "sglang":
        req = urllib.request.Request(
            f"{endpoint}/v1/videos",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                job = json.loads(resp.read().decode("utf-8"))
            vid_id = job["id"]
            while time.perf_counter() - t0 < timeout_s:
                time.sleep(0.5)
                with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
                    st = json.loads(r.read().decode("utf-8"))
                if st.get("status") == "completed":
                    e2e = round(time.perf_counter() - t0, 3)
                    return {"ok": True, "e2e_s": e2e, "inference_s": st.get("inference_time_s") or e2e}
                if st.get("status") == "failed":
                    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))}
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}
        except Exception as exc:
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)}
    else:
        w, h = (int(x) for x in payload["size"].split("x"))
        body = {
            "prompt": payload["prompt"],
            "width": w,
            "height": h,
            "num_frames": payload["num_frames"],
            "fps": 16,
            "num_inference_steps": payload["num_inference_steps"],
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
    ap.add_argument("--run-id", default="v7x_flex_wan22_serve_p3")
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    gcs_prefix = f"video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    subprocess.run(["bash", "-lc", "mount -o remount,size=850G /dev/shm || true"], check=False)
    kill_servers()
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{idx}.json", json.dumps({"idx": idx, "ip": ip, "hostname": hostname}).encode("utf-8"))

    nodes: dict[int, dict[str, Any]] = {}
    t_w = time.time()
    while len(nodes) < 4 and time.time() - t_w < 600:
        for i in range(4):
            raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
            if raw:
                nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < 4:
            time.sleep(3)

    setup_single_host_metadata_hook(args.gcs_bucket)
    ensure_engines(idx, need_vllm_omni=(idx == 3))
    refs = generate_i2v_images(Path("/dev/shm/wan_inputs"))

    # Stage needed Wan 2.2 repo(s); all 4 pods stage I2V-A14B for Phase 3B DP=4 cluster
    i2v_dir = stage_wan_repo(idx, args.gcs_bucket, "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    t2v_dir = stage_wan_repo(idx, args.gcs_bucket, "Wan-AI/Wan2.2-T2V-A14B-Diffusers") if idx in (2, 3) else i2v_dir

    workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 4, "n": 3},
        {"label": "480p (832x480) 5s 81f (20-step)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
        {"label": "480p (832x480) 5s 81f (40-step full)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 40, "n": 2},
        {"label": "720p (1280x720) 5s 81f (4-step fast)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 4, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
        {"label": "720p (1280x720) 5s 81f (40-step full)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 40, "n": 2},
        {"label": "720p (1280x720) 10s 161f (20-step)", "size": "1280x720", "nf": 161, "dur": 10.0, "steps": 20, "n": 2},
        {"label": "720p Portrait (720x1280) 5s 81f (20-step)", "size": "720x1280", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
    ]

    pod_configs = {
        0: [("wan22_i2v_sglang_bf16_u8", "sglang", i2v_dir, True, "BF16", 8, False, None)],
        1: [("wan22_i2v_sglang_fp8_cfg2_u4", "sglang", i2v_dir, True, "FP8", 4, True, "fp8")],
        2: [
            ("wan22_t2v_sglang_bf16_u8", "sglang", t2v_dir, False, "BF16", 8, False, None),
            ("wan22_t2v_sglang_fp8_u8", "sglang", t2v_dir, False, "FP8", 8, False, "fp8"),
        ],
        3: [("wan22_t2v_vllm_omni_bf16_u8", "vllm_omni", t2v_dir, False, "BF16", 8, False, None)],
    }

    port = 30310 + idx
    pod_rows: list[dict[str, Any]] = []
    for cfg_name, eng, m_dir, is_i2v, prec, uly, cfg_p, quant in pod_configs[idx]:
        try:
            proc, boot_s = start_wan_server(eng, m_dir, port, uly, cfg_p, quant)
            try:
                ep = f"http://127.0.0.1:{port}"
                for w_i, wl in enumerate(workloads):
                    p_warm: dict[str, Any] = {
                        "prompt": PROMPTS[0],
                        "size": wl["size"],
                        "num_frames": wl["nf"],
                        "fps": 16,
                        "num_inference_steps": wl["steps"],
                        "generator_device": "cpu",
                        "seed": 42 + w_i,
                    }
                    if is_i2v:
                        p_warm["input_reference"] = refs[wl["size"]]
                    warm = call_wan_video(eng, ep, p_warm)
                    outs = []
                    t0 = time.perf_counter()
                    for r_i in range(wl["n"]):
                        pr = dict(p_warm, prompt=PROMPTS[r_i % 4], seed=2000 + w_i * 10 + r_i)
                        outs.append(call_wan_video(eng, ep, pr))
                    ws = time.perf_counter() - t0
                    oks = [o for o in outs if o.get("ok")]
                    med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
                    pod_rows.append(
                        {
                            "config": cfg_name,
                            "engine": eng,
                            "task": "i2v" if is_i2v else "t2v",
                            "precision": prec,
                            "chips": 4,
                            "workload": wl["label"],
                            "size": wl["size"],
                            "num_frames": wl["nf"],
                            "steps": wl["steps"],
                            "cold_compile_s": warm.get("e2e_s"),
                            "warm_median_e2e_s": med_s,
                            "ms_per_step": round((med_s * 1000.0) / wl["steps"], 1) if med_s else None,
                            "rpm": round((len(oks) / ws) * 60.0, 3) if oks else 0.0,
                            "video_sec_per_wall_sec": round((len(oks) * wl["dur"]) / ws, 3) if oks else 0.0,
                            "ok": f"{len(oks)}/{wl['n']}",
                        }
                    )
                    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))
            finally:
                proc.terminate()
                kill_servers()
        except Exception as exc:
            log(idx, f"[{cfg_name}] failed: {exc}")

    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase3a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    t_s = time.time()
    while time.time() - t_s < 5400:
        if sum(1 for i in range(4) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase3a_pod_{i}.txt") is not None) == 4:
            break
        time.sleep(10)

    # Phase 3B: Full 16-Chip Slice DP=4 Cluster on Wan2.2-I2V-A14B
    dp_port = 30410 + idx
    proc_dp, _ = start_wan_server("sglang", i2v_dir, dp_port, 8, False, None)
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
            for lbl, sz, nf, dur, steps, conc, n_req in [
                ("720p 5s 81f I2V (4-step fast)", "1280x720", 81, 5.0, 4, 4, 12),
                ("720p 5s 81f I2V (20-step)", "1280x720", 81, 5.0, 20, 4, 8),
                ("720p 5s 81f I2V (20-step)", "1280x720", 81, 5.0, 20, 8, 12),
            ]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
                    list(p.map(lambda i: call_wan_video("sglang", eps[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "generator_device": "cpu", "seed": 42}), range(4)))
                t_c = time.perf_counter()
                reqs = [(eps[r % 4], {"prompt": PROMPTS[r % 4], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "generator_device": "cpu", "seed": 3000 + r}) for r in range(n_req)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_wan_video("sglang", it[0], it[1]), reqs))
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
