import concurrent.futures
import os
import time
import requests
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url

BUCKET = "ank-usc1-bucket-314837540096"
PREFIX = "models"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

TARGETS = [
    # 1. Priority #1A: MiniMax-H3 FL2VA partition (native SGLang t2va + fl2va i2v 768p 15s!)
    ("MiniMaxAI/MiniMax-H3", ("FL2VA/", "model_index.json", "modular_model_index.json")),
    # 2. Priority #1B: FastH3 4-step VSA distilled MiniMax-H3 + overlay
    ("kevin-mi/FastH3-4step-Preview-overlay", None),
    ("FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree", None),
    # 3. Priority #2: Official Lightricks/LTX-2.5-Diffusers
    ("Lightricks/LTX-2.5-Diffusers", None),
    # 4. Priority #3B: Wan2.2-I2V-A14B-Diffusers (T2V already staged!)
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", None),
]

DROP_EXTS = (".mp4", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".mov")


def stage_repo(repo_id: str, allow_prefixes):
    api = HfApi(token=HF_TOKEN)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    siblings = []
    for s in info.siblings:
        fn = s.rfilename
        if fn.lower().endswith(DROP_EXTS) or "/docs/" in fn or fn.startswith("docs/"):
            continue
        if allow_prefixes is not None:
            if not any(fn == p or fn.startswith(p) for p in allow_prefixes):
                continue
        siblings.append(s)

    total_bytes = sum((s.size or 0) for s in siblings)
    print(
        f"\n=== Staging {repo_id}: {len(siblings)} files ({total_bytes / 1e9:.2f} GB) -> gs://{BUCKET}/{PREFIX}/{repo_id}/ ===",
        flush=True,
    )

    def xfer(s):
        client = storage.Client()
        blob = client.bucket(BUCKET).blob(f"{PREFIX}/{repo_id}/{s.rfilename}")
        if blob.exists():
            blob.reload()
            if (s.size is None and blob.size is not None) or (
                (s.size or 0) > 0 and blob.size == s.size
            ):
                return (s.rfilename, "SKIP", blob.size or 0)
        url = hf_hub_url(repo_id=repo_id, filename=s.rfilename)
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        t0 = time.time()
        for attempt in range(1, 5):
            written = 0
            try:
                with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
                    resp.raise_for_status()
                    with blob.open("wb", chunk_size=64 * 1024 * 1024) as f:
                        for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                dt = max(time.time() - t0, 1e-3)
                if written >= 50 * 1024 * 1024:
                    print(
                        f"  [OK] {repo_id}/{s.rfilename}: {written / 1e9:.2f} GB in {dt:.1f}s ({(written / 1048576) / dt:.0f} MiB/s)",
                        flush=True,
                    )
                return (s.rfilename, "OK", written)
            except Exception as exc:
                print(
                    f"  [RETRY {attempt}] {repo_id}/{s.rfilename}: {exc}",
                    flush=True,
                )
                if attempt == 4:
                    raise
                time.sleep(3 * attempt)

    t_repo = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(xfer, s) for s in siblings]
        done = 0
        bytes_done = 0
        for fut in concurrent.futures.as_completed(futs):
            _, status, sz = fut.result()
            done += 1
            bytes_done += sz
            if done % 10 == 0 or done == len(siblings):
                print(
                    f"  [{repo_id}] {done}/{len(siblings)} files ({bytes_done / 1e9:.2f}/{total_bytes / 1e9:.2f} GB) in {time.time() - t_repo:.1f}s",
                    flush=True,
                )

    storage.Client().bucket(BUCKET).blob(
        f"{PREFIX}/{repo_id}/_VIDEO_SERVE_STAGE_DONE.txt"
    ).upload_from_string("ok\n")
    print(f"=== COMPLETE {repo_id} in {time.time() - t_repo:.1f}s ===", flush=True)


if __name__ == "__main__":
    for repo, prefixes in TARGETS:
        stage_repo(repo, prefixes)
