import importlib.metadata as md
import tomllib
from pathlib import Path

keep_pkgs = [
    "torch",
    "torchvision",
    "torchaudio",
    "torch-tpu",
    "libtpu",
    "jax",
    "jaxlib",
    "tpu-sync-torch",
    "transformers",
    "vllm",
]
constraints = []
for p in keep_pkgs:
    try:
        constraints.append(f"{p}=={md.version(p)}")
    except Exception:
        pass
Path("/tmp/vllm_omni_constraints.txt").write_text("\n".join(constraints) + "\n")
print("Constraints:\n" + "\n".join(constraints))

# Verify pyproject.toml remains untouched and valid; relax exact pins in requirements/common.txt if needed
tomllib.loads(Path("/workspace/vllm-omni/pyproject.toml").read_text())
common_req = Path("/workspace/vllm-omni/requirements/common.txt")
if common_req.exists():
    out = []
    for line in common_req.read_text().splitlines():
        s = line.strip()
        if s.startswith("kernels=="):
            # Optional CUDA hub kernels (NATTEN / FA3); skip or relax for CPU/TPU
            continue
        if s.startswith("cosmos-guardrail"):
            continue
        out.append(line)
    common_req.write_text("\n".join(out) + "\n")

print("Prepared /workspace/vllm-omni for TPU install (VLLM_OMNI_TARGET_DEVICE=cpu)")
