import importlib.metadata as md
from pathlib import Path

keep_pkgs = [
    "torch",
    "torchvision",
    "torch-tpu",
    "libtpu",
    "jax",
    "jaxlib",
    "tpu-sync-torch",
    "transformers",
]
constraints = []
for p in keep_pkgs:
    try:
        v = md.version(p)
        constraints.append(f"{p}=={v}")
    except Exception:
        pass
Path("/tmp/tpu_abi_constraints.txt").write_text("\n".join(constraints) + "\n")
print("Constraints:\n" + "\n".join(constraints))

import tomllib

toml_p = Path("/workspace/sglang/python/pyproject_tpu.toml")
txt = toml_p.read_text()
for marker in ("[[tool.uv.index]]", "[tool.uv.index]", "[tool.uv.sources]"):
    txt = txt.split(marker)[0]

drop_prefixes = (
    '"torch==',
    '"torchvision==',
    '"torch_tpu==',
    '"tpu-sync-torch==',
    '"libtpu==',
    '"jax==',
    '"jaxlib==',
    '"torchao==',
)
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
replacement = """        import os
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
assert target in stxt, "Expected forward_varlen target in tpu_splash_attn.py"
splash_p.write_text(stxt.replace(target, replacement, 1))
print("Patched tpu_splash_attn.py for SGLANG_TPU_SPLASH_VARLEN=1 support")
