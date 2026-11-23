"""Use the pinned Fast-dLLM implementation; never modify the external checkout."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

FAST_COMMIT = "a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"

def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

def load_upstream(root):
    root = Path(root).resolve()
    if git(root, "rev-parse", "HEAD") != FAST_COMMIT or git(root, "status", "--porcelain"):
        raise ValueError("Fast-dLLM must be clean at pinned commit " + FAST_COMMIT)
    base = root / "v1" / "llada"
    sys.path.insert(0, str(base))
    from model import modeling_llada
    if Path(modeling_llada.__file__).resolve() != base / "model" / "modeling_llada.py":
        raise ValueError("A different model package was already imported")
    spec = importlib.util.spec_from_file_location("rq2_fast_dllm_original", base / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model(model_dir, fast_root):
    import torch
    from transformers import AutoTokenizer
    upstream = load_upstream(fast_root)
    model_dir = Path(model_dir).resolve()
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if not metadata.exists() or metadata.read_text().splitlines()[0] != MODEL_REVISION:
        raise ValueError("Local model revision metadata missing or incorrect")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    expected = set(index["weight_map"].values()) | {"config.json", "tokenizer.json", "tokenizer_config.json"}
    for name in expected:
        local = model_dir / name
        stamp = model_dir / ".cache/huggingface/download" / (name + ".metadata")
        if not local.is_file() or not stamp.is_file() or stamp.read_text().splitlines()[0] != MODEL_REVISION:
            raise ValueError("Missing model file or revision mismatch: " + name)
    from model.configuration_llada import LLaDAConfig
    config = LLaDAConfig.from_pretrained(model_dir, local_files_only=True)
    config.use_cache = False
    config.flash_attention = True  # Same as Fast-dLLM eval_llada.py; falls back to PyTorch SDPA.
    model = upstream.LLaDAModelLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    return model, tokenizer, upstream
