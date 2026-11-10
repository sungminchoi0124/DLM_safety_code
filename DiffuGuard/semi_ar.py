"""Block-by-block (semi-autoregressive) variant of the original DiffuGuard generate().

Why this exists
---------------
The original utility/generate_function_llada.generate() plans gen_length // block_length blocks, but a
"safety guard" at the top of the block loop widens every block to span all remaining masks:

    global_mask_pos = (x == mask_id).nonzero(as_tuple=False)
    if global_mask_pos.numel() > 0:
        block_start = min(block_start, first); block_end = max(block_end, last)

With one block (block_length == gen_length, every jailbreak run of this repo) that changes nothing. With
block_length < gen_length (GSM8K: 256 / 32, as in the Fast-dLLM reproduction) the first block becomes the whole
tail: all tokens are decoded in the first block's steps (steps / num_blocks) and the remaining blocks only
burn forward passes. That is neither LLaDA's semi-autoregressive decoding nor the Fast-dLLM baseline.

What this does
--------------
Loads the original source text, switches that one guard off, and executes the result as a separate module.
Everything else (remasking rules, SAR schedule, audit, repair) is the original code, byte for byte. The
submodule file itself is not modified. If the guard line is not found exactly once (the original changed),
loading fails instead of silently running something else.

Only valid when the prompt contains no mask tokens and fill_all_masks is False: the guard is what fills
in-prompt masks (DIJA) in the original, so those runs must keep using the original function.
tests/test_semi_ar.py checks: (1) equal to the original when there is one block, (2) equal to the Fast-dLLM
baseline generate() block by block when the defense is off.
"""

import types
from pathlib import Path

GUARD = "        if global_mask_pos.numel() > 0:\n"
PATCHED = "        if False and global_mask_pos.numel() > 0:  # semi_ar.py: keep the planned block\n"
SOURCE = Path("utility") / "generate_function_llada.py"


def load(diffuguard_root):
    """Return a module holding the guard-free copy of the original generate()/generate_llada()."""
    src = Path(diffuguard_root).resolve() / SOURCE
    text = src.read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.count(GUARD) != 1:
        raise SystemExit(f"semi_ar: expected exactly one block-guard line in {src}, found {text.count(GUARD)}; "
                         "the original code changed, re-check the patch.")
    mod = types.ModuleType("diffuguard_generate_semi_ar")
    mod.__file__ = f"{src} [semi_ar patch]"
    exec(compile(text.replace(GUARD, PATCHED), mod.__file__, "exec"), mod.__dict__)
    mod.SEMI_AR_PATCH = {"source": str(SOURCE).replace("\\", "/"), "removed_guard": GUARD.strip(),
                         "replaced_with": PATCHED.strip()}
    return mod
