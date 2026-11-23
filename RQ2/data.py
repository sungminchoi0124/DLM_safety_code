"""Pinned DIJA input and stable manifests. No regeneration of attack prompts."""
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
PROMPT = REPO / "DIJA/third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json"
PROMPT_SHA = "534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95"
DIJA_COMMIT = "56ba341b3377423e6da212013aec6e825710a64c"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

def load_items(ids_file):
    if digest(PROMPT) != PROMPT_SHA:
        raise ValueError("DIJA input SHA-256 mismatch")
    items = json.loads(PROMPT.read_text())
    rows = [line.split() for line in Path(ids_file).read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(rows) != 100 or rows != [[f"dija-{i:03d}", str(i)] for i in range(100)]:
        raise ValueError("Expected the fixed 100-item source-order manifest")
    if len(items) != 100:
        raise ValueError("Expected 100 source items")
    return [(name, int(key), items[int(key)]) for name, key in rows]

def prepare(item, tokenizer):
    vanilla = item.get("vanilla prompt") or item.get("goal") or item.get("Behavior")
    refined = item.get("refined prompt") or item.get("refined_goal") or item.get("Refined_behavior")
    if not vanilla or not refined:
        raise ValueError("Missing vanilla or refined prompt")
    def build(text):
        text = re.sub(r"<mask:(\d+)>", lambda m: "<|mdm_mask|>" * int(m[1]), text)
        if "<|mdm_mask|>" not in text:
            text += "<startoftext>" + "<|mdm_mask|>" * 36 + "<endoftext>"
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                             add_generation_prompt=True)
        return tokenizer(chat, return_tensors="pt")["input_ids"]
    source, plain = build(refined), build(vanilla)
    # Preserve DiffuGuard generate_response's historical sum-of-equal-positions rule exactly.
    matching = min(sum(a == b for a, b in zip(plain[0].tolist(), source[0].tolist())), plain.shape[1])
    return source, matching, vanilla, refined
