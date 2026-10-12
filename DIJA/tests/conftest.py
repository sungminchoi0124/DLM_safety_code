"""Shared fixtures for CPU tests. Fixtures contain only benign synthetic text."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]  # DLM_safety_code/DIJA
DIJA_ROOT = PROJECT_DIR / "third_party" / "DIJA"
sys.path.insert(0, str(PROJECT_DIR))

MASK_ID = 126336
MASK_TOKEN = "<|mdm_mask|>"


class FakeModel(torch.nn.Module):
    """Mimics the interface generate_llada uses: model(x, attention_mask=...).logits and model.device.

    Token ids are folded into a small vocab so the real mask id (126336) can be fed in,
    and the mask id itself can never be predicted (vocab < mask id).
    """

    def __init__(self, vocab=64, dim=8):
        super().__init__()
        g = torch.Generator().manual_seed(1234)
        self.vocab = vocab
        self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=g))
        self.pos = torch.nn.Parameter(torch.randn(512, dim, generator=g))
        self.out = torch.nn.Parameter(torch.randn(dim, vocab, generator=g))

    @property
    def device(self):
        return self.emb.device

    def forward(self, x, attention_mask=None):
        h = self.emb[x % self.vocab] + self.pos[: x.shape[1]]
        return SimpleNamespace(logits=h @ self.out)


class FakeTokenizer:
    """Word-level tokenizer with the methods the original DIJA code calls."""

    def __init__(self, vocab=64):
        self.vocab = vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<user> {m['content']} </user>" for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def _ids(self, text):
        ids = []
        for piece in re.split(f"({re.escape(MASK_TOKEN)})", text):
            if piece == MASK_TOKEN:
                ids.append(MASK_ID)
            else:
                ids.extend(sum(map(ord, w)) % (self.vocab - 2) + 2 for w in piece.split())
        return ids

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([self._ids(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_tokens_to_ids(self, tok):
        return MASK_ID if tok == MASK_TOKEN else 0

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(f"w{int(t)}" for t in row if int(t) != MASK_ID) for row in ids]


BENIGN_ITEMS = [
    {"behavior": "Bread", "target": "Sure, here is a bread recipe", "category": "Cooking",
     "goal": "Write a recipe for bread",
     "refined_goal": "Write a recipe for bread. Step 1: <mask:6> Step 2: <mask:4>"},
    {"behavior": "Poem", "target": "Sure, here is a poem", "category": "Writing",
     "goal": "Write a short poem about the sea",
     "refined_goal": "Write a short poem about the sea. Line 1: <mask:5>"},
]


@pytest.fixture
def fake_model():
    return FakeModel().eval()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
