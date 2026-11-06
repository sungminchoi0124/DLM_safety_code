"""Shared fixtures for CPU tests. Fixtures contain only benign synthetic text."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]  # DLM_safety_code/DiffuGuard
DG_ROOT = PROJECT_DIR / "third_party" / "DiffuGuard"
sys.path.insert(0, str(PROJECT_DIR))

MASK_ID = 126336
MASK_TOKEN = "<|mdm_mask|>"


class FakeModel(torch.nn.Module):
    """Mimics what the original LLaDA code calls: model(x, attention_mask=..., output_hidden_states=...,
    return_dict=...).logits / .hidden_states and model.device.

    Token ids are folded into a small vocab so the real mask id (126336) can be fed in, and the mask id
    itself can never be predicted (vocab < mask id).
    """

    def __init__(self, vocab=64, dim=8, layers=3):
        super().__init__()
        g = torch.Generator().manual_seed(1234)
        self.vocab = vocab
        self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=g))
        self.pos = torch.nn.Parameter(torch.randn(1024, dim, generator=g))
        self.mix = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dim, dim, generator=g) / dim ** 0.5) for _ in range(layers)])
        self.out = torch.nn.Parameter(torch.randn(dim, vocab, generator=g))

    @property
    def device(self):
        return self.emb.device

    def forward(self, x, attention_mask=None, output_hidden_states=False, return_dict=True):
        h = self.emb[x % self.vocab] + self.pos[: x.shape[1]]
        hs = [h]
        for w in self.mix:
            # bidirectional "attention": every position sees the sequence mean
            h = torch.tanh(h @ w + h.mean(dim=1, keepdim=True))
            hs.append(h)
        return SimpleNamespace(logits=h @ self.out, hidden_states=tuple(hs) if output_hidden_states else None)


class _Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FakeTokenizer:
    """Word-level tokenizer with the methods the original DiffuGuard code calls."""

    def __init__(self, vocab=64):
        self.vocab = vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
        text = "".join(f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages)
        text += " assistant\n" if add_generation_prompt else ""
        if tokenize:
            return torch.tensor([self._ids(text)], dtype=torch.long)
        return text

    def _ids(self, text):
        ids = []
        for piece in re.split(f"({re.escape(MASK_TOKEN)})", text):
            if piece == MASK_TOKEN:
                ids.append(MASK_ID)
            else:
                ids.extend(sum(map(ord, w)) % (self.vocab - 2) + 2 for w in piece.split())
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._ids(text)
        if return_tensors == "pt":
            t = torch.tensor([ids], dtype=torch.long)
            return _Enc(input_ids=t, attention_mask=torch.ones_like(t))
        return _Enc(input_ids=ids, attention_mask=[1] * len(ids))

    def convert_tokens_to_ids(self, tok):
        return MASK_ID if tok == MASK_TOKEN else 0

    def convert_ids_to_tokens(self, i):
        return MASK_TOKEN if i == MASK_ID else f"w{i}"

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(f"w{int(t)}" for t in row if int(t) != MASK_ID) for row in ids]


BENIGN_ITEMS = [
    {"category": "Cooking", "goal": "Write a recipe for bread",
     "refined_goal": "Write a recipe for bread. Step 1: <mask:6> Step 2: <mask:4>"},
    {"category": "Writing", "goal": "Write a short poem about the sea",
     "refined_goal": "Write a short poem about the sea. Line 1: <mask:5>"},
]


@pytest.fixture
def fake_model():
    return FakeModel().eval()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
