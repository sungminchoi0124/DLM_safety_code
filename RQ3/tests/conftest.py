"""Tiny LLaDA look-alike (same module layout: model.model.transformer.{wte, blocks, ln_f, ff_out}) for CPU tests."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

RQ3 = Path(__file__).resolve().parents[1]
REPO = RQ3.parent
sys.path.insert(0, str(RQ3))
sys.path.insert(0, str(REPO / "DiffuGuard" / "third_party" / "DiffuGuard"))

V, D, MASK = 64, 32, 63


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.mix = nn.Linear(D, D)

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False):
        x = x + torch.tanh(self.lin(x)) + 0.3 * self.mix(x.mean(dim=1, keepdim=True))  # bidirectional mixing
        return x, None


class Core(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(V, D), wpe=nn.Embedding(512, D), blocks=nn.ModuleList([Block() for _ in range(n_blocks)]),
            ln_f=nn.LayerNorm(D), ff_out=nn.Linear(D, V)))


class FakeLLaDA(nn.Module):
    def __init__(self, n_blocks=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = Core(n_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, **kw):
        tr = self.model.transformer
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = tr.wte(input_ids) + tr.wpe(pos)[None]
        hs = [x]
        for i, b in enumerate(tr.blocks):
            x, _ = b(x)
            if i < len(tr.blocks) - 1:
                hs.append(x)
        x = tr.ln_f(x)
        hs.append(x)
        logits = 3.0 * tr.ff_out(x)
        logits[..., MASK] = -1e4
        return SimpleNamespace(logits=logits, hidden_states=tuple(hs) if output_hidden_states else None)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(ord(c) % (V - 2)) + 1 for c in text][:8] or [1]
        if return_tensors == "pt":
            t = torch.tensor([ids])
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}
        return SimpleNamespace(input_ids=ids)

    def batch_decode(self, x, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row) for row in x]


@pytest.fixture
def model():
    return FakeLLaDA().eval()


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def prompt():
    # DIJA-like: prompt text with masks inside
    g = torch.Generator().manual_seed(1)
    p = torch.randint(1, V - 2, (1, 20), generator=g)
    p[0, [5, 6, 7, 12, 13]] = MASK
    return p
