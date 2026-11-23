from dataclasses import replace
import pytest
import torch
from .test_without0 import FakeModel
from RQ2.decoding import generate_fast
from RQ2.types import DecodeConfig
from RQ2.upstream import load_upstream

@pytest.fixture(scope="module")
def upstream():
    return load_upstream('/root/Fast-dLLM')

@pytest.mark.parametrize('temperature', [0.0, 0.5])
@pytest.mark.parametrize('accel,threshold', [('A',0.9),('B',0.7),('C',0.5)])
def test_threshold_matches_original(upstream, temperature, accel, threshold):
    model = FakeModel()
    prompt = torch.tensor([[3, 5, 7, 9]])
    cfg = DecodeConfig(steps=32, gen_length=32, block_length=8, mask_id=31,
                       accel=accel, threshold=threshold, temperature=temperature)
    torch.manual_seed(0)
    expected, nfe = upstream.generate(model, prompt, steps=32, gen_length=32, block_length=8,
                                      temperature=temperature, mask_id=31, threshold=threshold)
    torch.manual_seed(0)
    actual = generate_fast(model, prompt, cfg, upstream)
    assert torch.equal(expected, actual.token_ids)
    assert nfe == actual.gen_nfe

@pytest.mark.parametrize('prompt_masks', [0, 45, 200])
def test_low_confidence_fallback_has_finite_completion_bound(upstream, prompt_masks):
    model = FakeModel()
    prompt = torch.tensor([[3] + [31] * prompt_masks + [5]])
    cfg = DecodeConfig(mask_id=31, accel='A', threshold=0.9)
    actual = generate_fast(model, prompt, cfg, upstream)
    assert actual.gen_nfe == prompt_masks + 128
    assert actual.n_masks == 0
    assert actual.blocks[0]['step_cap'] == prompt_masks + 32
    assert [b['generation_masks_remaining'] for b in actual.blocks] == [96, 64, 32, 0]

class ConfidentModel(FakeModel):
    def forward(self, x):
        result = super().forward(x)
        result.logits[:] = -1000
        result.logits[:, :, 3] = 1000
        return result

def test_all_confident_positions_finish_in_one_call_per_block(upstream):
    model = ConfidentModel()
    prompt = torch.tensor([[5] + [31] * 100 + [7]])
    cfg = DecodeConfig(mask_id=31, accel='A', threshold=0.9)
    result = generate_fast(model, prompt, cfg, upstream)
    assert result.gen_nfe == 4
    assert result.n_masks == 0
    assert result.token_ids[0, 0] == 5 and result.token_ids[0, 101] == 7
