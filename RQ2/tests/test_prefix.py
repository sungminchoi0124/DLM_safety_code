from types import SimpleNamespace
import pytest
import torch
from .test_without0 import FakeModel
from RQ2.decoding import generate_fast
from RQ2.reference import prefix_after_first_reference
from RQ2.types import DecodeConfig
from RQ2.upstream import load_upstream

class CacheModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.calls = []
    def forward(self, x, past_key_values=None, use_cache=False):
        prefix = past_key_values[0][0][:, 0, :, 0].long() if past_key_values is not None else x[:, :0]
        full = torch.cat([prefix, x], dim=1)
        self.calls.append({'use_cache':use_cache, 'prefix':prefix.shape[1], 'input':x.shape[1]})
        logits = super().forward(full).logits[:, prefix.shape[1]:]
        kv = full[:, None, :, None].float()
        return SimpleNamespace(logits=logits, past_key_values=[(kv,kv)] if use_cache else None)

@pytest.fixture(scope='module')
def upstream():
    return load_upstream('/root/Fast-dLLM')

@pytest.mark.parametrize('temperature', [0.0,0.5])
def test_prefix_matches_staged_original_tokens_nfe_and_actual_calls(upstream,temperature):
    model=CacheModel()
    prompt=torch.tensor([[3,5,7]])
    cfg=DecodeConfig(steps=32,gen_length=32,block_length=8,mask_id=31,
                     accel='D',threshold=0.9,cache='prefix',temperature=temperature)
    torch.manual_seed(0)
    expected,nfe=prefix_after_first_reference(model,prompt,cfg,upstream)
    model.calls=[]
    torch.manual_seed(0)
    result=generate_fast(model,prompt,cfg,upstream)
    assert torch.equal(result.token_ids,expected)
    assert result.gen_nfe==nfe==len(model.calls)
    assert sum(bool(c['prefix']) for c in model.calls)==sum(b['cache_reuse_nfe'] for b in result.blocks)
    assert all(not c['use_cache'] for c in model.calls[:result.blocks[0]['gen_nfe']])
    assert result.blocks[0]['cache_build_nfe']==result.blocks[0]['cache_reuse_nfe']==0
    assert [b['cache_prefix_length'] for b in result.blocks]==[0,11,19,27]
    for c in model.calls:
        if c['prefix']: assert c['input']+c['prefix']==35

@pytest.mark.parametrize('prompt_masks',[45,200])
def test_dija_first_block_matches_A_and_finishes_before_cache(upstream,prompt_masks):
    model=CacheModel()
    prompt=torch.tensor([[3]+[31]*prompt_masks+[5]])
    cfg=DecodeConfig(mask_id=31,accel='D',threshold=0.9,cache='prefix')
    torch.manual_seed(0)
    result=generate_fast(model,prompt,cfg,upstream)
    calls=list(model.calls)
    torch.manual_seed(0)
    baseline=generate_fast(model,prompt,DecodeConfig(mask_id=31,accel='A',threshold=0.9),upstream)
    first_end=prompt.shape[1]+32
    assert torch.equal(result.token_ids[:,:first_end],baseline.token_ids[:,:first_end])
    assert result.n_masks==0
    assert [b['generation_masks_remaining'] for b in result.blocks]==[96,64,32,0]
    assert result.blocks[0]['step_cap']==prompt_masks+32
    assert all(not c['use_cache'] for c in calls[:result.blocks[0]['gen_nfe']])
    assert sum(b['cache_build_nfe'] for b in result.blocks)==3
    assert sum(b['cache_reuse_nfe'] for b in result.blocks)>0

class MissingCache(CacheModel):
    def forward(self,*args,**kwargs):
        out=super().forward(*args,**kwargs)
        out.past_key_values=None
        return out

def test_no_silent_cache_fallback(upstream):
    with pytest.raises(RuntimeError,match='did not return'):
        generate_fast(MissingCache(),torch.tensor([[3,5]]),
                      DecodeConfig(steps=16,gen_length=16,block_length=8,mask_id=31,
                                   accel='D',threshold=0.9,cache='prefix'),upstream)
