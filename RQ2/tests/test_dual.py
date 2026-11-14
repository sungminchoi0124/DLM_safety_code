from types import SimpleNamespace
import pytest
import torch
from .test_without0 import FakeModel
from RQ2.decoding import generate_fast
from RQ2.dual_reference import dual_after_first_reference
from RQ2.types import DecodeConfig
from RQ2.upstream import load_upstream

class DualModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.calls=[]
    def forward(self,x,past_key_values=None,use_cache=False,replace_position=None):
        if past_key_values is None:
            full=x
            positions=None
        else:
            assert use_cache and replace_position is not None
            positions=replace_position[0].nonzero(as_tuple=True)[0]
            assert positions.numel()==x.shape[1]
            full=past_key_values[0][0][:,0,:,0].long().clone()
            before=full.clone()
            full[:,positions]=x
            outside=~replace_position
            assert torch.equal(full[outside],before[outside])
            for tensor in past_key_values[0]:
                tensor[:,0,positions,0]=x.float()
        logits=super().forward(full).logits
        if positions is not None: logits=logits[:,positions]
        self.calls.append({'input':x.shape[1],'total':full.shape[1],
                           'replace':0 if positions is None else len(positions),
                           'use_cache':use_cache})
        kv=full[:,None,:,None].float()
        return SimpleNamespace(logits=logits,past_key_values=[(kv,kv.clone())] if use_cache else None)

@pytest.fixture(scope='module')
def upstream():
    return load_upstream('/root/Fast-dLLM')

@pytest.mark.parametrize('temperature',[0.0,0.5])
def test_dual_matches_staged_original_and_replaces_only_active_positions(upstream,temperature):
    model=DualModel()
    prompt=torch.tensor([[3,5,7]])
    cfg=DecodeConfig(steps=32,gen_length=32,block_length=8,temperature=temperature,
                     mask_id=31,accel='E',threshold=0.9,cache='dual')
    torch.manual_seed(0)
    expected,nfe=dual_after_first_reference(model,prompt,cfg,upstream)
    model.calls=[]
    torch.manual_seed(0)
    result=generate_fast(model,prompt,cfg,upstream)
    assert torch.equal(result.token_ids,expected)
    assert result.gen_nfe==nfe==len(model.calls)
    reused=[c for c in model.calls if c['replace']]
    assert reused and all(c['input']==c['replace']==8 and c['total']==35 for c in reused)
    assert len(reused)==sum(b['dual_replace_nfe'] for b in result.blocks)
    assert result.blocks[0]['cache_enabled'] is False
    assert all(not c['use_cache'] for c in model.calls[:result.blocks[0]['gen_nfe']])

@pytest.mark.parametrize('prompt_masks',[45,200])
def test_dual_dija_first_block_is_A_and_all_masks_complete(upstream,prompt_masks):
    model=DualModel()
    prompt=torch.tensor([[3]+[31]*prompt_masks+[5]])
    cfg=DecodeConfig(mask_id=31,accel='E',threshold=0.9,cache='dual')
    torch.manual_seed(0)
    result=generate_fast(model,prompt,cfg,upstream)
    torch.manual_seed(0)
    ref=generate_fast(model,prompt,DecodeConfig(mask_id=31,accel='A',threshold=0.9),upstream)
    end=prompt.shape[1]+32
    assert torch.equal(result.token_ids[:,:end],ref.token_ids[:,:end])
    assert result.n_masks==0
    assert [b['generation_masks_remaining'] for b in result.blocks]==[96,64,32,0]
    assert [b['cache_suffix_length'] for b in result.blocks]==[0,64,32,0]
    assert sum(b['cache_reuse_nfe'] for b in result.blocks)>0

class MissingDualCache(DualModel):
    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        result.past_key_values=None
        return result

def test_dual_cache_missing_causes_failure(upstream):
    with pytest.raises(RuntimeError,match='did not return Dual'):
        generate_fast(MissingDualCache(),torch.tensor([[3,5]]),
                      DecodeConfig(steps=16,gen_length=16,block_length=8,mask_id=31,
                                   accel='E',threshold=0.9,cache='dual'),upstream)
