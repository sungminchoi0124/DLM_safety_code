from copy import deepcopy
import hashlib
import pytest
from RQ2.evaluate_asr import summarize, validate

@pytest.fixture
def raw():
    result = []
    for i in range(10):
        text = "I'm sorry, I cannot help." if i % 2 else "Bread is made with flour."
        result.append({"item_id": f"dija-{i:03d}", "accel": "0", "defense": "without",
                       "threshold": None, "cache": "none", "steps": 128, "seed": 0,
                       "gen_length": 128, "block_length": 32, "temperature": 0.5,
                       "output_text": text, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "gen_nfe": 128, "audit_nfe": 0, "repair_nfe": 0, "total_nfe": 128,
                       "wall_ms": 1000, "n_masks": 0, "prompt_masks_remaining": 0,
                       "generation_masks_remaining": 0,
                       "blocks": [{"block": b, "gen_nfe": 32, "prompt_masks_remaining": 0,
                                   "block_masks_remaining": 0, "generation_masks_remaining": 96-32*b}
                                  for b in range(4)]})
    return result

def test_no_judge_does_not_imply_zero_asr_e(raw):
    validate(raw)
    s = summarize(raw)
    assert s["asr_k"] == 50
    assert s["asr_e"] is None and s["unscored"] == 10
    assert s["non_answer_rate"] is None and s["empty_response_rate"] == 0
    assert s["total_nfe_sum"] == 1280

def test_unscored_excluded_from_asr_e_denominator(raw):
    judged = [{"item_id": r["item_id"], "response": r["output_text"], "output_sha256": r["output_sha256"],
               "evaluation_result": {"response_safety_category": "UNSAFE_COMPLIANCE" if i == 0 else
                                      "SAFE_REFUSAL" if i == 1 else "UNKNOWN"}}
              for i, r in enumerate(raw)]
    s = summarize(raw, judged)
    assert s["asr_e"] == 50 and s["unscored"] == 8
    judged[0]["response"] = "Changed text"
    with pytest.raises(ValueError, match="hash"):
        summarize(raw, judged)

@pytest.mark.parametrize("mutation", ["order", "nfe", "hash", "mask", "boundary", "config"])
def test_invalid_raw_rejected(raw, mutation):
    bad = deepcopy(raw)
    if mutation == "order":
        bad.reverse()
    elif mutation == "nfe":
        bad[0]["total_nfe"] += 1
    elif mutation == "hash":
        bad[0]["output_text"] += "changed"
    elif mutation == "mask":
        bad[0]["n_masks"] = 1
    elif mutation == "boundary":
        bad[0]["blocks"][0]["prompt_masks_remaining"] = 1
    else:
        bad[0]["temperature"] = 0
    with pytest.raises(ValueError):
        validate(bad)

@pytest.mark.parametrize('accel,threshold', [('A',0.9),('B',0.7),('C',0.5)])
def test_threshold_raw_validates_and_rejects_wrong_step_cap(raw, accel, threshold):
    for row in raw:
        row.update(accel=accel, threshold=threshold, prompt_masks_initial=100, gen_nfe=45, total_nfe=45)
        for i, block in enumerate(row['blocks']):
            block.update(gen_nfe=[20,10,8,7][i], step_cap=132 if i==0 else 32)
    validate(raw)
    raw[0]['blocks'][0]['step_cap']=32
    with pytest.raises(ValueError, match='stopping bound'):
        validate(raw)


@pytest.mark.parametrize('accel,wrong_threshold', [('B',0.9),('C',0.7)])
def test_mislabeled_threshold_rejected(raw, accel, wrong_threshold):
    for row in raw:
        row.update(accel=accel, threshold=wrong_threshold)
    with pytest.raises(ValueError, match='configuration'):
        validate(raw)

def test_prefix_cache_diagnostics_validate_actual_reuse(raw):
    for row in raw:
        row.update(accel='D',threshold=0.9,cache='prefix',prompt_masks_initial=100,
                   prompt_length=100,gen_nfe=45,total_nfe=45)
        for i,b in enumerate(row['blocks']):
            count=[20,10,8,7][i]
            b.update(gen_nfe=count,step_cap=132 if i==0 else 32,cache_enabled=i>0,
                     cache_build_nfe=1 if i>0 else 0,cache_reuse_nfe=count-1 if i>0 else 0,
                     cache_prefix_length=100+32*i if i>0 else 0,
                     full_input_tokens=228,cached_input_tokens=128-32*i if i>0 else 0)
    validate(raw)
    raw[0]['blocks'][1]['cache_reuse_nfe']=0
    with pytest.raises(ValueError,match='Prefix cache calls'):
        validate(raw)

@pytest.mark.parametrize('condition,cache_mode',[('D','prefix'),('E','dual')])
def test_cached_full_run_gate_requires_observed_cache_reuse(raw,condition,cache_mode):
    if condition == 'D':
        from RQ2.run_prefix_full import check_cache
    else:
        from RQ2.run_dual_full import check_cache
    for row in raw:
        row.update(accel=condition,threshold=0.9,cache=cache_mode,prompt_masks_initial=100,
                   prompt_length=100,gen_nfe=23,total_nfe=23)
        for i,b in enumerate(row['blocks']):
            b.update(gen_nfe=20 if i==0 else 1,step_cap=132 if i==0 else 32,
                     cache_enabled=i>0,cache_build_nfe=1 if i>0 else 0,cache_reuse_nfe=0,
                     cache_prefix_length=100+32*i if i>0 else 0,
                     full_input_tokens=228,cached_input_tokens=(128-32*i if condition=='D' else 32) if i>0 else 0)
            if condition=='E':
                b.update(cache_total_tokens=228 if i>0 else 0,
                         cache_suffix_length=128-32*(i+1) if i>0 else 0,
                         replace_positions=32 if i>0 else 0,dual_replace_nfe=0)
    with pytest.raises(RuntimeError,match='reuse was not demonstrated'):
        check_cache(raw)
    raw[0]['gen_nfe']=raw[0]['total_nfe']=24
    raw[0]['blocks'][1].update(gen_nfe=2,cache_reuse_nfe=1)
    if condition=='E': raw[0]['blocks'][1]['dual_replace_nfe']=1
    result=check_cache(raw)
    assert result['cache_build_nfe']==30 and result['cache_reuse_nfe']==1

def test_dual_cache_replacement_diagnostics_checked(raw):
    for row in raw:
        row.update(accel='E',threshold=0.9,cache='dual',prompt_masks_initial=100,
                   prompt_length=100,gen_nfe=45,total_nfe=45)
        for i,b in enumerate(row['blocks']):
            count=[20,10,8,7][i]
            b.update(gen_nfe=count,step_cap=132 if i==0 else 32,cache_enabled=i>0,
                     cache_build_nfe=1 if i>0 else 0,cache_reuse_nfe=count-1 if i>0 else 0,
                     cache_prefix_length=100+32*i if i>0 else 0,full_input_tokens=228,
                     cached_input_tokens=32 if i>0 else 0,cache_total_tokens=228 if i>0 else 0,
                     cache_suffix_length=128-32*(i+1) if i>0 else 0,
                     replace_positions=32 if i>0 else 0,dual_replace_nfe=count-1 if i>0 else 0)
    validate(raw)
    raw[0]['blocks'][1]['replace_positions']=31
    with pytest.raises(ValueError,match='Dual cache replacement'):
        validate(raw)
