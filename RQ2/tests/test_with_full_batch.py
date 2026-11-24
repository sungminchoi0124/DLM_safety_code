"""Long unattended jobs must preserve failed data and stop before later cells."""
import hashlib
import json
import subprocess
from pathlib import Path
import pytest
from RQ2.run_asr_with_full import ORDER,generate_all,validate_run,MODEL_REVISION
from RQ2.evaluate_asr import summarize


def state():
    return dict(conditions={c:dict(status='pending') for c in ORDER})


def verify_stub(p,c):
    return dict(condition=c,status='done',raw_sha256=c,n_done=100)


def completed(p):
    p.mkdir();(p/'status.json').write_text('{"status":"done"}')


def test_sequential_order_and_resume_skip(tmp_path):
    s=state();seen=[]
    def launch(c,p):seen.append(c);completed(p)
    generate_all(tmp_path,s,lambda:None,launch,verify_stub)
    assert seen==['0','A','D','E','B','C']
    seen.clear();generate_all(tmp_path,s,lambda:None,launch,verify_stub)
    assert not seen


def test_failed_condition_stops_later_cells(tmp_path):
    s=state();seen=[]
    def launch(c,p):
        seen.append(c)
        if c=='D':raise subprocess.CalledProcessError(1,['synthetic'])
        completed(p)
    with pytest.raises(subprocess.CalledProcessError):
        generate_all(tmp_path,s,lambda:None,launch,verify_stub)
    assert seen==['0','A','D'] and s['conditions']['D']['status']=='failed'
    assert not (tmp_path/'with_E').exists()


def test_partial_output_is_archived_before_retry(tmp_path):
    p=tmp_path/'with_0';p.mkdir();(p/'status.json').write_text('{"status":"failed"}')
    (p/'raw.jsonl').write_text('preserve this partial artifact\n')
    s=state()
    generate_all(tmp_path,s,lambda:None,lambda c,p:completed(p),verify_stub)
    archives=list((tmp_path/'failed_attempts').iterdir())
    assert len(archives)==1 and (archives[0]/'raw.jsonl').read_text()=='preserve this partial artifact\n'


def test_changed_completed_source_is_rejected(tmp_path):
    completed(tmp_path/'with_0');s=state();s['conditions']['0']['raw_sha256']='different'
    with pytest.raises(ValueError,match='source changed'):
        generate_all(tmp_path,s,lambda:None,lambda *a:pytest.fail('must not launch'),verify_stub)


@pytest.fixture
def synthetic_full(tmp_path):
    text='A benign synthetic answer.';digest=hashlib.sha256(text.encode()).hexdigest()
    rows=[]
    for i in range(100):
        rows.append(dict(item_id=f'dija-{i:03d}',source_index=i,item_seed=i,seed=0,accel='0',threshold=None,cache='none',
                         gen_length=128,block_length=32,temperature=.5,prompt_masks_initial=36,n_masks=0,
                         fixed_prompt_unchanged=True,gen_nfe=128,baseline_nfe=1,audit_nfe=0,repair_nfe=8,
                         total_nfe=137,measured_forward_calls=137,audit_record=dict(repaired=True),
                         output_text=text,output_sha256=digest,wall_ms=1.0))
    (tmp_path/'status.json').write_text('{"status":"done"}')
    (tmp_path/'config_resolved.json').write_text(json.dumps(dict(run_kind='full',n_items=100,accel='0',combined_commit='d321365a1f2dad28fdda36e1b540b04c0a56b624',model_revision=MODEL_REVISION)))
    s=summarize(rows);s['validation']='passed';(tmp_path/'summary_local.json').write_text(json.dumps(s))
    return tmp_path,rows


def save_raw(path,rows):
    (path/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))


def test_full_validator_accepts_complete_synthetic_records(synthetic_full):
    p,rows=synthetic_full;save_raw(p,rows)
    result=validate_run(p,'0');assert result['n_done']==100 and result['total_nfe']==13700


@pytest.mark.parametrize('defect',['seed','response_hash','count','gen_nfe'])
def test_full_validator_rejects_corrupt_or_wrong_run(synthetic_full,defect):
    p,rows=synthetic_full
    if defect=='seed':rows[0]['item_seed']=999
    if defect=='response_hash':rows[0]['output_text']='different text'
    if defect=='count':rows.pop()
    if defect=='gen_nfe':rows[0]['gen_nfe']=127;rows[0]['total_nfe']=rows[0]['measured_forward_calls']=136
    save_raw(p,rows)
    with pytest.raises(ValueError):validate_run(p,'0')
