"""Continue planned Mini judging without replaying known policy-blocked inputs.

Prompts, model, max tokens, and label semantics remain the original ones. Transport
and original-function retries are disabled for this recovery. Blocked inputs remain
unscored, never converted to SAFE/UNSAFE or replaced by another judge's verdict.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import shutil
import sys
from DiffuGuard import eval_asr as ev
from .data import REPO
from .run_asr import write_json
from .evaluate_asr import summarize

VALID={'SAFE_REFUSAL','UNSAFE_COMPLIANCE'}
CONDITIONS=['0','A','B','C','D','E']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def category(x):return (x.get('evaluation_result') or {}).get('response_safety_category')
def payload_key(pair):return hashlib.sha256(json.dumps(pair,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    a=parser.parse_args();root=a.run_root.resolve()
    if not root.is_relative_to(REPO/'RQ2/outputs'):parser.error('run_root must be in RQ2/outputs')
    ev.load_env();mod=ev.load_original(REPO/'DiffuGuard/third_party/DiffuGuard')
    mod.EVALUATION_SYSTEM_PROMPT=ev.system_prompt(mod,'code')
    lock=(REPO/'RQ2/outputs/asr_e/.judge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backup=root/'judging_recovery_backup'
    if not backup.exists():
        backup.mkdir()
        for name in ['batch_status.json','asr_e_status.json','manifest.json']:
            shutil.copyfile(root/name,backup/name)
        for c in CONDITIONS:
            shutil.copyfile(root/f'with_{c}/eval_c1821e66816b.json',backup/f'luna_{c}.json')
    original_hashes={str(root/f'with_{c}/raw.jsonl'):sha(root/f'with_{c}/raw.jsonl') for c in CONDITIONS}
    original_hashes.update({str(root/f'with_{c}/eval_c1821e66816b.json'):sha(backup/f'luna_{c}.json') for c in CONDITIONS})
    for path,value in original_hashes.items():
        if sha(Path(path))!=value:raise ValueError('Original generation/Luna verdict changed')
    ledger_path=root/'api_policy_blocks.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:
        e=json.loads((root/'with_E/generations_for_judge.json').read_text())
        record=next(x for x in e if x['item_id']=='dija-018')
        pair=ev.judge_pair(mod,record,'auto');blocked_hash=payload_key(pair)
        ledger=dict(blocked_payloads={blocked_hash:dict(code='bio_policy',source='existing Luna E/dija-018 HTTP 400; three failed attempts in original log')},
                    policy='Do not replay a blocked payload, including identical content in another planned judge. Preserve all original records; never infer labels.')
        write_json(ledger_path,ledger)
    mutex=threading.Lock()
    state_path=root/'asr_e_status.json';state=json.loads(state_path.read_text())
    state.update(status='running',current_model='gpt-4o-mini-2024-07-18',recovery_pid=os.getpid(),recovery_policy=ledger['policy'])
    state.pop('error_type',None)
    state['models']['gpt-6-luna']['status']='partial'
    batch_path=root/'batch_status.json';batch=json.loads(batch_path.read_text())
    batch.update(status='running',stage='judging',current_condition=None,pid=os.getpid());batch.pop('error_type',None)
    write_json(batch_path,batch);write_json(state_path,state)
    cfg=json.loads((REPO/'RQ2/configs/asr_e_judge_gpt4omini.json').read_text())
    settings={k:cfg[k] for k in ['judge_model','judge_prompt','question','reasoning_effort','max_completion_tokens']}
    model=settings['judge_model'];assert model=='gpt-4o-mini-2024-07-18'
    tag=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    provenance=dict(settings,base_url=os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
                    reasoning_override=os.environ.get('JUDGE_REASONING',''),
                    evaluation_source_sha256=sha(REPO/'DiffuGuard/third_party/DiffuGuard/evaluation.py'))
    client=ev.make_client().with_options(max_retries=0,timeout=120)
    state['models'][model]=dict(status='running',tag=tag,conditions={})
    summaries={}
    def call_once(item):
        pair=ev.judge_pair(mod,item,settings['question']);key=payload_key(pair)
        with mutex:
            if key in ledger['blocked_payloads']:
                return None,dict(status='blocked_payload_not_replayed',code=ledger['blocked_payloads'][key]['code'],payload_sha256=key)
        errors=[]
        class Tracked:
            def parse(self,**kwargs):
                try:return client.beta.chat.completions.parse(**kwargs)
                except Exception as error:
                    body=getattr(error,'body',{}) or {}
                    code=getattr(error,'code',None)
                    if isinstance(body,dict):
                        nested=body.get('error',body)
                        if isinstance(nested,dict):code=nested.get('code') or code
                    info=dict(status='api_error',error_type=type(error).__name__,code=code,http_status=getattr(error,'status_code',None),payload_sha256=key)
                    errors.append(info)
                    if code=='bio_policy':
                        info['status']='blocked_api_policy'
                        with mutex:
                            ledger['blocked_payloads'][key]=dict(code=code,source='Mini HTTP error during unchanged single-attempt scoring')
                            write_json(ledger_path,ledger)
                    raise
        wrapped=SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=Tracked())))
        result=mod.get_evaluation_from_model(wrapped,model,pair,max_retries=1)
        if result and result.get('response_safety_category') in VALID:return result,None
        return None,(errors[-1] if errors else dict(status='invalid_judge_result',payload_sha256=key))
    try:
        for c in CONDITIONS:
            state['current_condition']=c;p=root/f'with_{c}';target=p/f'eval_{tag}.json'
            raw=[json.loads(x) for x in (p/'raw.jsonl').read_text().splitlines() if x.strip()]
            gen=p/'generations_for_judge.json'
            data=json.loads((target if target.exists() else gen).read_text())
            assert len(data)==len(raw)==100
            assert all(x['item_id']==y['item_id'] and x['response']==y['output_text'] and x['output_sha256']==y['output_sha256'] for x,y in zip(data,raw))
            old_marker=json.loads((REPO/f'RQ2/outputs/full/without_{c}_seed0_n100/eval_{tag}.json.run_config.json').read_text())
            if any(old_marker[k]!=v for k,v in provenance.items()):raise ValueError('Settings differ from original Without judge')
            marker=Path(str(target)+'.run_config.json');expected=dict(provenance,source_raw_sha256=sha(p/'raw.jsonl'),source_generations_sha256=sha(gen))
            if marker.exists() and json.loads(marker.read_text())!=expected:raise ValueError('With judge provenance mismatch')
            write_json(marker,expected)
            def checkpoint():
                write_json(target,data)
                s=summarize(raw,data)
                state['models'][model]['conditions'][c]=dict(status='running',safe=s['safe'],unsafe=s['unsafe'],unscored=s['unscored'])
                write_json(state_path,state)
            todo=[]
            for i,item in enumerate(data):
                if category(item) in VALID:continue
                key=payload_key(ev.judge_pair(mod,item,settings['question']))
                if key in ledger['blocked_payloads']:
                    item['scoring_status']=dict(status='blocked_payload_not_replayed',code=ledger['blocked_payloads'][key]['code'],payload_sha256=key)
                else:todo.append((i,item))
            checkpoint();print(f'With {c}: pending API requests={len(todo)}; known blocked={sum(bool(x.get("scoring_status")) for x in data)}',flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures={pool.submit(call_once,item):i for i,item in todo}
                for future in as_completed(futures):
                    i=futures[future];result,error=future.result()
                    if result:
                        data[i]['evaluation_result']=result;data[i].pop('scoring_status',None)
                    else:data[i]['scoring_status']=error
                    checkpoint()
            s=summarize(raw,data);summaries[c]=s
            status='done' if not s['unscored'] else 'partial'
            state['models'][model]['conditions'][c]=dict(status=status,safe=s['safe'],unsafe=s['unsafe'],unscored=s['unscored'])
            write_json(p/f'summary_{tag}.json',dict(s,settings=settings,recovery_policy=ledger['policy']))
            write_json(Path(str(target)+'.judge.json'),dict(judge_model=model,judge_prompt='code',question='auto',reasoning_adapter=None,base_url=provenance['base_url'],evaluated_utc=datetime.now(timezone.utc).isoformat(),items=100,safe=s['safe'],unsafe=s['unsafe'],unscored=s['unscored'],asr=s['asr_e'],recovery=dict(max_retries=1,sdk_max_retries=0,blocked_payloads_not_replayed=True)))
            write_json(state_path,state)
        state['models'][model]['status']='done' if all(not s['unscored'] for s in summaries.values()) else 'partial'
        rows=[]
        for judge,jtag in [('gpt-6-luna','c1821e66816b'),(model,tag)]:
            for c in CONDITIONS:
                p=root/f'with_{c}';raw=[json.loads(x) for x in (p/'raw.jsonl').read_text().splitlines()]
                data=json.loads((p/f'eval_{jtag}.json').read_text());s=summarize(raw,data)
                ids={x['item_id'] for x in data if category(x) in VALID}
                w=REPO/f'RQ2/outputs/full/without_{c}_seed0_n100'
                wr=[json.loads(x) for x in (w/'raw.jsonl').read_text().splitlines() if json.loads(x)['item_id'] in ids]
                wd=[x for x in json.loads((w/f'eval_{jtag}.json').read_text()) if x['item_id'] in ids]
                ws=summarize(wr,wd) if wr else dict(asr_e=None)
                rows.append(dict(judge_model=judge,condition=c,n_input=100,n_scored=s['safe']+s['unsafe'],unscored=s['unscored'],safe=s['safe'],unsafe=s['unsafe'],asr_e_full100=s['asr_e'] if not s['unscored'] else None,asr_e_scored_subset=s['asr_e'],without_asr_e_same_scored_ids=ws['asr_e'],reduction_pp_same_scored_ids=ws['asr_e']-s['asr_e'] if s['asr_e'] is not None else None,asr_k=s['asr_k']))
        for path,value in original_hashes.items():
            if sha(Path(path))!=value:raise ValueError('Original generation or Luna verdict changed')
        missing=sum(x['unscored'] for x in rows);status='done' if not missing else 'partial'
        result=dict(status=status,n_input=1200,n_scored=1200-missing,unscored=missing,rows=rows,blocked_payloads=ledger,
                    definition='ASR-e excludes unscored; full100 is null for incomplete cells. Paired reductions use the same scored IDs in both rows.',original_sha256=original_hashes)
        write_json(root/'asr_e_comparison.json',result)
        with (root/'asr_e_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for ext in ['json','csv']:shutil.copyfile(root/f'asr_e_comparison.{ext}',REPO/f'RQ2/results/{root.name}_asr_e.{ext}')
        state.update(status=status,current_model=None,current_condition=None)
        batch.update(status=status,stage='done' if not missing else 'done_with_unscored',current_condition=None,unscored=missing,finished_utc=datetime.now(timezone.utc).isoformat(),recovery_command='python -m RQ2.resume_with_full_judging '+str(root))
        write_json(batch_path,batch)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        batch.update(status='failed',error_type=type(error).__name__);write_json(batch_path,batch)
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(state_path,state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
